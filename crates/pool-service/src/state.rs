//! Core pool state: the current mining job, captcha sessions with disjoint
//! nonce ranges, and single-use share tokens.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use primitive_types::U512;
use rand::RngCore;

/// How long a captcha session stays solvable after issuance.
pub const SESSION_TTL: Duration = Duration::from_secs(120);
/// How long a share token stays redeemable via /siteverify.
pub const TOKEN_TTL: Duration = Duration::from_secs(300);
/// Each session gets a disjoint slice of the nonce space this many nonces wide.
/// 2^64 nonces is unreachable within a session TTL at browser hash rates.
pub const SESSION_RANGE_BITS: u32 = 64;

/// Caps on live entries in the in-memory maps. `/api/session` is free and
/// unauthenticated, so without a ceiling a request flood becomes a memory-DoS
/// on the anti-abuse service itself. At the ~300 bytes/entry ballpark the
/// defaults bound each map to tens of MB.
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    pub max_sessions: usize,
    pub max_tokens: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            max_sessions: 100_000,
            max_tokens: 100_000,
        }
    }
}

/// The job the pool is currently handing out to captcha sessions.
#[derive(Debug, Clone)]
pub struct Job {
    /// Job id assigned by the upstream node (or synthesized in standalone mode).
    pub job_id: String,
    /// Block header hash to mine over.
    pub header: [u8; 32],
    /// Full network difficulty for this job (block found if a share meets it).
    pub network_difficulty: U512,
}

/// A captcha session: one issued challenge, bound to a disjoint nonce range.
#[derive(Debug, Clone)]
pub struct Session {
    pub id: String,
    /// Snapshot of the job at issuance time. Shares are verified against this
    /// header even if the pool has since moved to a newer job (grace period =
    /// session TTL); only network-difficulty solutions for the *current* job
    /// are submitted upstream.
    pub job: Job,
    pub nonce_start: U512,
    pub nonce_end: U512, // exclusive
    pub share_difficulty: U512,
    pub issued_at: Instant,
    pub solved: bool,
}

/// A verified share, redeemable exactly once by the protected site's backend.
#[derive(Debug, Clone)]
pub struct ShareToken {
    pub created_at: Instant,
    pub created_unix: u64,
    pub consumed: bool,
}

#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct PoolStats {
    pub sessions_issued: u64,
    pub shares_accepted: u64,
    pub shares_rejected: u64,
    pub tokens_verified: u64,
    pub blocks_found: u64,
}

pub struct PoolState {
    /// Current job, None until the first job arrives from the job source.
    current_job: Mutex<Option<Job>>,
    sessions: Mutex<HashMap<String, Session>>,
    tokens: Mutex<HashMap<String, ShareToken>>,
    stats: Mutex<PoolStats>,
    /// Monotonic counter used to carve disjoint nonce ranges out of U512 space.
    range_counter: AtomicU64,
    /// Difficulty for captcha shares (expected hashes per solve).
    pub share_difficulty: U512,
    /// Site secret checked by /siteverify.
    ///
    /// NOTE: single-tenant by design for now. All tokens are interchangeable
    /// across any backend holding this secret; per-sitekey secrets land with
    /// the host-registration follow-up.
    pub site_secret: String,
    limits: Limits,
    /// Sender used to push network-difficulty solutions upstream.
    pub solution_tx: tokio::sync::mpsc::Sender<FoundBlock>,
}

/// A share that met full network difficulty: a real block solution.
#[derive(Debug, Clone)]
pub struct FoundBlock {
    pub job_id: String,
    pub nonce: U512,
}

pub enum ShareOutcome {
    /// Share accepted; token the client can hand to the protected site.
    Accepted {
        token: String,
        block_found: bool,
    },
    Rejected(&'static str),
}

fn random_id() -> String {
    let mut bytes = [0u8; 24];
    rand::rng().fill_bytes(&mut bytes);
    hex::encode(bytes)
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

impl PoolState {
    pub fn new(
        share_difficulty: U512,
        site_secret: String,
        solution_tx: tokio::sync::mpsc::Sender<FoundBlock>,
        limits: Limits,
    ) -> Arc<Self> {
        assert!(
            !share_difficulty.is_zero(),
            "share difficulty must be non-zero"
        );
        Arc::new(PoolState {
            current_job: Mutex::new(None),
            sessions: Mutex::new(HashMap::new()),
            tokens: Mutex::new(HashMap::new()),
            stats: Mutex::new(PoolStats::default()),
            range_counter: AtomicU64::new(0),
            share_difficulty,
            site_secret,
            limits,
            solution_tx,
        })
    }

    /// Install a new job (from the upstream node or the standalone generator).
    /// Existing sessions keep their job snapshot and remain solvable until TTL.
    pub fn set_job(&self, job: Job) {
        log::info!(
            "New job {}: header=0x{} network_difficulty={}",
            job.job_id,
            hex::encode(job.header),
            pow_core::format_u512(job.network_difficulty)
        );
        *self.current_job.lock().unwrap() = Some(job);
    }

    pub fn current_job(&self) -> Option<Job> {
        self.current_job.lock().unwrap().clone()
    }

    /// Issue a new captcha session with a disjoint nonce range.
    pub fn issue_session(&self) -> Result<Session, &'static str> {
        let job = self.current_job().ok_or("no_job_available")?;

        if self.sessions.lock().unwrap().len() >= self.limits.max_sessions {
            return Err("at_capacity");
        }

        // Carve the next disjoint 2^64-nonce slice. The counter is 64 bits and
        // slices are 2^64 wide, so slices live in the bottom 2^128 of the U512
        // nonce space and never collide with each other.
        let index = self.range_counter.fetch_add(1, Ordering::Relaxed);
        let nonce_start = U512::from(index) << SESSION_RANGE_BITS;
        let nonce_end = nonce_start + (U512::one() << SESSION_RANGE_BITS);

        let session = Session {
            id: random_id(),
            job,
            nonce_start,
            nonce_end,
            share_difficulty: self.share_difficulty,
            issued_at: Instant::now(),
            solved: false,
        };

        self.sessions
            .lock()
            .unwrap()
            .insert(session.id.clone(), session.clone());
        self.stats.lock().unwrap().sessions_issued += 1;
        Ok(session)
    }

    /// Verify a submitted share and, if valid, mint a single-use token.
    pub fn submit_share(&self, session_id: &str, nonce: U512) -> ShareOutcome {
        let session = {
            let mut sessions = self.sessions.lock().unwrap();
            match sessions.get_mut(session_id) {
                None => return self.reject("unknown_session"),
                Some(s) => {
                    if s.solved {
                        return self.reject("session_already_solved");
                    }
                    if s.issued_at.elapsed() > SESSION_TTL {
                        sessions.remove(session_id);
                        return self.reject("session_expired");
                    }
                    // Mark solved optimistically; unmark on verification failure
                    // would allow retries, but a failed share means a buggy or
                    // malicious client, so we burn the session either way.
                    s.solved = true;
                    s.clone()
                }
            }
        };

        if nonce < session.nonce_start || nonce >= session.nonce_end {
            return self.reject("nonce_out_of_range");
        }

        let (share_valid, hash) = pow_core::is_valid_nonce(
            session.job.header,
            nonce.to_big_endian(),
            session.share_difficulty,
        );
        if !share_valid {
            return self.reject("share_below_target");
        }

        // Jackpot check: does this share meet full network difficulty for the
        // job it was issued against? Only submit if that job is still current.
        // Target derivation delegates to pow-core's canonical JobContext.
        let network_target =
            pow_core::JobContext::new(session.job.header, session.job.network_difficulty).target;
        let mut block_found = false;
        if hash < network_target {
            let still_current = self
                .current_job()
                .map(|j| j.job_id == session.job.job_id)
                .unwrap_or(false);
            if still_current {
                log::info!(
                    "BLOCK FOUND by captcha share! job={} nonce={:x}",
                    session.job.job_id,
                    nonce
                );
                // This is the revenue event: count it and be loud if the
                // upstream queue can't take it.
                match self.solution_tx.try_send(FoundBlock {
                    job_id: session.job.job_id.clone(),
                    nonce,
                }) {
                    Ok(()) => {
                        block_found = true;
                        self.stats.lock().unwrap().blocks_found += 1;
                    }
                    Err(e) => {
                        log::error!(
                            "DROPPED block solution for job {}: upstream queue unavailable ({})",
                            session.job.job_id,
                            e
                        );
                    }
                }
            }
        }

        let token = random_id();
        {
            let mut tokens = self.tokens.lock().unwrap();
            // A full token map means someone is farming shares faster than
            // TTL expiry; refuse rather than grow without bound. The client
            // paid hashes for this, so make the refusal explicit.
            if tokens.len() >= self.limits.max_tokens {
                return self.reject("at_capacity");
            }
            tokens.insert(
                token.clone(),
                ShareToken {
                    created_at: Instant::now(),
                    created_unix: unix_now(),
                    consumed: false,
                },
            );
        }
        self.stats.lock().unwrap().shares_accepted += 1;
        ShareOutcome::Accepted { token, block_found }
    }

    /// Redeem a share token (the reCAPTCHA/Turnstile "siteverify" semantics):
    /// valid exactly once, only with the correct site secret.
    pub fn verify_token(&self, secret: &str, token: &str) -> Result<u64, &'static str> {
        if !constant_time_eq::constant_time_eq(secret.as_bytes(), self.site_secret.as_bytes()) {
            return Err("invalid_secret");
        }
        let mut tokens = self.tokens.lock().unwrap();
        match tokens.get_mut(token) {
            None => Err("invalid_token"),
            Some(t) if t.consumed => Err("token_already_consumed"),
            Some(t) if t.created_at.elapsed() > TOKEN_TTL => {
                tokens.remove(token);
                Err("token_expired")
            }
            Some(t) => {
                t.consumed = true;
                let ts = t.created_unix;
                self.stats.lock().unwrap().tokens_verified += 1;
                Ok(ts)
            }
        }
    }

    pub fn stats(&self) -> PoolStats {
        self.stats.lock().unwrap().clone()
    }

    /// Drop expired sessions and tokens. Called periodically.
    pub fn gc(&self) {
        self.sessions
            .lock()
            .unwrap()
            .retain(|_, s| s.issued_at.elapsed() <= SESSION_TTL);
        self.tokens
            .lock()
            .unwrap()
            .retain(|_, t| t.created_at.elapsed() <= TOKEN_TTL);
    }

    fn reject(&self, reason: &'static str) -> ShareOutcome {
        self.stats.lock().unwrap().shares_rejected += 1;
        ShareOutcome::Rejected(reason)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_state(
        share_difficulty: u64,
    ) -> (Arc<PoolState>, tokio::sync::mpsc::Receiver<FoundBlock>) {
        let (tx, rx) = tokio::sync::mpsc::channel(4);
        let state = PoolState::new(
            U512::from(share_difficulty),
            "secret".into(),
            tx,
            Limits::default(),
        );
        state.set_job(Job {
            job_id: "1".into(),
            header: [7u8; 32],
            network_difficulty: U512::MAX, // effectively unreachable
        });
        (state, rx)
    }

    /// Brute-force a valid share within the session's range.
    fn solve(session: &Session) -> U512 {
        let mut nonce = session.nonce_start;
        loop {
            let (ok, _) = pow_core::is_valid_nonce(
                session.job.header,
                nonce.to_big_endian(),
                session.share_difficulty,
            );
            if ok {
                return nonce;
            }
            nonce += U512::one();
        }
    }

    #[test]
    fn accepts_valid_share_and_verifies_token_once() {
        let (state, _rx) = test_state(2); // ~2 hashes per solve
        let session = state.issue_session().unwrap();
        let nonce = solve(&session);

        let token = match state.submit_share(&session.id, nonce) {
            ShareOutcome::Accepted { token, block_found } => {
                assert!(!block_found);
                token
            }
            ShareOutcome::Rejected(r) => panic!("rejected: {r}"),
        };

        assert!(state.verify_token("secret", &token).is_ok());
        assert_eq!(
            state.verify_token("secret", &token),
            Err("token_already_consumed")
        );
        assert_eq!(state.verify_token("wrong", &token), Err("invalid_secret"));
    }

    #[test]
    fn rejects_out_of_range_nonce() {
        let (state, _rx) = test_state(1); // difficulty 1: every nonce is a valid hash
        let session = state.issue_session().unwrap();
        // A nonce outside the assigned slice must be rejected even though its
        // hash trivially meets the share target.
        let outside = session.nonce_end;
        match state.submit_share(&session.id, outside) {
            ShareOutcome::Rejected("nonce_out_of_range") => {}
            other => panic!(
                "expected out-of-range rejection, got {:?}",
                matches_str(&other)
            ),
        }
    }

    #[test]
    fn burns_session_after_one_submission() {
        let (state, _rx) = test_state(1);
        let session = state.issue_session().unwrap();
        let nonce = session.nonce_start;
        assert!(matches!(
            state.submit_share(&session.id, nonce),
            ShareOutcome::Accepted { .. }
        ));
        match state.submit_share(&session.id, nonce) {
            ShareOutcome::Rejected("session_already_solved") => {}
            other => panic!("expected burned session, got {:?}", matches_str(&other)),
        }
    }

    #[test]
    fn disjoint_ranges() {
        let (state, _rx) = test_state(1);
        let a = state.issue_session().unwrap();
        let b = state.issue_session().unwrap();
        assert!(a.nonce_end <= b.nonce_start || b.nonce_end <= a.nonce_start);
    }

    #[test]
    fn block_found_signalled_when_share_meets_network_difficulty() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        // Network difficulty == share difficulty == 2: any valid share is a block.
        let state = PoolState::new(U512::from(2u64), "secret".into(), tx, Limits::default());
        state.set_job(Job {
            job_id: "1".into(),
            header: [9u8; 32],
            network_difficulty: U512::from(2u64),
        });
        let session = state.issue_session().unwrap();
        let nonce = solve(&session);
        match state.submit_share(&session.id, nonce) {
            ShareOutcome::Accepted { block_found, .. } => assert!(block_found),
            ShareOutcome::Rejected(r) => panic!("rejected: {r}"),
        }
        let block = rx
            .try_recv()
            .expect("block solution should be queued upstream");
        assert_eq!(block.job_id, "1");
        assert_eq!(block.nonce, nonce);
    }

    #[test]
    fn session_cap_rejects_flood() {
        let (tx, _rx) = tokio::sync::mpsc::channel(4);
        let state = PoolState::new(
            U512::from(1u64),
            "secret".into(),
            tx,
            Limits {
                max_sessions: 2,
                max_tokens: 2,
            },
        );
        state.set_job(Job {
            job_id: "1".into(),
            header: [7u8; 32],
            network_difficulty: U512::MAX,
        });
        assert!(state.issue_session().is_ok());
        assert!(state.issue_session().is_ok());
        assert!(matches!(state.issue_session(), Err("at_capacity")));
    }

    #[test]
    fn dropped_block_does_not_count_or_flag() {
        // Zero-capacity channel: try_send always fails, simulating a stalled
        // upstream. The share must still be accepted (the user did the work),
        // but block_found must be false and blocks_found must stay 0.
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        drop(rx); // closed channel -> try_send errors
        let state = PoolState::new(U512::from(2u64), "secret".into(), tx, Limits::default());
        state.set_job(Job {
            job_id: "1".into(),
            header: [9u8; 32],
            network_difficulty: U512::from(2u64),
        });
        let session = state.issue_session().unwrap();
        let nonce = solve(&session);
        match state.submit_share(&session.id, nonce) {
            ShareOutcome::Accepted { block_found, .. } => assert!(!block_found),
            ShareOutcome::Rejected(r) => panic!("rejected: {r}"),
        }
        assert_eq!(state.stats().blocks_found, 0);
    }

    fn matches_str(o: &ShareOutcome) -> &'static str {
        match o {
            ShareOutcome::Accepted { .. } => "accepted",
            ShareOutcome::Rejected(r) => r,
        }
    }
}
