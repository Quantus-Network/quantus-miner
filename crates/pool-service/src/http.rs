//! HTTP API for the captcha widget (session/share) and for protected sites
//! (siteverify), plus optional static file serving for the demo page.

use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;

use primitive_types::U512;
use serde::{Deserialize, Serialize};
use warp::Filter;

use crate::rate_limit::SessionIssuerLimiter;
use crate::state::{PoolState, ShareOutcome};

#[derive(Serialize)]
struct SessionResponse {
    session_id: String,
    /// Hex, 32 bytes, no 0x prefix.
    header_hash: String,
    /// Hex U512, no 0x prefix. Solver iterates nonce_start..nonce_end.
    nonce_start: String,
    nonce_end: String,
    /// Share target as hex U512: a share is valid iff hash < share_target.
    /// (Target form avoids the solver needing U512 division.)
    share_target: String,
    /// Expected number of hashes to find a share (= share difficulty), FYI/UX.
    expected_hashes: u64,
    expires_in_secs: u64,
}

#[derive(Deserialize)]
struct ShareRequest {
    session_id: String,
    /// Hex U512 nonce, no 0x prefix.
    nonce: String,
}

#[derive(Serialize)]
struct ShareResponse {
    success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    token: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<&'static str>,
    block_found: bool,
}

/// Request shape mirrors reCAPTCHA/Turnstile: `secret` + `response`.
#[derive(Deserialize)]
struct SiteVerifyRequest {
    secret: String,
    response: String,
}

#[derive(Serialize)]
struct SiteVerifyResponse {
    success: bool,
    #[serde(rename = "challenge_ts", skip_serializing_if = "Option::is_none")]
    challenge_ts: Option<u64>,
    #[serde(rename = "error-codes", skip_serializing_if = "Vec::is_empty")]
    error_codes: Vec<&'static str>,
}

pub async fn serve(
    state: Arc<PoolState>,
    limiter: Arc<SessionIssuerLimiter>,
    addr: SocketAddr,
    serve_dir: Option<PathBuf>,
) {
    let with_state = {
        let state = state.clone();
        warp::any().map(move || state.clone())
    };
    let with_limiter = {
        let limiter = limiter.clone();
        warp::any().map(move || limiter.clone())
    };

    // POST /api/session -> new captcha challenge
    let session = warp::path!("api" / "session")
        .and(warp::post())
        .and(warp::addr::remote())
        .and(with_limiter.clone())
        .and(with_state.clone())
        .map(
            |remote: Option<SocketAddr>, limiter: Arc<SessionIssuerLimiter>, state: Arc<PoolState>| {
                let ip = remote
                    .map(|a| a.ip())
                    .unwrap_or(IpAddr::from([127, 0, 0, 1]));
                if let Err(reason) = limiter.check(ip) {
                    return reply_error(warp::http::StatusCode::TOO_MANY_REQUESTS, reason);
                }
                match state.issue_session() {
                    Err(reason) => reply_error(warp::http::StatusCode::SERVICE_UNAVAILABLE, reason),
                    Ok(s) => {
                        let target = U512::MAX / s.share_difficulty;
                        let expected = s.share_difficulty.min(U512::from(u64::MAX)).as_u64();
                        warp::reply::with_status(
                            warp::reply::json(&SessionResponse {
                                session_id: s.id,
                                header_hash: hex::encode(s.job.header),
                                nonce_start: format!("{:x}", s.nonce_start),
                                nonce_end: format!("{:x}", s.nonce_end),
                                share_target: format!("{:x}", target),
                                expected_hashes: expected,
                                expires_in_secs: crate::state::SESSION_TTL.as_secs(),
                            }),
                            warp::http::StatusCode::OK,
                        )
                    }
                }
            },
        );

    // POST /api/share -> verify a solved nonce, mint a token
    let share = warp::path!("api" / "share")
        .and(warp::post())
        .and(warp::body::content_length_limit(4096))
        .and(warp::body::json())
        .and(with_state.clone())
        .map(|req: ShareRequest, state: Arc<PoolState>| {
            let nonce = match U512::from_str_radix(req.nonce.trim_start_matches("0x"), 16) {
                Ok(n) => n,
                Err(_) => {
                    return warp::reply::with_status(
                        warp::reply::json(&ShareResponse {
                            success: false,
                            token: None,
                            error: Some("malformed_nonce"),
                            block_found: false,
                        }),
                        warp::http::StatusCode::BAD_REQUEST,
                    )
                }
            };
            match state.submit_share(&req.session_id, nonce) {
                ShareOutcome::Accepted { token, block_found } => warp::reply::with_status(
                    warp::reply::json(&ShareResponse {
                        success: true,
                        token: Some(token),
                        error: None,
                        block_found,
                    }),
                    warp::http::StatusCode::OK,
                ),
                ShareOutcome::Rejected(reason) => warp::reply::with_status(
                    warp::reply::json(&ShareResponse {
                        success: false,
                        token: None,
                        error: Some(reason),
                        block_found: false,
                    }),
                    warp::http::StatusCode::FORBIDDEN,
                ),
            }
        });

    // POST /siteverify -> protected site's backend redeems a token
    let siteverify = warp::path!("siteverify")
        .and(warp::post())
        .and(warp::body::content_length_limit(4096))
        // Accept both JSON and form encoding, like the incumbents do.
        .and(
            warp::body::json()
                .or(warp::body::form())
                .unify()
                .map(|req: SiteVerifyRequest| req),
        )
        .and(with_state.clone())
        .map(|req: SiteVerifyRequest, state: Arc<PoolState>| {
            match state.verify_token(&req.secret, &req.response) {
                Ok(ts) => warp::reply::json(&SiteVerifyResponse {
                    success: true,
                    challenge_ts: Some(ts),
                    error_codes: vec![],
                }),
                Err(code) => warp::reply::json(&SiteVerifyResponse {
                    success: false,
                    challenge_ts: None,
                    error_codes: vec![code],
                }),
            }
        });

    // GET /api/stats -> pool counters
    let stats = warp::path!("api" / "stats")
        .and(warp::get())
        .and(with_state.clone())
        .map(|state: Arc<PoolState>| warp::reply::json(&state.stats()));

    let cors = warp::cors()
        .allow_any_origin()
        .allow_methods(vec!["GET", "POST", "OPTIONS"])
        .allow_headers(vec!["content-type"]);

    let api = session.or(share).or(siteverify).or(stats);

    log::info!("HTTP API listening on {}", addr);
    if let Some(dir) = serve_dir {
        log::info!("Serving demo statics from {}", dir.display());
        let statics = warp::get().and(warp::fs::dir(dir));
        warp::serve(api.or(statics).with(cors)).run(addr).await;
    } else {
        warp::serve(api.with(cors)).run(addr).await;
    }
}

fn reply_error(
    status: warp::http::StatusCode,
    error: &'static str,
) -> warp::reply::WithStatus<warp::reply::Json> {
    #[derive(Serialize)]
    struct Err<'a> {
        error: &'a str,
    }
    warp::reply::with_status(warp::reply::json(&Err { error }), status)
}
