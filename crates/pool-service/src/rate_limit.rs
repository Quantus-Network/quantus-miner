//! Per-client rate limiting for unauthenticated endpoints.
//!
//! `/api/session` is free, so a hard map cap alone still lets an attacker
//! churn entries at the TTL boundary. A per-IP issuance ceiling bounds that
//! even when the global cap has headroom.

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Mutex;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy)]
pub struct SessionRateLimit {
    /// Max session issuances per IP per window.
    pub max_per_ip: u32,
    pub window: Duration,
}

impl Default for SessionRateLimit {
    fn default() -> Self {
        SessionRateLimit {
            // 60 sessions/min/IP is generous for real users (a page load is
            // one) but stops a single host from dominating issuance.
            max_per_ip: 60,
            window: Duration::from_secs(60),
        }
    }
}

#[derive(Default)]
struct IpWindow {
    count: u32,
    window_start: Option<Instant>,
}

pub struct SessionIssuerLimiter {
    cfg: SessionRateLimit,
    by_ip: Mutex<HashMap<IpAddr, IpWindow>>,
}

impl SessionIssuerLimiter {
    pub fn new(cfg: SessionRateLimit) -> Self {
        SessionIssuerLimiter {
            cfg,
            by_ip: Mutex::new(HashMap::new()),
        }
    }

    /// Returns Ok(()) if this IP may issue another session, Err otherwise.
    pub fn check(&self, ip: IpAddr) -> Result<(), &'static str> {
        let now = Instant::now();
        let mut map = self.by_ip.lock().unwrap();
        let entry = map.entry(ip).or_default();
        let window_start = entry.window_start.get_or_insert_with(Instant::now);
        if now.duration_since(*window_start) >= self.cfg.window {
            *window_start = now;
            entry.count = 0;
        }
        if entry.count >= self.cfg.max_per_ip {
            return Err("rate_limited");
        }
        entry.count += 1;
        Ok(())
    }

    /// Drop stale IP windows so the side table doesn't grow forever.
    pub fn gc(&self) {
        let cutoff = self.cfg.window * 2;
        let now = Instant::now();
        self.by_ip.lock().unwrap().retain(|_, w| {
            w.window_start
                .map(|t| now.duration_since(t) < cutoff)
                .unwrap_or(true)
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{Ipv4Addr, Ipv6Addr};

    #[test]
    fn allows_up_to_cap_then_rejects() {
        let lim = SessionIssuerLimiter::new(SessionRateLimit {
            max_per_ip: 2,
            window: Duration::from_secs(60),
        });
        let ip = IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1));
        assert!(lim.check(ip).is_ok());
        assert!(lim.check(ip).is_ok());
        assert_eq!(lim.check(ip), Err("rate_limited"));
    }

    #[test]
    fn ips_are_independent() {
        let lim = SessionIssuerLimiter::new(SessionRateLimit {
            max_per_ip: 1,
            window: Duration::from_secs(60),
        });
        assert!(lim.check(IpAddr::V4(Ipv4Addr::new(1, 2, 3, 4))).is_ok());
        assert!(lim.check(IpAddr::V6(Ipv6Addr::LOCALHOST)).is_ok());
    }
}
