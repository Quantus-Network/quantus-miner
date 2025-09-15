use std::env;
use std::path::Path;
use std::process::Command;

/// Build script for miner-cli:
/// - Computes short git SHA (optionally with "-dirty") if available
/// - Appends it to the semantic version
/// - Injects QUANTUS_VERSION and overrides CARGO_PKG_VERSION at compile time so `--version` includes the SHA
fn main() {
    // Rebuild the crate when git metadata changes (best-effort; ignored if not a git repo).
    println!("cargo:rerun-if-env-changed=QUANTUS_CI_GIT_SHA");
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/refs/heads");
    println!("cargo:rerun-if-changed=.git/packed-refs");

    let pkg_ver = env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.0.0".to_string());

    // Allow CI to inject a SHA explicitly; otherwise discover via git.
    let sha = env::var("QUANTUS_CI_GIT_SHA")
        .ok()
        .or_else(git_sha_short)
        .unwrap_or_default();

    // Compose the display version. If no SHA is available, keep the original pkg version.
    let display_version = if sha.is_empty() {
        pkg_ver.clone()
    } else {
        // Use semver build metadata format: 1.2.3+abc123[-dirty]
        format!("{pkg_ver}+{sha}")
    };

    // Expose both:
    // - QUANTUS_VERSION: for explicit usage via env!("QUANTUS_VERSION")
    // - CARGO_PKG_VERSION: override so Clap's #[command(version)] picks up the augmented version
    println!("cargo:rustc-env=QUANTUS_VERSION={}", display_version);
    println!("cargo:rustc-env=CARGO_PKG_VERSION={}", display_version);
}

/// Try to get a short git SHA (12 chars) and append "-dirty" if the working tree is modified.
/// Returns None if git is unavailable or this isn't a git checkout.
fn git_sha_short() -> Option<String> {
    // Quick existence check to avoid spawning git in non-repos.
    if !Path::new(".git").exists() {
        return None;
    }

    // Obtain short SHA
    let sha = Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .and_then(|out| {
            if out.status.success() {
                let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if s.is_empty() {
                    None
                } else {
                    Some(s)
                }
            } else {
                None
            }
        })?;

    // Detect dirty working tree (best-effort)
    let dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .map(|out| !out.stdout.is_empty())
        .unwrap_or(false);

    if dirty {
        Some(format!("{sha}-dirty"))
    } else {
        Some(sha)
    }
}
