use std::env;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

fn main() {
    // Always re-run if version env or git state might change
    println!("cargo:rerun-if-env-changed=CARGO_PKG_VERSION");
    println!("cargo:rerun-if-env-changed=GITHUB_SHA");
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/refs");
    println!("cargo:rerun-if-changed=.git/packed-refs");

    let pkg_version = env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.0.0".to_string());
    let short_sha = env::var("GITHUB_SHA")
        .ok()
        .and_then(|s| s.get(0..8).map(|t| t.to_string()))
        .or_else(|| discover_short_git_sha().ok());

    let version = match short_sha {
        Some(sha) if !sha.is_empty() => format!("{pkg_version}+{sha}"),
        _ => pkg_version.clone(),
    };

    // Expose to the crate as an env! binding
    println!("cargo:rustc-env=MINER_VERSION={version}");

    // Optional extra info for diagnostics
    if let Some(sha) = env::var("GITHUB_SHA").ok() {
        println!("cargo:rustc-env=MINER_BUILD_SHA={}", sha);
    }

    // Helpful build log
    println!("cargo:warning=miner-cli: computed MINER_VERSION={version}");
}

/// Attempts to determine the short git SHA (first 8 hex chars) without requiring the `git` binary.
/// Handles both normal repositories and worktrees where `.git` is a file pointing to the real gitdir.
fn discover_short_git_sha() -> io::Result<String> {
    let git_dir = find_git_dir()?;
    let head_path = git_dir.join("HEAD");
    let head = fs::read_to_string(&head_path)?.trim().to_string();

    // HEAD can be either a direct SHA or a ref: refs/heads/branch
    if let Some(ref_line) = head.strip_prefix("ref: ") {
        let ref_path = git_dir.join(ref_line.trim());
        if ref_path.is_file() {
            let mut contents = String::new();
            fs::File::open(ref_path)?.read_to_string(&mut contents)?;
            let sha = contents.trim();
            return Ok(shorten_sha(sha));
        }
        // Not a loose ref; try packed-refs
        let packed = git_dir.join("packed-refs");
        if packed.is_file() {
            let contents = fs::read_to_string(packed)?;
            for line in contents.lines() {
                // Skip comments and peeled lines
                if line.starts_with('#') || line.starts_with('^') || line.trim().is_empty() {
                    continue;
                }
                // Format: <sha> <ref>
                if let Some((sha, r)) = line.split_once(' ') {
                    if r.trim() == ref_line.trim() {
                        return Ok(shorten_sha(sha.trim()));
                    }
                }
            }
        }
        Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "Could not resolve ref '{}' in {:?}",
                ref_line.trim(),
                git_dir
            ),
        ))
    } else {
        // HEAD contained a SHA directly (e.g., detached HEAD)
        Ok(shorten_sha(head.trim()))
    }
}

/// Finds the actual .git directory, handling both:
/// - Standard repo: a ".git" directory exists, or
/// - Worktree/submodule: ".git" is a file containing "gitdir: <path>"
fn find_git_dir() -> io::Result<PathBuf> {
    let mut dir = env::current_dir()?;
    loop {
        let dot_git = dir.join(".git");
        if dot_git.is_dir() {
            return Ok(dot_git);
        } else if dot_git.is_file() {
            // Read the pointer file
            let contents = fs::read_to_string(&dot_git)?;
            if let Some(p) = contents.strip_prefix("gitdir:") {
                let p = p.trim();
                let path = Path::new(p);
                // If relative, it's relative to the location of the .git file's parent
                let resolved = if path.is_absolute() {
                    path.to_path_buf()
                } else {
                    dir.join(path)
                };
                if resolved.is_dir() {
                    return Ok(resolved);
                } else {
                    return Err(io::Error::new(
                        io::ErrorKind::NotFound,
                        format!("Resolved gitdir does not exist: {}", resolved.display()),
                    ));
                }
            }
        }
        // Ascend until root
        if !dir.pop() {
            break;
        }
    }

    Err(io::Error::new(
        io::ErrorKind::NotFound,
        "Could not locate .git directory",
    ))
}

fn shorten_sha(full: &str) -> String {
    let s = full.trim();
    if s.len() >= 8 {
        s[..8].to_string()
    } else {
        s.to_string()
    }
}
