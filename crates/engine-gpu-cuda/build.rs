use std::env;
use std::ffi::OsStr;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Build script to compile CUDA kernels (.cu) to PTX when the `cuda` feature is enabled on this crate.
/// - Scans common kernel directories (`src/kernels`, `src/cuda`) for `.cu` files
/// - Uses `nvcc` to compile each `.cu` into a `.ptx` artifact under `$OUT_DIR`
/// - Emits cargo rebuild hints (`rerun-if-changed`) for all discovered `.cu` files
///
/// Configuration (environment variables):
/// - CARGO_FEATURE_CUDA: Automatically set by Cargo when this crate is built with `features = ["cuda"]`
/// - NVCC: Optional explicit path to the `nvcc` binary
/// - CUDA_HOME or CUDA_PATH: If set, used to derive the default `nvcc` path (`$CUDA_HOME/bin/nvcc`)
/// - CUDA_ARCH: Target architecture for PTX generation (default: "sm_70")
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Only do CUDA work if this crate is built with the `cuda` feature.
    // Cargo sets CARGO_FEATURE_<FEATURE_NAME_UPPERCASE> for the current crate.
    let cuda_feature_enabled = env::var_os("CARGO_FEATURE_CUDA").is_some();
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUDA");

    if !cuda_feature_enabled {
        // No CUDA feature on this crate; nothing to do.
        println!("cargo:warning=engine-gpu-cuda built without 'cuda' feature; skipping kernel compilation.");
        return Ok(());
    }

    // Gather kernel sources (.cu) from common locations
    let kernel_roots = [
        Path::new("src").join("kernels"),
        Path::new("src").join("cuda"),
    ];

    let mut cu_files: Vec<PathBuf> = Vec::new();
    for root in kernel_roots.iter() {
        if root.is_dir() {
            scan_cu_files(root, &mut cu_files)?;
        }
    }

    // If there are no kernels yet, that's fine; keep the build green with a helpful message.
    if cu_files.is_empty() {
        println!("cargo:warning=No CUDA kernels (.cu) found under src/kernels or src/cuda; nothing to compile.");
        return Ok(());
    }

    // Emit rebuild hints for kernels and config env vars that affect compilation
    for cu in &cu_files {
        println!("cargo:rerun-if-changed={}", cu.display());
    }
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");

    // Determine nvcc path
    let nvcc = find_nvcc().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "Unable to locate `nvcc`. Set NVCC or CUDA_HOME/CUDA_PATH, or ensure nvcc is in PATH.",
        )
    });

    let nvcc = match nvcc {
        Ok(p) => p,
        Err(e) => {
            eprintln!("cargo:warning={}", e);
            eprintln!("cargo:warning=Skipping CUDA kernel compilation; crate will build but GPU runtime will not have PTX artifacts.");
            return Ok(());
        }
    };

    // Architecture (SM version) can be overridden; default to sm_70 as a reasonable baseline for modern GPUs
    let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_70".to_string());

    // Output directory for PTX artifacts
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    // Also expose the PTX directory to the crate if desired
    println!(
        "cargo:rustc-env=ENGINE_GPU_CUDA_PTX_DIR={}",
        out_dir.display()
    );

    // Compile each .cu into a .ptx
    for cu in &cu_files {
        let stem = cu
            .file_stem()
            .and_then(OsStr::to_str)
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "Invalid kernel filename"))?;

        let ptx_path = out_dir.join(format!("{stem}.ptx"));

        match compile_to_ptx(&nvcc, &arch, cu, &ptx_path) {
            Ok(()) => {
                println!(
                    "cargo:warning=nvcc OK: {} -> {}",
                    cu.display(),
                    ptx_path.display()
                );
            }
            Err(err) => {
                eprintln!("cargo:warning=nvcc failed for {}: {err}", cu.display());
                eprintln!(
                    "cargo:warning=Skipping this kernel; build continues without PTX for {}",
                    stem
                );
            }
        }
    }

    Ok(())
}

fn scan_cu_files(root: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            scan_cu_files(&path, out)?;
        } else if path.extension().and_then(OsStr::to_str) == Some("cu") {
            out.push(path);
        }
    }
    Ok(())
}

fn find_nvcc() -> Option<PathBuf> {
    // 1) Explicit override
    if let Some(p) = env::var_os("NVCC") {
        let pb = PathBuf::from(p);
        if pb.is_file() {
            return Some(pb);
        }
    }

    // 2) CUDA_HOME or CUDA_PATH
    if let Some(home) = env::var_os("CUDA_HOME").or_else(|| env::var_os("CUDA_PATH")) {
        let mut candidate = PathBuf::from(home);
        candidate.push("bin");
        candidate.push(exe("nvcc"));
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    // 3) PATH lookup — try invoking `nvcc --version`
    if let Ok(output) = Command::new("nvcc").arg("--version").output() {
        if output.status.success() {
            return Some(PathBuf::from("nvcc"));
        }
    }

    None
}

fn compile_to_ptx(nvcc: &Path, arch: &str, src: &Path, out: &Path) -> io::Result<()> {
    // Ensure parent directory exists
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }

    // Example command:
    //   nvcc -ptx -arch=sm_70 -o <OUT_DIR>/kernel.ptx <src.cu>
    let status = Command::new(nvcc)
        .args(["-ptx", "-arch", arch])
        .arg("-o")
        .arg(out)
        .arg(src)
        .status()?;

    if !status.success() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!("nvcc returned non-zero exit status: {status}"),
        ));
    }

    Ok(())
}

#[cfg(target_os = "windows")]
fn exe(name: &str) -> String {
    format!("{name}.exe")
}

#[cfg(not(target_os = "windows"))]
fn exe(name: &str) -> String {
    name.to_string()
}
