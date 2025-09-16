use std::env;
use std::ffi::OsStr;
use std::fs;
use std::io::{self};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Return true when the host GCC major version is greater than 14.
/// We try $CC --version first, then gcc --version, and parse the first X.Y-like token.
fn host_gcc_too_new() -> bool {
    fn probe(bin: &str) -> Option<String> {
        Command::new(bin)
            .arg("--version")
            .stdout(Stdio::piped())
            .output()
            .ok()
            .and_then(|o| {
                if o.status.success() {
                    Some(String::from_utf8_lossy(&o.stdout).into_owned())
                } else {
                    None
                }
            })
    }

    let cc = env::var("CC").unwrap_or_else(|_| "gcc".to_string());
    let ver_out = probe(&cc).or_else(|| probe("gcc"));
    if let Some(s) = ver_out {
        // Scan tokens separated by non [0-9|.] and take the first X.Y...
        for tok in s.split(|c: char| !c.is_ascii_digit() && c != '.') {
            if tok.is_empty() {
                continue;
            }
            if let Some((maj, _)) = tok.split_once('.') {
                if let Ok(m) = maj.parse::<u32>() {
                    return m > 14;
                }
            }
        }
    }
    false
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Only do CUDA work if this crate is built with the `cuda` feature.
    let cuda_feature_enabled = env::var_os("CARGO_FEATURE_CUDA").is_some();
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUDA");

    if !cuda_feature_enabled {
        // Generate empty bindings so include! compiles; skip CUDA work.
        let out_dir = PathBuf::from(env::var("OUT_DIR")?);
        generate_empty_bindings(&out_dir)?;
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

    // If there are no kernels yet, still generate an empty bindings file so include! compiles.
    if cu_files.is_empty() {
        let out_dir = PathBuf::from(env::var("OUT_DIR")?);
        generate_empty_bindings(&out_dir)?;
        println!("cargo:warning=No CUDA kernels (.cu) found under src/kernels or src/cuda; generated empty PTX/CUBIN bindings.");
        return Ok(());
    }

    // Emit rebuild hints for kernels and env vars that affect compilation
    for cu in &cu_files {
        println!("cargo:rerun-if-changed={}", cu.display());
    }
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=MINER_CUDA_ALLOW_UNSUPPORTED_COMPILER");
    println!("cargo:rerun-if-env-changed=MINER_NVCC_CCBIN");

    // Determine nvcc path
    let nvcc = find_nvcc().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "Unable to locate `nvcc`. Set NVCC or CUDA_HOME/CUDA_PATH, or ensure nvcc is in PATH.",
        )
    })?;

    // Architecture (SM version) can be overridden; default to sm_70 as a reasonable baseline for modern GPUs
    let raw_arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_70".to_string());
    // Normalize to a compute/sm pair:
    // - Use compute_* for -arch
    // - Use sm_* for -code
    let (arch_compute, arch_sm) = if let Some(s) = raw_arch.strip_prefix("sm_") {
        (format!("compute_{s}"), raw_arch.clone())
    } else if let Some(s) = raw_arch.strip_prefix("compute_") {
        (raw_arch.clone(), format!("sm_{s}"))
    } else if let Some(s) = raw_arch.strip_prefix("sm") {
        (format!("compute{s}"), raw_arch.clone())
    } else {
        ("compute_70".to_string(), "sm_70".to_string())
    };
    println!("cargo:warning=NVCC={}", nvcc.display());
    println!("cargo:warning=CUDA_ARCH (normalized): compute={arch_compute}, sm={arch_sm}");

    // Output directory for artifacts
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    println!(
        "cargo:rustc-env=ENGINE_GPU_CUDA_PTX_DIR={}",
        out_dir.display()
    );

    // Preflight: locate CUDA include directories and ensure cuda_runtime.h exists.
    // Prefer $CUDA_HOME/targets/x86_64-linux/include, then $CUDA_HOME/include. Derive CUDA_HOME from NVCC if needed.
    let cuda_home_env = env::var_os("CUDA_HOME").map(PathBuf::from).or_else(|| {
        // Derive CUDA_HOME from NVCC path (<home>/bin/nvcc)
        nvcc.parent()
            .and_then(Path::parent)
            .map(|p| p.to_path_buf())
    });

    let mut include_dirs: Vec<PathBuf> = Vec::new();

    // If CUDA_HOME was found (either env or derived from NVCC), add its common include locations.
    if let Some(home) = cuda_home_env.as_ref() {
        let t_inc = home.join("targets").join("x86_64-linux").join("include");
        let i_inc = home.join("include");
        if t_inc.is_dir() {
            include_dirs.push(t_inc);
        }
        if i_inc.is_dir() {
            include_dirs.push(i_inc);
        }
    }

    // Fallback: common system install prefix in CUDA devel images.
    // This allows builds to succeed even when CUDA_HOME isn't set in the container.
    for fallback in [
        Path::new("/usr/local/cuda/targets/x86_64-linux/include"),
        Path::new("/usr/local/cuda/include"),
    ] {
        if fallback.is_dir() {
            include_dirs.push(fallback.to_path_buf());
        }
    }

    // Deduplicate include dirs to avoid noisy logs and duplicate -I flags
    include_dirs.sort();
    include_dirs.dedup();

    for inc in &include_dirs {
        println!("cargo:warning=CUDA_INCLUDE_DIR={}", inc.display());
    }

    let has_runtime_h = include_dirs
        .iter()
        .any(|d| d.join("cuda_runtime.h").is_file());
    if !has_runtime_h {
        return Err(Box::new(io::Error::other(format!(
            "engine-gpu-cuda: missing CUDA headers (cuda_runtime.h). \
                 Ensure the matching CUDA toolkit is installed (see README), \
                 and CUDA_HOME is set (current CUDA_HOME={:?}). Searched: {}",
            cuda_home_env,
            include_dirs
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ))));
    }

    // Extra nvcc options and optional host compiler override
    let mut extra_nvcc_flags: Vec<String> = Vec::new();
    let allow_unsupported = env::var("MINER_CUDA_ALLOW_UNSUPPORTED_COMPILER")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if allow_unsupported || host_gcc_too_new() {
        println!("cargo:warning=nvcc: enabling -allow-unsupported-compiler");
        extra_nvcc_flags.push("-allow-unsupported-compiler".to_string());
    }
    let ccbin = env::var("MINER_NVCC_CCBIN").ok();
    if let Some(ref cc) = ccbin {
        println!("cargo:warning=nvcc: using -ccbin {cc}");
    }

    // Compile each .cu into artifacts
    // - PTX: for driver JIT (fallback)
    // - CUBIN: native SASS per-SM (preferred; avoids PTX JIT/ISA mismatches)
    let mut generated_ptx: Vec<(String, PathBuf)> = Vec::new();
    let mut generated_cubin: Vec<(String, PathBuf)> = Vec::new();

    for cu in &cu_files {
        let stem = cu
            .file_stem()
            .and_then(OsStr::to_str)
            .ok_or_else(|| io::Error::other("Invalid kernel filename"))?
            .to_string();

        let ptx_path = out_dir.join(format!("{stem}.ptx"));
        let cubin_path = out_dir.join(format!("{stem}.cubin"));

        // PTX
        match compile_to_ptx(
            &nvcc,
            &arch_compute,
            cu,
            &ptx_path,
            &include_dirs,
            ccbin.as_deref(),
            &extra_nvcc_flags,
        ) {
            Ok(()) => {
                println!(
                    "cargo:warning=nvcc PTX OK: {} -> {}",
                    cu.display(),
                    ptx_path.display()
                );
                generated_ptx.push((stem.clone(), ptx_path.clone()));
            }
            Err(err) => {
                eprintln!("cargo:warning=nvcc PTX failed for {}: {err}", cu.display());
                eprintln!(
                    "cargo:warning=Skipping PTX for {stem} (will rely on CUBIN if available)"
                );
            }
        }

        // CUBIN (preferred at runtime)
        match compile_to_cubin(
            &nvcc,
            &arch_compute,
            &arch_sm,
            cu,
            &cubin_path,
            &include_dirs,
            ccbin.as_deref(),
            &extra_nvcc_flags,
        ) {
            Ok(()) => {
                println!(
                    "cargo:warning=nvcc CUBIN OK: {} -> {}",
                    cu.display(),
                    cubin_path.display()
                );
                generated_cubin.push((stem, cubin_path.clone()));
            }
            Err(err) => {
                eprintln!(
                    "cargo:warning=nvcc CUBIN failed for {}: {err}",
                    cu.display()
                );
                eprintln!(
                    "cargo:warning=Skipping CUBIN for {stem} (will fall back to PTX if present)"
                );
            }
        }
    }

    // Fail fast if no artifacts were generated (prevents producing a GPU binary without embeds)
    if generated_ptx.is_empty() && generated_cubin.is_empty() {
        return Err(Box::new(io::Error::other(
            "engine-gpu-cuda: no CUDA artifacts embedded (PTX/CUBIN). Ensure NVCC is set and CUDA_ARCH=sm_XX; see README Fedora section.",
        )));
    }

    // Generate an embedded PTX/CUBIN bindings file so the crate can include!() the artifacts at compile time.
    // Output: $OUT_DIR/ptx_bindings.rs with:
    //   - pub mod ptx_embedded { pub const <STEM>_PTX: &str = "..."; pub fn get(name: &str) -> Option<&'static str>; }
    //   - pub mod cubin_embedded { pub static <STEM>_CUBIN: &'static [u8] = &[…]; pub fn get_cubin(name: &str) -> Option<&'static [u8]>; }
    let bindings_path = out_dir.join("ptx_bindings.rs");
    let mut rs = String::new();
    rs.push_str("// @generated by engine-gpu-cuda/build.rs — DO NOT EDIT MANUALLY\n");
    rs.push_str("#[allow(dead_code)]\n");

    // PTX module (string)
    rs.push_str("pub mod ptx_embedded {\n");
    for (stem, path) in &generated_ptx {
        let const_name = to_const_name(stem, "_PTX");
        let ptx = std::fs::read_to_string(path).unwrap_or_else(|_| String::new());
        rs.push_str(&format!("    pub const {const_name}: &str = r###\""));
        rs.push_str(&ptx);
        rs.push_str("\"###;\n");
    }
    rs.push_str("    pub fn get(name: &str) -> Option<&'static str> {\n");
    rs.push_str("        match name {\n");
    for (stem, _) in &generated_ptx {
        let const_name = to_const_name(stem, "_PTX");
        rs.push_str(&format!("            \"{stem}\" => Some({const_name}),\n"));
    }
    rs.push_str("            _ => None,\n");
    rs.push_str("        }\n");
    rs.push_str("    }\n");
    rs.push_str("}\n");

    // CUBIN module (bytes)
    rs.push_str("pub mod cubin_embedded {\n");
    for (stem, path) in &generated_cubin {
        let const_name = to_const_name(stem, "_CUBIN");
        let bytes = std::fs::read(path).unwrap_or_default();
        rs.push_str(&format!(
            "    pub static {const_name}: &'static [u8] = &[\n        "
        ));
        for (i, b) in bytes.iter().enumerate() {
            if i > 0 && i % 16 == 0 {
                rs.push_str("\n        ");
            }
            rs.push_str(&format!("{b},"));
        }
        rs.push_str("\n    ];\n");
    }
    rs.push_str("    pub fn get_cubin(name: &str) -> Option<&'static [u8]> {\n");
    rs.push_str("        match name {\n");
    for (stem, _) in &generated_cubin {
        let const_name = to_const_name(stem, "_CUBIN");
        rs.push_str(&format!("            \"{stem}\" => Some({const_name}),\n"));
    }
    rs.push_str("            _ => None,\n");
    rs.push_str("        }\n");
    rs.push_str("    }\n");
    rs.push_str("}\n");

    std::fs::write(&bindings_path, rs)?;

    Ok(())
}

fn to_const_name(stem: &str, suffix: &str) -> String {
    let mut s = String::with_capacity(stem.len() + suffix.len());
    for c in stem.chars() {
        if c.is_ascii_alphanumeric() {
            s.push(c.to_ascii_uppercase());
        } else {
            s.push('_');
        }
    }
    s.push_str(suffix);
    s
}

fn generate_empty_bindings(out_dir: &Path) -> io::Result<()> {
    let bindings_path = out_dir.join("ptx_bindings.rs");
    let mut rs = String::new();
    rs.push_str("// @generated by engine-gpu-cuda/build.rs — DO NOT EDIT MANUALLY\n");
    rs.push_str("#[allow(dead_code)]\n");
    rs.push_str("pub mod ptx_embedded { pub fn get(_: &str) -> Option<&'static str> { None } }\n");
    rs.push_str(
        "pub mod cubin_embedded { pub fn get_cubin(_: &str) -> Option<&'static [u8]> { None } }\n",
    );
    std::fs::write(&bindings_path, rs)
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

fn compile_to_ptx(
    nvcc: &Path,
    arch: &str,
    src: &Path,
    out: &Path,
    includes: &[PathBuf],
    ccbin: Option<&str>,
    extra_flags: &[String],
) -> io::Result<()> {
    // Ensure parent directory exists
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }

    // nvcc -ptx -arch=<compute_XX> -I<inc>... [-ccbin cc] [extra] -o <out.ptx> <src.cu>
    let mut cmd = Command::new(nvcc);
    cmd.args(["-ptx", "-arch", arch]).arg("-o").arg(out);
    if let Some(cc) = ccbin {
        cmd.arg("-ccbin").arg(cc);
    }
    for f in extra_flags {
        cmd.arg(f);
    }
    for inc in includes {
        cmd.arg("-I").arg(inc);
    }
    cmd.arg(src);

    let status = cmd.status()?;
    if !status.success() {
        return Err(io::Error::other(format!(
            "nvcc returned non-zero exit status (ptx): {status}"
        )));
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn compile_to_cubin(
    nvcc: &Path,
    arch_compute: &str,
    arch_sm: &str,
    src: &Path,
    out: &Path,
    includes: &[PathBuf],
    ccbin: Option<&str>,
    extra_flags: &[String],
) -> io::Result<()> {
    // Ensure parent directory exists
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }

    // nvcc -cubin -arch=<compute_XX> -code=<sm_XX> -I<inc>... [-ccbin cc] [extra] -o <out.cubin> <src.cu>
    let mut cmd = Command::new(nvcc);
    cmd.args(["-cubin", "-arch", arch_compute, "-code", arch_sm])
        .arg("-o")
        .arg(out);
    if let Some(cc) = ccbin {
        cmd.arg("-ccbin").arg(cc);
    }
    for f in extra_flags {
        cmd.arg(f);
    }
    for inc in includes {
        cmd.arg("-I").arg(inc);
    }
    cmd.arg(src);

    let status = cmd.status()?;
    if !status.success() {
        return Err(io::Error::other(format!(
            "nvcc returned non-zero exit status (cubin): {status}"
        )));
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
