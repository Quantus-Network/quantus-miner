//! GPU tier detection using regex patterns
//!
//! This module provides a table-driven approach to GPU detection,
//! using regex patterns with word boundaries to avoid substring matching issues.

use once_cell::sync::Lazy;
use regex::Regex;

/// GPU tier configuration
struct GpuTier {
    /// Regex pattern to match GPU name (case-insensitive)
    pattern: &'static str,
    /// Human-readable tier name
    name: &'static str,
    /// Divisor for max_workgroups (higher = more conservative)
    workgroup_divisor: u32,
    /// Minimum workgroups to use
    min_workgroups: u32,
}

/// Compiled GPU tier with regex
struct CompiledGpuTier {
    regex: Regex,
    name: &'static str,
    workgroup_divisor: u32,
    min_workgroups: u32,
}

impl CompiledGpuTier {
    fn from_tier(tier: &GpuTier) -> Self {
        Self {
            regex: Regex::new(&format!("(?i){}", tier.pattern)).expect("Invalid GPU tier regex"),
            name: tier.name,
            workgroup_divisor: tier.workgroup_divisor,
            min_workgroups: tier.min_workgroups,
        }
    }
}

// GPU tiers are checked in order - first match wins
// Use word boundaries (\b) to avoid substring issues like "550" matching "5500"

const NVIDIA_TIERS: &[GpuTier] = &[
    // Blackwell (RTX 50 series)
    GpuTier {
        pattern: r"\b50[89]0\b",
        name: "NVIDIA RTX 50 Flagship (Blackwell)",
        workgroup_divisor: 6,
        min_workgroups: 5120,
    },
    GpuTier {
        pattern: r"\b50[67]0\b|rtx 50",
        name: "NVIDIA RTX 50 (Blackwell)",
        workgroup_divisor: 7,
        min_workgroups: 4608,
    },
    // Ada Lovelace (RTX 40 series)
    GpuTier {
        pattern: r"\b40[89]0\b",
        name: "NVIDIA RTX 40 Flagship (Ada)",
        workgroup_divisor: 8,
        min_workgroups: 4096,
    },
    GpuTier {
        pattern: r"\b40[67]0\b|rtx 40",
        name: "NVIDIA RTX 40 (Ada)",
        workgroup_divisor: 10,
        min_workgroups: 3072,
    },
    // Ampere/Turing (RTX 30/20 series)
    GpuTier {
        pattern: r"\b30[5-9]0\b|\b20[6-8]0\b|rtx 30|rtx 20",
        name: "NVIDIA RTX 30/20 (Ampere/Turing)",
        workgroup_divisor: 12,
        min_workgroups: 2048,
    },
    // Turing/Pascal (GTX 16/10 series)
    GpuTier {
        pattern: r"\b16[56]0\b|\b10[3-8]0\b|gtx 16|gtx 10",
        name: "NVIDIA GTX 16/10 (Turing/Pascal)",
        workgroup_divisor: 16,
        min_workgroups: 1024,
    },
    // Maxwell (GTX 900 series)
    GpuTier {
        pattern: r"\b9[5-8]0\b|gtx 9",
        name: "NVIDIA GTX 900 (Maxwell)",
        workgroup_divisor: 18,
        min_workgroups: 768,
    },
    // Kepler/Maxwell (GTX 700 series)
    GpuTier {
        pattern: r"\b7[5-8]0\b|gtx 7",
        name: "NVIDIA GTX 700 (Kepler/Maxwell)",
        workgroup_divisor: 20,
        min_workgroups: 512,
    },
    // Legacy GTX
    GpuTier {
        pattern: r"gtx [456]",
        name: "NVIDIA GTX Legacy (Fermi/Kepler)",
        workgroup_divisor: 24,
        min_workgroups: 384,
    },
    GpuTier {
        pattern: r"\bgtx\b",
        name: "NVIDIA GTX (Unknown)",
        workgroup_divisor: 20,
        min_workgroups: 512,
    },
    // Mobile
    GpuTier {
        pattern: r"\bmx[1-5]\d0|geforce mx",
        name: "NVIDIA MX (Mobile)",
        workgroup_divisor: 24,
        min_workgroups: 384,
    },
    GpuTier {
        pattern: r"geforce gt\b|\bgt \d{3}\b",
        name: "NVIDIA GT (Entry-Level)",
        workgroup_divisor: 28,
        min_workgroups: 256,
    },
    // Professional
    GpuTier {
        pattern: r"quadro|rtx a\d|tesla|\ba100\b|\bh100\b|\bl4\b",
        name: "NVIDIA Quadro/Professional",
        workgroup_divisor: 10,
        min_workgroups: 2560,
    },
];

const AMD_TIERS: &[GpuTier] = &[
    // RDNA 4 (RX 9000 series)
    GpuTier {
        pattern: r"rx 9|\b90[78]0\b",
        name: "AMD RX 9000 (RDNA 4)",
        workgroup_divisor: 8,
        min_workgroups: 4096,
    },
    // RDNA 3 Discrete
    GpuTier {
        pattern: r"\b7900\b",
        name: "AMD RX 7900 (RDNA 3 Flagship)",
        workgroup_divisor: 9,
        min_workgroups: 3584,
    },
    GpuTier {
        pattern: r"rx 7|\b7[6-8]00\b",
        name: "AMD RX 7000 (RDNA 3)",
        workgroup_divisor: 10,
        min_workgroups: 3072,
    },
    // RDNA 3 APUs - check before discrete to avoid substring match
    GpuTier {
        pattern: r"\b780m\b|radeon 780m",
        name: "AMD Radeon 780M (RDNA 3 APU)",
        workgroup_divisor: 12,
        min_workgroups: 2048,
    },
    GpuTier {
        pattern: r"\b7[46]0m\b|radeon 7[46]0m",
        name: "AMD Radeon 7x0M (RDNA 3 APU)",
        workgroup_divisor: 16,
        min_workgroups: 1024,
    },
    // RDNA 2 Discrete
    GpuTier {
        pattern: r"\b6[89][05]0\b",
        name: "AMD RX 6900/6800 (RDNA 2 Flagship)",
        workgroup_divisor: 12,
        min_workgroups: 2560,
    },
    GpuTier {
        pattern: r"\b6[67][05]0\b",
        name: "AMD RX 6700/6600 (RDNA 2)",
        workgroup_divisor: 14,
        min_workgroups: 2048,
    },
    GpuTier {
        pattern: r"\b6[45]00\b",
        name: "AMD RX 6500/6400 (RDNA 2 Entry)",
        workgroup_divisor: 22,
        min_workgroups: 512,
    },
    GpuTier {
        pattern: r"rx 6\d{3}",
        name: "AMD RX 6000 (RDNA 2)",
        workgroup_divisor: 14,
        min_workgroups: 2048,
    },
    // RDNA 2 APUs - check before discrete
    GpuTier {
        pattern: r"\b680m\b|radeon 680m",
        name: "AMD Radeon 680M (RDNA 2 APU)",
        workgroup_divisor: 16,
        min_workgroups: 1536,
    },
    GpuTier {
        pattern: r"\b6[16]0m\b|radeon 6[16]0m",
        name: "AMD Radeon 6x0M (RDNA 2 APU)",
        workgroup_divisor: 22,
        min_workgroups: 768,
    },
    // RDNA 1 (RX 5000 series) - 4-digit patterns to avoid matching Polaris 3-digit
    GpuTier {
        pattern: r"\b5700\b",
        name: "AMD RX 5700 (RDNA 1)",
        workgroup_divisor: 16,
        min_workgroups: 1536,
    },
    GpuTier {
        pattern: r"\b5[56]00\b",
        name: "AMD RX 5000 (RDNA 1)",
        workgroup_divisor: 18,
        min_workgroups: 1024,
    },
    GpuTier {
        pattern: r"rx 5\d{3}",
        name: "AMD RX 5000 (RDNA 1)",
        workgroup_divisor: 18,
        min_workgroups: 1024,
    },
    // Polaris (RX 400/500 series) - 3-digit models with boundaries
    GpuTier {
        pattern: r"rx [45]\d0\b|\b[45][6-9]0\b|\b590\b|rx 5.0x",
        name: "AMD RX 500/400 (Polaris)",
        workgroup_divisor: 20,
        min_workgroups: 768,
    },
    // Vega
    GpuTier {
        pattern: r"radeon vii\b",
        name: "AMD Radeon VII (Vega 20)",
        workgroup_divisor: 12,
        min_workgroups: 2048,
    },
    GpuTier {
        pattern: r"vega\s*64",
        name: "AMD Vega 64 (Discrete)",
        workgroup_divisor: 14,
        min_workgroups: 1536,
    },
    GpuTier {
        pattern: r"vega\s*56",
        name: "AMD Vega 56 (Discrete)",
        workgroup_divisor: 16,
        min_workgroups: 1280,
    },
    GpuTier {
        pattern: r"vega",
        name: "AMD Vega (APU)",
        workgroup_divisor: 28,
        min_workgroups: 384,
    },
    // GCN
    GpuTier {
        pattern: r"fury|nano",
        name: "AMD R9 Fury/Nano (Fiji)",
        workgroup_divisor: 16,
        min_workgroups: 1280,
    },
    GpuTier {
        pattern: r"r9.*(3[89]0|2[89]0)|\b[23][89]0x?\b",
        name: "AMD R9 (GCN)",
        workgroup_divisor: 20,
        min_workgroups: 768,
    },
    GpuTier {
        pattern: r"r7.*(3[67]0|2[67]0)|\b[23][67]0x?\b",
        name: "AMD R7 (GCN)",
        workgroup_divisor: 22,
        min_workgroups: 512,
    },
    // Professional
    GpuTier {
        pattern: r"radeon pro|instinct mi|mi[123]\d0|firepro|w[567]\d{3}",
        name: "AMD Radeon Pro/Instinct",
        workgroup_divisor: 10,
        min_workgroups: 2560,
    },
    // OEM/APU fallbacks
    GpuTier {
        pattern: r"radeon\s*\(tm\)\s*[67]\d{2}\b|radeon [67]\d{2}\b",
        name: "AMD Radeon OEM (Polaris Rebrand)",
        workgroup_divisor: 24,
        min_workgroups: 512,
    },
    GpuTier {
        pattern: r"radeon\s*(\(tm\)\s*)?graphics",
        name: "AMD Radeon Graphics (APU)",
        workgroup_divisor: 26,
        min_workgroups: 384,
    },
];

const INTEL_TIERS: &[GpuTier] = &[
    // Battlemage (Arc B-Series)
    GpuTier {
        pattern: r"arc b|\bb5[78]0\b",
        name: "Intel Arc B-Series (Battlemage)",
        workgroup_divisor: 10,
        min_workgroups: 2560,
    },
    // Alchemist Mobile - check BEFORE desktop (a770m contains a770)
    GpuTier {
        pattern: r"\ba7[37]0m\b",
        name: "Intel Arc A7 Mobile (Alchemist)",
        workgroup_divisor: 14,
        min_workgroups: 1536,
    },
    GpuTier {
        pattern: r"\ba5[57]0m\b",
        name: "Intel Arc A5 Mobile (Alchemist)",
        workgroup_divisor: 16,
        min_workgroups: 1024,
    },
    GpuTier {
        pattern: r"\ba3[57]0m\b",
        name: "Intel Arc A3 Mobile (Alchemist)",
        workgroup_divisor: 20,
        min_workgroups: 512,
    },
    // Alchemist Desktop
    GpuTier {
        pattern: r"\ba7[57]0\b",
        name: "Intel Arc A7 (Alchemist)",
        workgroup_divisor: 12,
        min_workgroups: 2048,
    },
    GpuTier {
        pattern: r"\ba580\b",
        name: "Intel Arc A5 (Alchemist)",
        workgroup_divisor: 14,
        min_workgroups: 1536,
    },
    GpuTier {
        pattern: r"\ba3[18]0\b",
        name: "Intel Arc A3 (Alchemist)",
        workgroup_divisor: 18,
        min_workgroups: 768,
    },
    GpuTier {
        pattern: r"arc a|\barc\b",
        name: "Intel Arc (Unknown)",
        workgroup_divisor: 16,
        min_workgroups: 1024,
    },
    // Integrated
    GpuTier {
        pattern: r"iris xe max",
        name: "Intel Iris Xe Max (Discrete)",
        workgroup_divisor: 20,
        min_workgroups: 512,
    },
    GpuTier {
        pattern: r"iris xe",
        name: "Intel Iris Xe (Integrated)",
        workgroup_divisor: 24,
        min_workgroups: 384,
    },
    GpuTier {
        pattern: r"iris pro",
        name: "Intel Iris Pro (Integrated)",
        workgroup_divisor: 26,
        min_workgroups: 256,
    },
    GpuTier {
        pattern: r"iris plus|iris\b",
        name: "Intel Iris Plus (Integrated)",
        workgroup_divisor: 26,
        min_workgroups: 320,
    },
    GpuTier {
        pattern: r"uhd.*(7\d{2}|graphics 7)",
        name: "Intel UHD 700 (Integrated)",
        workgroup_divisor: 26,
        min_workgroups: 320,
    },
    GpuTier {
        pattern: r"uhd.*(6\d{2}|graphics 6)",
        name: "Intel UHD 600 (Integrated)",
        workgroup_divisor: 28,
        min_workgroups: 256,
    },
    GpuTier {
        pattern: r"uhd",
        name: "Intel UHD Graphics (Integrated)",
        workgroup_divisor: 28,
        min_workgroups: 256,
    },
    GpuTier {
        pattern: r"hd graphics|hd [456]\d{2}",
        name: "Intel HD Graphics (Integrated)",
        workgroup_divisor: 30,
        min_workgroups: 192,
    },
];

const QUALCOMM_TIERS: &[GpuTier] = &[
    // Snapdragon X (Adreno X1)
    GpuTier {
        pattern: r"x elite|x plus|adreno x|x1-[89]",
        name: "Qualcomm Adreno X1 (Snapdragon X)",
        workgroup_divisor: 14,
        min_workgroups: 1536,
    },
    // Adreno 700 series
    GpuTier {
        pattern: r"adreno.*7|\b7[234]0\b",
        name: "Qualcomm Adreno 700 Series",
        workgroup_divisor: 16,
        min_workgroups: 1024,
    },
    // Adreno 600 series
    GpuTier {
        pattern: r"adreno.*6|\b6[1-9]\d\b",
        name: "Qualcomm Adreno 600 Series",
        workgroup_divisor: 20,
        min_workgroups: 512,
    },
    // Adreno 500 series
    GpuTier {
        pattern: r"adreno.*5|\b5[1-4]\d\b",
        name: "Qualcomm Adreno 500 Series",
        workgroup_divisor: 24,
        min_workgroups: 384,
    },
];

const APPLE_TIERS: &[GpuTier] = &[
    // M4 series
    GpuTier {
        pattern: r"m4 ultra",
        name: "Apple M4 Ultra",
        workgroup_divisor: 4,
        min_workgroups: 1600,
    },
    GpuTier {
        pattern: r"m4 max",
        name: "Apple M4 Max",
        workgroup_divisor: 4,
        min_workgroups: 800,
    },
    GpuTier {
        pattern: r"m4 pro",
        name: "Apple M4 Pro",
        workgroup_divisor: 4,
        min_workgroups: 400,
    },
    GpuTier {
        pattern: r"\bm4\b",
        name: "Apple M4",
        workgroup_divisor: 4,
        min_workgroups: 200,
    },
    // M3 series
    GpuTier {
        pattern: r"m3 ultra",
        name: "Apple M3 Ultra",
        workgroup_divisor: 4,
        min_workgroups: 1520,
    },
    GpuTier {
        pattern: r"m3 max",
        name: "Apple M3 Max",
        workgroup_divisor: 4,
        min_workgroups: 800,
    },
    GpuTier {
        pattern: r"m3 pro",
        name: "Apple M3 Pro",
        workgroup_divisor: 4,
        min_workgroups: 360,
    },
    GpuTier {
        pattern: r"\bm3\b",
        name: "Apple M3",
        workgroup_divisor: 4,
        min_workgroups: 200,
    },
    // M2 series
    GpuTier {
        pattern: r"m2 ultra",
        name: "Apple M2 Ultra",
        workgroup_divisor: 4,
        min_workgroups: 1520,
    },
    GpuTier {
        pattern: r"m2 max",
        name: "Apple M2 Max",
        workgroup_divisor: 4,
        min_workgroups: 760,
    },
    GpuTier {
        pattern: r"m2 pro",
        name: "Apple M2 Pro",
        workgroup_divisor: 4,
        min_workgroups: 380,
    },
    GpuTier {
        pattern: r"\bm2\b",
        name: "Apple M2",
        workgroup_divisor: 4,
        min_workgroups: 200,
    },
    // M1 series
    GpuTier {
        pattern: r"m1 ultra",
        name: "Apple M1 Ultra",
        workgroup_divisor: 4,
        min_workgroups: 1280,
    },
    GpuTier {
        pattern: r"m1 max",
        name: "Apple M1 Max",
        workgroup_divisor: 4,
        min_workgroups: 640,
    },
    GpuTier {
        pattern: r"m1 pro",
        name: "Apple M1 Pro",
        workgroup_divisor: 4,
        min_workgroups: 320,
    },
    GpuTier {
        pattern: r"\bm1\b",
        name: "Apple M1",
        workgroup_divisor: 4,
        min_workgroups: 160,
    },
];

/// Compiled GPU tier tables (lazily initialized)
struct GpuTierTables {
    nvidia: Vec<CompiledGpuTier>,
    amd: Vec<CompiledGpuTier>,
    intel: Vec<CompiledGpuTier>,
    qualcomm: Vec<CompiledGpuTier>,
    apple: Vec<CompiledGpuTier>,
}

static GPU_TIERS: Lazy<GpuTierTables> = Lazy::new(|| GpuTierTables {
    nvidia: NVIDIA_TIERS
        .iter()
        .map(CompiledGpuTier::from_tier)
        .collect(),
    amd: AMD_TIERS.iter().map(CompiledGpuTier::from_tier).collect(),
    intel: INTEL_TIERS.iter().map(CompiledGpuTier::from_tier).collect(),
    qualcomm: QUALCOMM_TIERS
        .iter()
        .map(CompiledGpuTier::from_tier)
        .collect(),
    apple: APPLE_TIERS.iter().map(CompiledGpuTier::from_tier).collect(),
});

/// Result of GPU tier matching
pub struct GpuTierMatch {
    pub name: &'static str,
    pub workgroup_divisor: u32,
    pub min_workgroups: u32,
    pub is_fallback: bool,
}

/// Find matching tier from a list of compiled tiers
fn find_matching_tier(name: &str, tiers: &[CompiledGpuTier]) -> Option<GpuTierMatch> {
    tiers
        .iter()
        .find(|tier| tier.regex.is_match(name))
        .map(|tier| GpuTierMatch {
            name: tier.name,
            workgroup_divisor: tier.workgroup_divisor,
            min_workgroups: tier.min_workgroups,
            is_fallback: false,
        })
}

/// Detect GPU tier based on adapter info
///
/// Returns (tier_name, workgroup_divisor, min_workgroups, is_fallback)
pub fn detect_gpu_tier(vendor_name: &str, vendor_id: u32, is_metal_backend: bool) -> GpuTierMatch {
    let name_lower = vendor_name.to_lowercase();

    // Determine vendor and find matching tier
    if name_lower.contains("nvidia") || vendor_id == 0x10DE {
        find_matching_tier(&name_lower, &GPU_TIERS.nvidia).unwrap_or(GpuTierMatch {
            name: "NVIDIA Unknown",
            workgroup_divisor: 20,
            min_workgroups: 512,
            is_fallback: true,
        })
    } else if name_lower.contains("amd") || name_lower.contains("radeon") || vendor_id == 0x1002 {
        find_matching_tier(&name_lower, &GPU_TIERS.amd).unwrap_or(GpuTierMatch {
            name: "AMD Unknown",
            workgroup_divisor: 24,
            min_workgroups: 512,
            is_fallback: true,
        })
    } else if name_lower.contains("intel") || vendor_id == 0x8086 {
        find_matching_tier(&name_lower, &GPU_TIERS.intel).unwrap_or(GpuTierMatch {
            name: "Intel Unknown",
            workgroup_divisor: 24,
            min_workgroups: 256,
            is_fallback: true,
        })
    } else if name_lower.contains("qualcomm")
        || name_lower.contains("adreno")
        || vendor_id == 0x5143
    {
        find_matching_tier(&name_lower, &GPU_TIERS.qualcomm).unwrap_or(GpuTierMatch {
            name: "Qualcomm Adreno (Unknown)",
            workgroup_divisor: 24,
            min_workgroups: 384,
            is_fallback: true,
        })
    } else if is_metal_backend {
        // Apple Silicon - detected by Metal backend
        find_matching_tier(&name_lower, &GPU_TIERS.apple).unwrap_or(GpuTierMatch {
            name: "Apple Silicon Unknown",
            workgroup_divisor: 4,
            min_workgroups: 160,
            is_fallback: true,
        })
    } else {
        GpuTierMatch {
            name: "Unknown GPU",
            workgroup_divisor: 16,
            min_workgroups: 512,
            is_fallback: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nvidia_detection() {
        // RTX 40 series
        let tier = detect_gpu_tier("NVIDIA GeForce RTX 4090", 0x10DE, false);
        assert_eq!(tier.name, "NVIDIA RTX 40 Flagship (Ada)");
        assert!(!tier.is_fallback);

        let tier = detect_gpu_tier("NVIDIA GeForce RTX 4070 Ti", 0x10DE, false);
        assert_eq!(tier.name, "NVIDIA RTX 40 (Ada)");

        // RTX 30 series
        let tier = detect_gpu_tier("NVIDIA GeForce RTX 3080", 0x10DE, false);
        assert_eq!(tier.name, "NVIDIA RTX 30/20 (Ampere/Turing)");

        // GTX 10 series
        let tier = detect_gpu_tier("NVIDIA GeForce GTX 1080 Ti", 0x10DE, false);
        assert_eq!(tier.name, "NVIDIA GTX 16/10 (Turing/Pascal)");
    }

    #[test]
    fn test_amd_rdna_vs_polaris() {
        // RDNA 1 - should NOT match Polaris
        let tier = detect_gpu_tier("AMD Radeon RX 5500 XT", 0x1002, false);
        assert_eq!(tier.name, "AMD RX 5000 (RDNA 1)");
        assert!(!tier.is_fallback);

        let tier = detect_gpu_tier("AMD Radeon RX 5600 XT", 0x1002, false);
        assert_eq!(tier.name, "AMD RX 5000 (RDNA 1)");

        let tier = detect_gpu_tier("AMD Radeon RX 5700 XT", 0x1002, false);
        assert_eq!(tier.name, "AMD RX 5700 (RDNA 1)");

        // Polaris - should match Polaris
        let tier = detect_gpu_tier("AMD Radeon RX 580", 0x1002, false);
        assert_eq!(tier.name, "AMD RX 500/400 (Polaris)");

        let tier = detect_gpu_tier("AMD Radeon RX 560X", 0x1002, false);
        assert_eq!(tier.name, "AMD RX 500/400 (Polaris)");

        let tier = detect_gpu_tier("AMD Radeon RX 550", 0x1002, false);
        assert_eq!(tier.name, "AMD RX 500/400 (Polaris)");
    }

    #[test]
    fn test_amd_vega_apu() {
        let tier = detect_gpu_tier("AMD Radeon(TM) Vega 8 Graphics", 0x1002, false);
        assert_eq!(tier.name, "AMD Vega (APU)");
        assert!(!tier.is_fallback);
    }

    #[test]
    fn test_intel_arc_mobile_vs_desktop() {
        // Mobile should match mobile tier
        let tier = detect_gpu_tier("Intel Arc A770M", 0x8086, false);
        assert_eq!(tier.name, "Intel Arc A7 Mobile (Alchemist)");
        assert!(!tier.is_fallback);

        // Desktop should match desktop tier
        let tier = detect_gpu_tier("Intel Arc A770", 0x8086, false);
        assert_eq!(tier.name, "Intel Arc A7 (Alchemist)");
        assert!(!tier.is_fallback);
    }

    #[test]
    fn test_apple_silicon() {
        let tier = detect_gpu_tier("Apple M1 Pro", 0, true);
        assert_eq!(tier.name, "Apple M1 Pro");
        assert!(!tier.is_fallback);

        let tier = detect_gpu_tier("Apple M3 Max", 0, true);
        assert_eq!(tier.name, "Apple M3 Max");
        assert!(!tier.is_fallback);
    }
}
