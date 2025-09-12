/**
 * Quantus External Miner - CUDA Kernel (G1 bring-up)
 *
 * This kernel provides a minimal, correctness-first pipeline for the GPU path:
 * - Implement 512-bit Montgomery multiplication (8×64-bit limbs) on device.
 * - Keep y in Montgomery domain during iteration and convert to normal domain before output.
 * - For bring-up (G1), the kernel writes normalized y values back to host memory.
 *   The host will compute SHA3-512(y_be64) and distances for parity validation.
 *
 * Notes:
 * - Limbs are little-endian: limb 0 is the least significant 64 bits.
 * - CIOS Montgomery reduction is used with 64×64→128 products via __umul64hi.
 * - This skeleton intentionally excludes early-exit and on-device SHA3; those are part of G2.
 *
 * Build:
 * - The engine-gpu-cuda crate provides a build.rs that compiles this .cu into PTX when the
 *   "cuda" feature is enabled, placing artifacts under $OUT_DIR and exposing ENGINE_GPU_CUDA_PTX_DIR.
 */

#include <stdint.h>
#include <cuda_runtime.h>

extern "C" {

// -------------------------------------------------------------------------------------------------
// Utilities: 64×64→128 multiply (lo, hi), add-with-carry helpers, compare/subtract
// -------------------------------------------------------------------------------------------------

__device__ __forceinline__ void mul64wide(uint64_t a, uint64_t b, uint64_t &lo, uint64_t &hi) {
    lo = a * b;
    hi = __umul64hi(a, b);
}

// sum := x + y, carry_out returns 0 or 1
__device__ __forceinline__ uint64_t add64_carry(uint64_t x, uint64_t y, uint64_t &carry_out) {
    uint64_t s = x + y;
    carry_out = (s < x) ? 1ull : 0ull;
    return s;
}

// sum := x + y + carry_in, carry_out returns 0 or 1
__device__ __forceinline__ uint64_t add64_2carry(uint64_t x, uint64_t y, uint64_t carry_in, uint64_t &carry_out) {
    uint64_t s1 = x + y;
    uint64_t c1 = (s1 < x) ? 1ull : 0ull;
    uint64_t s2 = s1 + carry_in;
    uint64_t c2 = (s2 < s1) ? 1ull : 0ull;
    carry_out = c1 + c2;
    return s2;
}

// return true if a (LE limbs) >= b (LE limbs), by numeric value
__device__ __forceinline__ bool ge_le_8(const uint64_t a[8], const uint64_t b[8]) {
    // Compare from most significant limb to least
    for (int i = 7; i >= 0; --i) {
        if (a[i] != b[i]) {
            return a[i] > b[i];
        }
    }
    return true; // equal
}

// a := a - b (LE limbs)
__device__ __forceinline__ void sub_le_in_place_8(uint64_t a[8], const uint64_t b[8]) {
    uint64_t borrow = 0;
    for (int i = 0; i < 8; ++i) {
        uint64_t bi = b[i];
        uint64_t ai = a[i];
        uint64_t tmp = ai - bi - borrow;
        // borrow occurs if ai < (bi + borrow)
        uint64_t needed = (ai < bi) || (borrow && ai == bi) ? 1ull : 0ull;
        a[i] = tmp;
        borrow = needed;
    }
}

// Convert 8 LE limbs into 64 BE bytes into out[64] (for host hashing later, if needed)
// Not used inside the kernel (G1 writes limbs), but kept here for reference.
// __device__ __forceinline__ void le8_to_be64(const uint64_t le[8], uint8_t out[64]) {
//     for (int i = 0; i < 8; ++i) {
//         uint64_t limb = le[7 - i]; // most significant limb first
//         for (int b = 0; b < 8; ++b) {
//             out[i * 8 + (7 - b)] = (uint8_t)((limb >> (b * 8)) & 0xFF);
//         }
//     }
// }

// -------------------------------------------------------------------------------------------------
// Montgomery arithmetic (CIOS) for 512-bit numbers (8×64-bit limbs), little-endian.
// -------------------------------------------------------------------------------------------------

// out <- (a * b * R^{-1}) mod n
__device__ __forceinline__ void mont_mul_512(
    const uint64_t a[8],
    const uint64_t b[8],
    const uint64_t n[8],
    const uint64_t n0_inv,
    uint64_t out[8]
) {
    // 9-limb accumulator (LE); accumulates 128-bit intermediates via split-add with carries
    uint64_t acc[9];
#pragma unroll
    for (int k = 0; k < 9; ++k) acc[k] = 0ull;

    for (int i = 0; i < 8; ++i) {
        // acc += a[i] * b
        uint64_t ai = a[i];
        uint64_t carry = 0ull;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            uint64_t lo, hi;
            mul64wide(ai, b[j], lo, hi);

            // acc[j] += lo + carry, propagate carry to hi
            uint64_t c0, c1;
            uint64_t s0 = add64_carry(acc[j], lo, c0);
            uint64_t s1 = add64_carry(s0, carry, c1);
            acc[j] = s1;
            // new carry = hi + c0 + c1
            carry = hi + c0 + c1;
        }
        // acc[8] += carry
        uint64_t c_acc8;
        acc[8] = add64_carry(acc[8], carry, c_acc8);
        // c_acc8 overflow beyond 9th limb is discarded (by design in CIOS with next steps)

        // m = (acc[0] * n0_inv) mod 2^64
        uint64_t m = (uint64_t)(acc[0] * n0_inv);

        // acc += m * n
        uint64_t carry2 = 0ull;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            uint64_t lo2, hi2;
            mul64wide(m, n[j], lo2, hi2);

            uint64_t c0, c1;
            uint64_t s0 = add64_carry(acc[j], lo2, c0);
            uint64_t s1 = add64_carry(s0, carry2, c1);
            acc[j] = s1;
            carry2 = hi2 + c0 + c1;
        }
        // acc[8] += carry2
        uint64_t c_acc8_b;
        acc[8] = add64_carry(acc[8], carry2, c_acc8_b);

        // Shift acc right by one limb (drop acc[0])
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc[j] = acc[j + 1];
        }
        acc[8] = 0ull;
    }

    // Conditional subtract: if acc >= n, subtract n
    if (ge_le_8(acc, n)) {
        sub_le_in_place_8(acc, n);
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        out[i] = acc[i];
    }
}

// to_mont(x) = x * R^2 mod n
__device__ __forceinline__ void to_mont_512(
    const uint64_t x[8],
    const uint64_t r2[8],
    const uint64_t n[8],
    const uint64_t n0_inv,
    uint64_t out[8]
) {
    mont_mul_512(x, r2, n, n0_inv, out);
}

// from_mont(x̂) = x̂ * 1 mod n
__device__ __forceinline__ void from_mont_512(
    const uint64_t xhat[8],
    const uint64_t n[8],
    const uint64_t n0_inv,
    uint64_t out[8]
) {
    // Multiply by 1 (Montgomery): one = [1,0,..,0]
    uint64_t one[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) one[i] = 0ull;
    one[0] = 1ull;
    mont_mul_512(xhat, one, n, n0_inv, out);
}

// -------------------------------------------------------------------------------------------------
// Kernel: G1 bring-up
//
// Each thread:
//  - Loads y0 (normal domain) for that thread.
//  - Computes y_hat0 = to_mont(y0) on device.
//  - Iterates iters_per_thread times:
//      y_hat = mont_mul(y_hat, m_hat)
//      y = from_mont(y_hat)
//      Writes y (LE limbs) to y_out at [thread_offset + iter]
// -------------------------------------------------------------------------------------------------

extern "C" __global__ void qpow_montgomery_g1_kernel(
    // Per-job constants (each 8 limbs, LE)
    const uint64_t* __restrict__ m,        // not used in G1 directly (we pass m_hat)
    const uint64_t* __restrict__ n,
    const uint64_t  n0_inv,
    const uint64_t* __restrict__ r2,
    const uint64_t* __restrict__ m_hat,

    // Per-thread starting state (normal domain)
    const uint64_t* __restrict__ y0,       // length: num_threads * 8 limbs

    // Output buffer for normalized y (for host SHA3 in G1)
    uint64_t* __restrict__ y_out,          // length: num_threads * iters_per_thread * 8 limbs

    // Threading parameters
    const uint32_t num_threads,
    const uint32_t iters_per_thread
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_threads) {
        return;
    }

    // Local copies of constants (consider placing in __constant__ memory for G2+)
    uint64_t n_loc[8], r2_loc[8], mhat_loc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        n_loc[i]    = n[i];
        r2_loc[i]   = r2[i];
        mhat_loc[i] = m_hat[i];
    }

    // Load this thread's y0 (normal domain)
    uint64_t y0_loc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        y0_loc[i] = y0[tid * 8u + i];
    }

    // Transform to Montgomery domain
    uint64_t yhat[8];
    to_mont_512(y0_loc, r2_loc, n_loc, n0_inv, yhat);

    // Iterate and emit normalized y per step
    // Output stride per thread: iters_per_thread * 8 limbs
    uint64_t* out_base = y_out + (static_cast<size_t>(tid) * static_cast<size_t>(iters_per_thread) * 8ull);

    for (uint32_t iter = 0; iter < iters_per_thread; ++iter) {
        // y_hat = y_hat * m_hat
        uint64_t yhat_next[8];
        mont_mul_512(yhat, mhat_loc, n_loc, n0_inv, yhat_next);

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            yhat[i] = yhat_next[i];
        }

        // y = from_mont(y_hat)
        uint64_t y_norm[8];
        from_mont_512(yhat, n_loc, n0_inv, y_norm);

        // Store normalized y (LE limbs) for host SHA3 and distance validation
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            out_base[iter * 8u + i] = y_norm[i];
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Host-callable launcher wrapper (optional; typically loaded via PTX and launched from Rust)
// -------------------------------------------------------------------------------------------------

// The Rust host side will load this kernel from PTX and launch it using `cust`/`rustacuda`.
// Example signature in Rust (pseudo):
//
// launch!(module.qpow_montgomery_g1_kernel<<<grid, block, 0, stream>>>(
//     d_m.as_device_ptr(),
//     d_n.as_device_ptr(),
//     n0_inv,
//     d_r2.as_device_ptr(),
//     d_mhat.as_device_ptr(),
//     d_y0.as_device_ptr(),
//     d_y_out.as_device_ptr(),
//     num_threads,
//     iters_per_thread
// ))?;
//
// Note: For G1, the host will compute SHA3-512(y) and distances on the CPU,
// validating correctness against cpu-fast/cpu-montgomery on small ranges.
//
 
// -------------------------------------------------------------------------------------------------
// G2 additions: device SHA3-512, threshold compare, and early-exit
// -------------------------------------------------------------------------------------------------
 
// 64-bit rotate-left
__device__ __forceinline__ uint64_t rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}
 
// Keccak-f[1600] round constants
__device__ __constant__ uint64_t KECCAK_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL
};
 
// Load/store helpers (little-endian)
__device__ __forceinline__ uint64_t load64_le(const uint8_t* p) {
    return ((uint64_t)p[0])       |
           ((uint64_t)p[1] << 8)  |
           ((uint64_t)p[2] << 16) |
           ((uint64_t)p[3] << 24) |
           ((uint64_t)p[4] << 32) |
           ((uint64_t)p[5] << 40) |
           ((uint64_t)p[6] << 48) |
           ((uint64_t)p[7] << 56);
}
__device__ __forceinline__ void store64_le(uint8_t* p, uint64_t v) {
    p[0] = (uint8_t)(v);
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
    p[4] = (uint8_t)(v >> 32);
    p[5] = (uint8_t)(v >> 40);
    p[6] = (uint8_t)(v >> 48);
    p[7] = (uint8_t)(v >> 56);
}
 
// Keccak-f[1600] permutation
__device__ __forceinline__ void keccak_f1600(uint64_t s[25]) {
#pragma unroll
    for (int round = 0; round < 24; ++round) {
        // Theta
        uint64_t C[5], D[5];
#pragma unroll
        for (int x = 0; x < 5; ++x) {
            C[x] = s[x] ^ s[x + 5] ^ s[x + 10] ^ s[x + 15] ^ s[x + 20];
        }
#pragma unroll
        for (int x = 0; x < 5; ++x) {
            D[x] = C[(x + 4) % 5] ^ rotl64(C[(x + 1) % 5], 1);
        }
#pragma unroll
        for (int x = 0; x < 5; ++x) {
#pragma unroll
            for (int y = 0; y < 5; ++y) {
                s[x + 5 * y] ^= D[x];
            }
        }
 
        // Rho and Pi
        uint64_t B[25];
        const int r[25] = {
            0,  1, 62, 28, 27,
           36, 44,  6, 55, 20,
            3, 10, 43, 25, 39,
           41, 45, 15, 21,  8,
           18,  2, 61, 56, 14
        };
#pragma unroll
        for (int x = 0; x < 5; ++x) {
#pragma unroll
            for (int y = 0; y < 5; ++y) {
                int idx = x + 5 * y;
                int X = y;
                int Y = (2 * x + 3 * y) % 5;
                B[X + 5 * Y] = rotl64(s[idx], r[idx]);
            }
        }
 
        // Chi
#pragma unroll
        for (int x = 0; x < 5; ++x) {
#pragma unroll
            for (int y = 0; y < 5; ++y) {
                s[x + 5 * y] = B[x + 5 * y] ^ ((~B[((x + 1) % 5) + 5 * y]) & B[((x + 2) % 5) + 5 * y]);
            }
        }
 
        // Iota
        s[0] ^= KECCAK_RC[round];
    }
}
 
// Device SHA3-512 for a single 64-byte message
__device__ __forceinline__ void sha3_512_64bytes(const uint8_t in_be64[64], uint8_t out_be64[64]) {
    // Initialize state to zero
    uint64_t s[25];
#pragma unroll
    for (int i = 0; i < 25; ++i) s[i] = 0ull;
 
    // Absorb (rate = 72 bytes). Message is 64 bytes: append 0x06 then pad with zeros and set last of rate |= 0x80
    uint8_t block[72];
#pragma unroll
    for (int i = 0; i < 72; ++i) block[i] = 0;
#pragma unroll
    for (int i = 0; i < 64; ++i) block[i] = in_be64[i];
    block[64] = 0x06;
    block[71] ^= 0x80;
 
    // XOR into state lanes as little-endian 64-bit words
#pragma unroll
    for (int i = 0; i < 9; ++i) {
        s[i] ^= load64_le(&block[i * 8]);
    }
 
    // Permute
    keccak_f1600(s);
 
    // Squeeze 64 bytes (8 lanes)
    uint8_t out_le[64];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        store64_le(&out_le[i * 8], s[i]);
    }
 
    // The standard digest byte order expected by host libraries is the emitted byte stream.
    // We keep it as-is; host target/threshold bytes should be in the same orientation to XOR/compare.
#pragma unroll
    for (int i = 0; i < 64; ++i) {
        out_be64[i] = out_le[i];
    }
}
 
// Convert 8 LE limbs into 64 BE bytes (big-endian numeric representation)
__device__ __forceinline__ void le8_to_be64_bytes(const uint64_t le[8], uint8_t out[64]) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint64_t limb = le[7 - i]; // most significant limb first
#pragma unroll
        for (int b = 0; b < 8; ++b) {
            out[i * 8 + (7 - b)] = (uint8_t)((limb >> (b * 8)) & 0xFF);
        }
    }
}
 
// Compare two 64-byte big-endian numbers: return true if a <= b
__device__ __forceinline__ bool be64_leq(const uint8_t a[64], const uint8_t b[64]) {
#pragma unroll
    for (int i = 0; i < 64; ++i) {
        if (a[i] != b[i]) {
            return a[i] < b[i];
        }
    }
    return true; // equal
}
 
// Kernel: G2 — device SHA3-512 + threshold compare + early-exit
//
// Notes:
// - Signature includes additional G2 parameters; host launcher must be updated to pass them.
// - Early-exit: a single global int flag claimed via atomicCAS; earliest winning thread writes result.
//
extern "C" __global__ void qpow_montgomery_g2_kernel(
    // Per-job constants (each 8 limbs, LE)
    const uint64_t* __restrict__ m,
    const uint64_t* __restrict__ n,
    const uint64_t  n0_inv,
    const uint64_t* __restrict__ r2,
    const uint64_t* __restrict__ m_hat,
 
    // Per-thread starting state (normal domain)
    const uint64_t* __restrict__ y0,            // length: num_threads * 8 limbs
 
    // G2-specific inputs/outputs
    const uint8_t*  __restrict__ target_be,     // 64 bytes
    const uint8_t*  __restrict__ threshold_be,  // 64 bytes
    int*            __restrict__ found_flag,    // 0 -> not found, 1 -> found
    uint32_t*       __restrict__ out_index,     // linear index (t * iters + j)
    uint8_t*        __restrict__ out_distance_be, // 64 bytes
 
    // Threading parameters
    const uint32_t num_threads,
    const uint32_t iters_per_thread
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_threads) {
        return;
    }
 
    // Quick early-exit check
    if (atomicAdd(found_flag, 0) != 0) {
        return;
    }
 
    // Local copies of constants
    uint64_t n_loc[8], r2_loc[8], mhat_loc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        n_loc[i]    = n[i];
        r2_loc[i]   = r2[i];
        mhat_loc[i] = m_hat[i];
    }
 
    // Load this thread's y0 (normal domain) and move to Montgomery domain
    uint64_t y0_loc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        y0_loc[i] = y0[tid * 8u + i];
    }
    uint64_t yhat[8];
    to_mont_512(y0_loc, r2_loc, n_loc, n0_inv, yhat);
 
    // Iterate and check threshold
    const uint32_t iters = iters_per_thread;
    for (uint32_t j = 0; j < iters; ++j) {
        // Respect early-exit
        if (atomicAdd(found_flag, 0) != 0) {
            return;
        }
 
        // y_hat = y_hat * m_hat
        uint64_t yhat_next[8];
        mont_mul_512(yhat, mhat_loc, n_loc, n0_inv, yhat_next);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            yhat[i] = yhat_next[i];
        }
 
        // y = from_mont(y_hat)
        uint64_t y_norm[8];
        from_mont_512(yhat, n_loc, n0_inv, y_norm);
 
        // y_be64 (64 bytes) from LE limbs
        uint8_t y_be[64];
        le8_to_be64_bytes(y_norm, y_be);
 
        // H = SHA3-512(y_be)
        uint8_t h_be[64];
        sha3_512_64bytes(y_be, h_be);
 
        // distance = target XOR H (byte-wise)
        uint8_t dist_be[64];
#pragma unroll
        for (int i = 0; i < 64; ++i) {
            dist_be[i] = target_be[i] ^ h_be[i];
        }
 
        // Compare distance <= threshold
        if (be64_leq(dist_be, threshold_be)) {
            // Try to claim the flag
            if (atomicCAS(found_flag, 0, 1) == 0) {
                // Write linear index for host to reconstruct nonce
                if (out_index) {
                    *out_index = tid * iters + j;
                }
                // Write distance
                if (out_distance_be) {
#pragma unroll
                    for (int i = 0; i < 64; ++i) {
                        out_distance_be[i] = dist_be[i];
                    }
                }
            }
            return; // early-exit after claiming
        }
    }
}
 
} // extern "C"