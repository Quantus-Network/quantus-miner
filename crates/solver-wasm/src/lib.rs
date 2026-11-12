//! Browser captcha solver: grinds Poseidon2 nonce hashes over a block header.
//!
//! Exposed as a raw C-ABI WASM module (no wasm-bindgen) so it can be built
//! with a bare `cargo build --target wasm32-unknown-unknown` and loaded with
//! minimal JS glue. All I/O goes through one static buffer:
//!
//! ```text
//! offset   0..32   header hash                     (in)
//! offset  32..96   nonce, 64-byte big-endian       (in: start; out: found/next)
//! offset  96..160  share target, 64-byte BE        (in)
//! offset 160..224  winning hash, 64-byte BE        (out, only when found)
//! ```
//!
//! JS calls `solve(iters)` in chunks (e.g. 5–20k iterations) to keep the
//! worker responsive; a return of 0 means "keep going" (the nonce field has
//! been advanced), 1 means the nonce field now holds a valid share.

use core::cell::UnsafeCell;

use primitive_types::U512;

const HEADER_OFF: usize = 0;
const NONCE_OFF: usize = 32;
const TARGET_OFF: usize = 96;
const HASH_OFF: usize = 160;
const IO_SIZE: usize = 224;

/// Shared I/O buffer. `UnsafeCell` (rather than `static mut`) makes interior
/// mutability explicit at the type level; the `Sync` assertion below states
/// the invariant that is otherwise implicit.
struct IoBuf(UnsafeCell<[u8; IO_SIZE]>);

// SAFETY: `wasm32-unknown-unknown` is single-threaded. The JS glue owns one
// module instance, calls `solve()` synchronously on the worker thread, and
// never reenters or touches the buffer while `solve()` is running. Under that
// contract there is exactly one live reference to `IO.0` at any time.
unsafe impl Sync for IoBuf {}

static IO: IoBuf = IoBuf(UnsafeCell::new([0u8; IO_SIZE]));

/// Pointer to the shared I/O buffer in WASM linear memory.
#[no_mangle]
pub extern "C" fn io_ptr() -> *mut u8 {
    IO.0.get() as *mut u8
}

/// Size of the shared I/O buffer, for JS-side sanity checks.
#[no_mangle]
pub extern "C" fn io_len() -> u32 {
    IO_SIZE as u32
}

/// Grind up to `iters` nonces. Returns 1 if a share was found.
#[no_mangle]
pub extern "C" fn solve(iters: u32) -> u32 {
    // SAFETY: single-threaded target; the reference does not outlive this
    // call and the JS caller does not reenter `solve`.
    let io = unsafe { &mut *IO.0.get() };

    let mut header = [0u8; 32];
    header.copy_from_slice(&io[HEADER_OFF..HEADER_OFF + 32]);
    let mut nonce = U512::from_big_endian(&io[NONCE_OFF..NONCE_OFF + 64]);
    let target = U512::from_big_endian(&io[TARGET_OFF..TARGET_OFF + 64]);

    for _ in 0..iters {
        let nonce_bytes = nonce.to_big_endian();
        let hash = qpow_math::get_nonce_hash(header, nonce_bytes);
        if hash < target {
            io[NONCE_OFF..NONCE_OFF + 64].copy_from_slice(&nonce_bytes);
            io[HASH_OFF..HASH_OFF + 64].copy_from_slice(&hash.to_big_endian());
            return 1;
        }
        nonce = nonce.saturating_add(U512::one());
    }

    // Not found: persist progress so the next chunk resumes where we stopped.
    io[NONCE_OFF..NONCE_OFF + 64].copy_from_slice(&nonce.to_big_endian());
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_matches_pool_verification() {
        // Difficulty 4 -> target = MAX/4; expect a solution within a few iters.
        let header = [7u8; 32];
        let difficulty = U512::from(4u64);
        let target = U512::MAX / difficulty;
        let start = U512::from(1u64) << 64; // arbitrary "session range" start

        // Seed the buffer in a scoped borrow, then drop it before calling
        // `solve`, which takes its own `&mut` to the same `UnsafeCell`.
        {
            // SAFETY: single-threaded test; the borrow ends at the closing
            // brace, so no second `&mut` can overlap `solve` below.
            let io = unsafe { &mut *IO.0.get() };
            io[HEADER_OFF..HEADER_OFF + 32].copy_from_slice(&header);
            io[NONCE_OFF..NONCE_OFF + 64].copy_from_slice(&start.to_big_endian());
            io[TARGET_OFF..TARGET_OFF + 64].copy_from_slice(&target.to_big_endian());
        }

        let mut found = 0;
        for _ in 0..64 {
            found = solve(256);
            if found == 1 {
                break;
            }
        }
        assert_eq!(found, 1, "should find a share at difficulty 4");

        // SAFETY: `solve` has returned; no `&mut` to the buffer is live.
        let io = unsafe { &*IO.0.get() };
        let nonce = U512::from_big_endian(&io[NONCE_OFF..NONCE_OFF + 64]);
        let (valid, hash) = qpow_math::is_valid_nonce(header, nonce.to_big_endian(), difficulty);
        assert!(
            valid,
            "pool-side verification must accept the solver's nonce"
        );
        let reported = U512::from_big_endian(&io[HASH_OFF..HASH_OFF + 64]);
        assert_eq!(hash, reported);
    }
}
