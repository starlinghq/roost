// Spike S6: host expert GEMV (bf16 weights x f32 activation -> f32, batch 1) in Rust with AVX2+FMA intrinsics.
// Same kernel shape as the .NET Vector256 version (four rows share each x load) so the comparison is the compiler and runtime.
use std::arch::x86_64::*;
use std::time::Instant;

fn bf16_to_f32(v: u16) -> f32 { f32::from_bits((v as u32) << 16) }

fn scalar(w: &[u16], x: &[f32], y: &mut [f32], n0: usize, n1: usize, k: usize) {
    for n in n0..n1 {
        let r = &w[n * k..(n + 1) * k];
        let mut acc = 0f64;
        for i in 0..k { acc += bf16_to_f32(r[i]) as f64 * x[i] as f64; }
        y[n - n0] = acc as f32;
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn widen(p: *const u16) -> __m256 {
    let v = _mm_loadu_si128(p as *const __m128i);
    _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(v), 16))
}

#[target_feature(enable = "avx2,fma")]
unsafe fn hsum(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    _mm_cvtss_f32(s)
}

#[target_feature(enable = "avx2,fma")]
unsafe fn avx2(w: *const u16, x: *const f32, y: *mut f32, n0: usize, n1: usize, k: usize) {
    let mut n = n0;
    while n + 4 <= n1 {
        let r0 = w.add(n * k); let r1 = r0.add(k); let r2 = r1.add(k); let r3 = r2.add(k);
        let (mut a0, mut a1, mut a2, mut a3) = (_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps());
        let mut i = 0;
        while i < k {
            let xv = _mm256_loadu_ps(x.add(i));
            a0 = _mm256_fmadd_ps(widen(r0.add(i)), xv, a0);
            a1 = _mm256_fmadd_ps(widen(r1.add(i)), xv, a1);
            a2 = _mm256_fmadd_ps(widen(r2.add(i)), xv, a2);
            a3 = _mm256_fmadd_ps(widen(r3.add(i)), xv, a3);
            i += 8;
        }
        *y.add(n - n0) = hsum(a0); *y.add(n + 1 - n0) = hsum(a1); *y.add(n + 2 - n0) = hsum(a2); *y.add(n + 3 - n0) = hsum(a3);
        n += 4;
    }
    while n < n1 {
        let r = std::slice::from_raw_parts(w.add(n * k), k);
        let xs = std::slice::from_raw_parts(x, k);
        let mut acc = 0f64;
        for i in 0..k { acc += bf16_to_f32(r[i]) as f64 * xs[i] as f64; }
        *y.add(n - n0) = acc as f32;
        n += 1;
    }
}

fn run(w: &[u16], x: &[f32], y: &mut [f32], n: usize, k: usize, threads: usize) {
    if threads == 1 { unsafe { avx2(w.as_ptr(), x.as_ptr(), y.as_mut_ptr(), 0, n, k) }; return; }
    let chunk = (n.div_ceil(threads) + 3) & !3;
    std::thread::scope(|s| {
        for (t, ys) in y.chunks_mut(chunk).enumerate() {
            let n0 = t * chunk; let n1 = (n0 + ys.len()).min(n);
            let (wp, xp, yp) = (w.as_ptr() as usize, x.as_ptr() as usize, ys.as_mut_ptr() as usize);
            s.spawn(move || unsafe { avx2(wp as *const u16, xp as *const f32, yp as *mut f32, n0, n1, k) });
        }
    });
}

// Streaming read bandwidth over the same buffer layout: the ceiling a bandwidth-bound GEMV can reach.
fn stream_read(buf: &[u64], threads: usize) -> u64 {
    let chunk = buf.len().div_ceil(threads);
    std::thread::scope(|s| {
        let hs: Vec<_> = buf.chunks(chunk).map(|c| s.spawn(move || c.iter().fold(0u64, |a, &v| a ^ v))).collect();
        hs.into_iter().map(|h| h.join().unwrap()).fold(0, |a, v| a ^ v)
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let threads: usize = args.get(1).map_or(1, |s| s.parse().unwrap());
    let reps: usize = args.get(2).map_or(5, |s| s.parse().unwrap());
    println!("rust {}; avx2={} fma={} avx512f={}; threads={}", env!("CARGO_PKG_VERSION"),
        is_x86_feature_detected!("avx2"), is_x86_feature_detected!("fma"), is_x86_feature_detected!("avx512f"), threads);
    let mut seed = 7u64;
    let mut rnd = || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; (seed >> 11) as f64 / (1u64 << 53) as f64 };
    for (name, n, k) in [("gate_up 1536x2048", 1536usize, 2048usize), ("down 2048x768", 2048, 768)] {
        for experts in [1usize, 128] {
            let per = n * k;
            let w: Vec<u16> = (0..per * experts).map(|_| ((((rnd() - 0.5) * 0.05) as f32).to_bits() >> 16) as u16).collect();
            let x: Vec<f32> = (0..k).map(|_| (rnd() - 0.5) as f32).collect();
            let mut y = vec![0f32; n]; let mut yref = vec![0f32; n];
            scalar(&w[..per], &x, &mut yref, 0, n, k);
            let iters = 400;
            let mut best = f64::MAX;
            for _ in 0..reps {
                let t0 = Instant::now();
                for it in 0..iters { run(&w[(it % experts) * per..(it % experts + 1) * per], &x, &mut y, n, k, threads); }
                best = best.min(t0.elapsed().as_secs_f64() / iters as f64);
            }
            // y holds the result for expert (iters-1)%experts; recheck against scalar on that expert.
            let e = (iters - 1) % experts;
            scalar(&w[e * per..(e + 1) * per], &x, &mut yref, 0, n, k);
            let err = y.iter().zip(&yref).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            let bytes = (per * 2) as f64;
            println!("RESULT impl=rust-avx2 shape=\"{name}\" experts_rotated={experts} threads={threads} time_us={:.1} GB/s={:.2} GFLOP/s={:.2} max_abs_err_vs_f64={:.2e}",
                best * 1e6, bytes / best / 1e9, 2.0 * per as f64 / best / 1e9, err);
        }
    }
    // Memory read ceiling: 1 GiB buffer.
    let big: Vec<u64> = (0..(1usize << 27)).map(|i| i as u64).collect();
    let mut best = f64::MAX; let mut sink = 0;
    for _ in 0..reps { let t0 = Instant::now(); sink ^= stream_read(&big, threads); best = best.min(t0.elapsed().as_secs_f64()); }
    println!("RESULT impl=rust-stream-read-ceiling threads={threads} GB/s={:.2} (sink {sink})", (big.len() * 8) as f64 / best / 1e9);
}
