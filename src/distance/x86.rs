use std::arch::x86_64::*;

use super::Kernels;

pub(super) fn avx2() -> Option<Kernels> {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        Some(Kernels { name: "avx2+fma", dot: avx2_dot, l2_sq: avx2_l2_sq, dot_u8: avx2_dot_u8 })
    } else {
        None
    }
}

pub(super) fn avx512() -> Option<Kernels> {
    if is_x86_feature_detected!("avx512f") {
        Some(Kernels { name: "avx512f", dot: avx512_dot, l2_sq: avx512_l2_sq, dot_u8: avx512_dot_u8 })
    } else {
        None
    }
}

pub(super) fn hamming_popcnt(a: &[u32], b: &[u32]) -> u32 {
    assert_eq!(a.len(), b.len());
    unsafe { hamming_popcnt_impl(a, b) }
}

#[target_feature(enable = "popcnt")]
fn hamming_popcnt_impl(a: &[u32], b: &[u32]) -> u32 {
    a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
}

// The wrappers below are only reachable through the tables built above,
// which check CPU support first.

fn avx2_dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    unsafe { dot_avx2(a, b) }
}

fn avx2_l2_sq(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    unsafe { l2_sq_avx2(a, b) }
}

fn avx2_dot_u8(q: &[f32], c: &[u8]) -> f32 {
    assert_eq!(q.len(), c.len());
    unsafe { dot_u8_avx2(q, c) }
}

fn avx512_dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    unsafe { dot_avx512(a, b) }
}

fn avx512_l2_sq(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    unsafe { l2_sq_avx512(a, b) }
}

fn avx512_dot_u8(q: &[f32], c: &[u8]) -> f32 {
    assert_eq!(q.len(), c.len());
    unsafe { dot_u8_avx512(q, c) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn hsum256(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps::<1>(v);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps::<1>(s, s));
    _mm_cvtss_f32(s)
}

#[target_feature(enable = "avx2,fma")]
fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut i = 0;
    let mut s0 = _mm256_setzero_ps();
    let mut s1 = _mm256_setzero_ps();
    let mut s2 = _mm256_setzero_ps();
    let mut s3 = _mm256_setzero_ps();
    unsafe {
        while i + 32 <= n {
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), s0);
            s1 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i + 8)), _mm256_loadu_ps(pb.add(i + 8)), s1);
            s2 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i + 16)), _mm256_loadu_ps(pb.add(i + 16)), s2);
            s3 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i + 24)), _mm256_loadu_ps(pb.add(i + 24)), s3);
            i += 32;
        }
        while i + 8 <= n {
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), s0);
            i += 8;
        }
    }
    let mut sum = hsum256(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

#[target_feature(enable = "avx2,fma")]
fn l2_sq_avx2(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut i = 0;
    let mut s0 = _mm256_setzero_ps();
    let mut s1 = _mm256_setzero_ps();
    let mut s2 = _mm256_setzero_ps();
    let mut s3 = _mm256_setzero_ps();
    unsafe {
        while i + 32 <= n {
            let d0 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)));
            let d1 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i + 8)), _mm256_loadu_ps(pb.add(i + 8)));
            let d2 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i + 16)), _mm256_loadu_ps(pb.add(i + 16)));
            let d3 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i + 24)), _mm256_loadu_ps(pb.add(i + 24)));
            s0 = _mm256_fmadd_ps(d0, d0, s0);
            s1 = _mm256_fmadd_ps(d1, d1, s1);
            s2 = _mm256_fmadd_ps(d2, d2, s2);
            s3 = _mm256_fmadd_ps(d3, d3, s3);
            i += 32;
        }
        while i + 8 <= n {
            let d = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)));
            s0 = _mm256_fmadd_ps(d, d, s0);
            i += 8;
        }
    }
    let mut sum = hsum256(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
    while i < n {
        let d = a[i] - b[i];
        sum += d * d;
        i += 1;
    }
    sum
}

#[target_feature(enable = "avx2,fma")]
fn dot_u8_avx2(q: &[f32], c: &[u8]) -> f32 {
    let n = q.len();
    let (pq, pc) = (q.as_ptr(), c.as_ptr());
    let mut i = 0;
    let mut s0 = _mm256_setzero_ps();
    let mut s1 = _mm256_setzero_ps();
    unsafe {
        while i + 16 <= n {
            let raw = _mm_loadu_si128(pc.add(i) as *const __m128i);
            let lo = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(raw));
            let hi = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128::<8>(raw)));
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(pq.add(i)), lo, s0);
            s1 = _mm256_fmadd_ps(_mm256_loadu_ps(pq.add(i + 8)), hi, s1);
            i += 16;
        }
        if i + 8 <= n {
            let raw = _mm_loadl_epi64(pc.add(i) as *const __m128i);
            let v = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(raw));
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(pq.add(i)), v, s0);
            i += 8;
        }
    }
    let mut sum = hsum256(_mm256_add_ps(s0, s1));
    while i < n {
        sum += q[i] * c[i] as f32;
        i += 1;
    }
    sum
}

#[target_feature(enable = "avx512f")]
fn dot_avx512(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut i = 0;
    let mut s0 = _mm512_setzero_ps();
    let mut s1 = _mm512_setzero_ps();
    unsafe {
        while i + 32 <= n {
            s0 = _mm512_fmadd_ps(_mm512_loadu_ps(pa.add(i)), _mm512_loadu_ps(pb.add(i)), s0);
            s1 = _mm512_fmadd_ps(_mm512_loadu_ps(pa.add(i + 16)), _mm512_loadu_ps(pb.add(i + 16)), s1);
            i += 32;
        }
        if i + 16 <= n {
            s0 = _mm512_fmadd_ps(_mm512_loadu_ps(pa.add(i)), _mm512_loadu_ps(pb.add(i)), s0);
            i += 16;
        }
        if i < n {
            // masked tail: dead lanes load as zero, so they add nothing
            let m: __mmask16 = (1u16 << (n - i)) - 1;
            let x = _mm512_maskz_loadu_ps(m, pa.add(i));
            let y = _mm512_maskz_loadu_ps(m, pb.add(i));
            s1 = _mm512_fmadd_ps(x, y, s1);
        }
    }
    _mm512_reduce_add_ps(_mm512_add_ps(s0, s1))
}

#[target_feature(enable = "avx512f")]
fn l2_sq_avx512(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut i = 0;
    let mut s0 = _mm512_setzero_ps();
    let mut s1 = _mm512_setzero_ps();
    unsafe {
        while i + 32 <= n {
            let d0 = _mm512_sub_ps(_mm512_loadu_ps(pa.add(i)), _mm512_loadu_ps(pb.add(i)));
            let d1 = _mm512_sub_ps(_mm512_loadu_ps(pa.add(i + 16)), _mm512_loadu_ps(pb.add(i + 16)));
            s0 = _mm512_fmadd_ps(d0, d0, s0);
            s1 = _mm512_fmadd_ps(d1, d1, s1);
            i += 32;
        }
        if i + 16 <= n {
            let d = _mm512_sub_ps(_mm512_loadu_ps(pa.add(i)), _mm512_loadu_ps(pb.add(i)));
            s0 = _mm512_fmadd_ps(d, d, s0);
            i += 16;
        }
        if i < n {
            let m: __mmask16 = (1u16 << (n - i)) - 1;
            let x = _mm512_maskz_loadu_ps(m, pa.add(i));
            let y = _mm512_maskz_loadu_ps(m, pb.add(i));
            let d = _mm512_sub_ps(x, y);
            s1 = _mm512_fmadd_ps(d, d, s1);
        }
    }
    _mm512_reduce_add_ps(_mm512_add_ps(s0, s1))
}

#[target_feature(enable = "avx512f")]
fn dot_u8_avx512(q: &[f32], c: &[u8]) -> f32 {
    let n = q.len();
    let (pq, pc) = (q.as_ptr(), c.as_ptr());
    let mut i = 0;
    let mut s0 = _mm512_setzero_ps();
    let mut s1 = _mm512_setzero_ps();
    unsafe {
        while i + 32 <= n {
            let r0 = _mm_loadu_si128(pc.add(i) as *const __m128i);
            let r1 = _mm_loadu_si128(pc.add(i + 16) as *const __m128i);
            let v0 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(r0));
            let v1 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(r1));
            s0 = _mm512_fmadd_ps(_mm512_loadu_ps(pq.add(i)), v0, s0);
            s1 = _mm512_fmadd_ps(_mm512_loadu_ps(pq.add(i + 16)), v1, s1);
            i += 32;
        }
        if i + 16 <= n {
            let r = _mm_loadu_si128(pc.add(i) as *const __m128i);
            let v = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(r));
            s0 = _mm512_fmadd_ps(_mm512_loadu_ps(pq.add(i)), v, s0);
            i += 16;
        }
    }
    let mut sum = _mm512_reduce_add_ps(_mm512_add_ps(s0, s1));
    while i < n {
        sum += q[i] * c[i] as f32;
        i += 1;
    }
    sum
}
