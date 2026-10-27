use std::arch::aarch64::*;

use super::Kernels;

// NEON is part of the aarch64 baseline, no runtime check needed.
pub(super) fn neon() -> Option<Kernels> {
    Some(Kernels { name: "neon", dot, l2_sq, dot_u8 })
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut i = 0;
    let mut sum;
    unsafe {
        let mut s0 = vdupq_n_f32(0.0);
        let mut s1 = vdupq_n_f32(0.0);
        let mut s2 = vdupq_n_f32(0.0);
        let mut s3 = vdupq_n_f32(0.0);
        while i + 16 <= n {
            s0 = vfmaq_f32(s0, vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
            s1 = vfmaq_f32(s1, vld1q_f32(pa.add(i + 4)), vld1q_f32(pb.add(i + 4)));
            s2 = vfmaq_f32(s2, vld1q_f32(pa.add(i + 8)), vld1q_f32(pb.add(i + 8)));
            s3 = vfmaq_f32(s3, vld1q_f32(pa.add(i + 12)), vld1q_f32(pb.add(i + 12)));
            i += 16;
        }
        while i + 4 <= n {
            s0 = vfmaq_f32(s0, vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
            i += 4;
        }
        sum = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    }
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut i = 0;
    let mut sum;
    unsafe {
        let mut s0 = vdupq_n_f32(0.0);
        let mut s1 = vdupq_n_f32(0.0);
        while i + 8 <= n {
            let d0 = vsubq_f32(vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
            let d1 = vsubq_f32(vld1q_f32(pa.add(i + 4)), vld1q_f32(pb.add(i + 4)));
            s0 = vfmaq_f32(s0, d0, d0);
            s1 = vfmaq_f32(s1, d1, d1);
            i += 8;
        }
        if i + 4 <= n {
            let d = vsubq_f32(vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
            s0 = vfmaq_f32(s0, d, d);
            i += 4;
        }
        sum = vaddvq_f32(vaddq_f32(s0, s1));
    }
    while i < n {
        let d = a[i] - b[i];
        sum += d * d;
        i += 1;
    }
    sum
}

fn dot_u8(q: &[f32], c: &[u8]) -> f32 {
    assert_eq!(q.len(), c.len());
    let n = q.len();
    let (pq, pc) = (q.as_ptr(), c.as_ptr());
    let mut i = 0;
    let mut sum;
    unsafe {
        let mut s0 = vdupq_n_f32(0.0);
        let mut s1 = vdupq_n_f32(0.0);
        while i + 8 <= n {
            let w = vmovl_u8(vld1_u8(pc.add(i)));
            let lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w)));
            let hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w)));
            s0 = vfmaq_f32(s0, vld1q_f32(pq.add(i)), lo);
            s1 = vfmaq_f32(s1, vld1q_f32(pq.add(i + 4)), hi);
            i += 8;
        }
        sum = vaddvq_f32(vaddq_f32(s0, s1));
    }
    while i < n {
        sum += q[i] * c[i] as f32;
        i += 1;
    }
    sum
}
