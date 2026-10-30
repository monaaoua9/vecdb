mod common;

use common::Rng;
use vecdb::distance::Kernels;

fn rel_close(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-3 * (1.0 + a.abs().max(b.abs()))
}

#[test]
fn every_kernel_matches_scalar() {
    let mut r = Rng::new(7);
    let all = Kernels::available();
    println!("kernels under test: {:?}", all.iter().map(|k| k.name).collect::<Vec<_>>());
    for dim in (1..=130).chain([255, 256, 257, 768, 1023, 1536]) {
        let a: Vec<f32> = (0..dim).map(|_| r.gauss()).collect();
        let b: Vec<f32> = (0..dim).map(|_| r.gauss()).collect();
        let c: Vec<u8> = (0..dim).map(|_| (r.next_u64() & 0xff) as u8).collect();
        for k in &all {
            assert!(rel_close((k.dot)(&a, &b), (Kernels::SCALAR.dot)(&a, &b)), "{} dot dim={dim}", k.name);
            assert!(rel_close((k.l2_sq)(&a, &b), (Kernels::SCALAR.l2_sq)(&a, &b)), "{} l2 dim={dim}", k.name);
            assert!(
                rel_close((k.dot_u8)(&a, &c), (Kernels::SCALAR.dot_u8)(&a, &c)),
                "{} dot_u8 dim={dim}",
                k.name
            );
        }
    }
}

#[test]
fn l2_of_identical_vectors_is_zero() {
    let v: Vec<f32> = (0..97).map(|i| i as f32 * 0.1).collect();
    for k in Kernels::available() {
        assert_eq!((k.l2_sq)(&v, &v), 0.0, "{}", k.name);
    }
}

#[test]
fn empty_slices() {
    for k in Kernels::available() {
        assert_eq!((k.dot)(&[], &[]), 0.0);
        assert_eq!((k.l2_sq)(&[], &[]), 0.0);
        assert_eq!((k.dot_u8)(&[], &[]), 0.0);
    }
}

#[test]
fn hamming_matches_scalar() {
    let mut r = Rng::new(11);
    for words in (1..=40).chain([48, 96]) {
        let a: Vec<u32> = (0..words).map(|_| r.next_u64() as u32).collect();
        let b: Vec<u32> = (0..words).map(|_| r.next_u64() as u32).collect();
        let want: u32 = a.iter().zip(&b).map(|(x, y)| (x ^ y).count_ones()).sum();
        assert_eq!(vecdb::distance::hamming(&a, &b), want, "words={words}");
        assert_eq!(vecdb::distance::scalar::hamming(&a, &b), want);
    }
}
