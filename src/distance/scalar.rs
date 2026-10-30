pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, ra) = a.as_chunks::<8>();
    let (cb, rb) = b.as_chunks::<8>();
    for (x, y) in ca.iter().zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, ra) = a.as_chunks::<8>();
    let (cb, rb) = b.as_chunks::<8>();
    for (x, y) in ca.iter().zip(cb) {
        for i in 0..8 {
            let d = x[i] - y[i];
            acc[i] += d * d;
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        let d = x - y;
        s += d * d;
    }
    s
}

pub fn dot_u8(q: &[f32], codes: &[u8]) -> f32 {
    let mut acc = [0f32; 8];
    let (cq, rq) = q.as_chunks::<8>();
    let (cc, rc) = codes.as_chunks::<8>();
    for (x, y) in cq.iter().zip(cc) {
        for i in 0..8 {
            acc[i] += x[i] * y[i] as f32;
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in rq.iter().zip(rc) {
        s += x * *y as f32;
    }
    s
}

pub fn hamming(a: &[u32], b: &[u32]) -> u32 {
    a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
}
