//! Fast Walsh-Hadamard Transform (iterative, in-place).
//!
//! For an input vector of length n = 2^L, the unnormalized FWHT H satisfies
//! `H^T H = n * I`. In epistasis this means the full-order OLS fit under
//! global (Hadamard) encoding is `beta_hat = (1/n) * fwht(y)`, computable
//! in `O(n log n)` without materializing the `n x n` design matrix.
//!
//! The butterfly pattern: at each stage h = 1, 2, 4, ..., n/2, pair up
//! `(a, b)` where `b = a + h` and `a` has the h-bit clear, then write
//! `a' = a + b, b' = a - b`.

pub fn fwht_inplace_f64(data: &mut [f64]) {
    let n = data.len();
    assert!(n.is_power_of_two(), "fwht requires length to be a power of two");
    let mut h: usize = 1;
    while h < n {
        let step = h * 2;
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let x = data[j];
                let y = data[j + h];
                data[j] = x + y;
                data[j + h] = x - y;
            }
            i += step;
        }
        h = step;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fwht_length_one_is_identity() {
        let mut v = vec![3.5];
        fwht_inplace_f64(&mut v);
        assert_eq!(v, vec![3.5]);
    }

    #[test]
    fn fwht_length_two() {
        let mut v = vec![1.0, 2.0];
        fwht_inplace_f64(&mut v);
        assert_eq!(v, vec![3.0, -1.0]);
    }

    #[test]
    fn fwht_length_four_matches_hadamard_matrix() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0];
        fwht_inplace_f64(&mut v);
        // H4 = [[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]] applied to v:
        //   row 0: 1+2+3+4 = 10
        //   row 1: 1-2+3-4 = -2
        //   row 2: 1+2-3-4 = -4
        //   row 3: 1-2-3+4 = 0
        assert_eq!(v, vec![10.0, -2.0, -4.0, 0.0]);
    }

    #[test]
    fn fwht_is_its_own_inverse_up_to_n() {
        let original = vec![1.0, -2.5, 0.0, 7.0, -3.0, 4.5, 1.25, -0.5];
        let mut v = original.clone();
        fwht_inplace_f64(&mut v);
        fwht_inplace_f64(&mut v);
        let n = original.len() as f64;
        for (a, b) in v.iter().zip(original.iter()) {
            assert!((a - n * b).abs() < 1e-12, "{} vs {}*{}", a, n, b);
        }
    }

    #[test]
    fn fwht_gram_is_n_identity() {
        // A column of H_n dotted with itself equals n; distinct columns dot to 0.
        // Equivalently, applying FWHT to a unit basis vector e_k yields row k of H.
        let n = 8;
        let mut cols: Vec<Vec<f64>> = Vec::new();
        for k in 0..n {
            let mut e = vec![0.0; n];
            e[k] = 1.0;
            fwht_inplace_f64(&mut e);
            cols.push(e);
        }
        for i in 0..n {
            for j in 0..n {
                let dot: f64 = (0..n).map(|k| cols[i][k] * cols[j][k]).sum();
                if i == j {
                    assert!((dot - n as f64).abs() < 1e-12);
                } else {
                    assert!(dot.abs() < 1e-12);
                }
            }
        }
    }
}
