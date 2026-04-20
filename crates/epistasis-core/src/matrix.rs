//! build_model_matrix kernel: elementwise product of selected encoded columns.
//!
//! Sites are passed in flat layout: `sites_flat` concatenates all site indices
//! and `sites_offsets` has length `m + 1` where site `j` occupies
//! `sites_flat[offsets[j]..offsets[j + 1]]`. This avoids `Vec<Vec<usize>>`
//! heap fragmentation and keeps the parallel inner loop branchless.

use ndarray::{ArrayView2, ArrayViewMut2, Axis};
use rayon::prelude::*;

pub fn validate_sites(
    sites_flat: &[i64],
    sites_offsets: &[usize],
    vec_len: usize,
) -> Result<(), String> {
    if sites_offsets.is_empty() {
        return Err("sites_offsets must have at least one entry.".into());
    }
    if *sites_offsets.last().unwrap() != sites_flat.len() {
        return Err(format!(
            "sites_offsets last entry ({}) does not match sites_flat length ({}).",
            sites_offsets.last().unwrap(),
            sites_flat.len()
        ));
    }
    for w in sites_offsets.windows(2) {
        if w[1] < w[0] {
            return Err("sites_offsets must be non-decreasing.".into());
        }
    }
    for (j, w) in sites_offsets.windows(2).enumerate() {
        let start = w[0];
        let end = w[1];
        if start == end {
            return Err(format!("site at index {} is empty.", j));
        }
        for &idx in &sites_flat[start..end] {
            if idx < 0 || (idx as usize) >= vec_len {
                return Err(format!(
                    "site at index {} has index {} out of range for encoded width {}.",
                    j, idx, vec_len
                ));
            }
        }
    }
    Ok(())
}

const PAR_THRESHOLD: usize = 1 << 15;

#[inline]
fn fill_column(
    encoded: ArrayView2<'_, i8>,
    idx: &[i64],
    mut col: ndarray::ArrayViewMut1<'_, i8>,
) {
    let n = encoded.nrows();
    if idx.len() == 1 {
        let c = idx[0] as usize;
        let src = encoded.column(c);
        for i in 0..n {
            col[i] = src[i];
        }
        return;
    }
    for i in 0..n {
        let mut acc: i8 = 1;
        for &k in idx {
            acc = acc.wrapping_mul(encoded[[i, k as usize]]);
        }
        col[i] = acc;
    }
}

pub fn build_into(
    encoded: ArrayView2<'_, i8>,
    sites_flat: &[i64],
    sites_offsets: &[usize],
    mut out: ArrayViewMut2<'_, i8>,
) {
    let (n, _vec_len) = encoded.dim();
    let m = sites_offsets.len() - 1;
    debug_assert_eq!(out.dim(), (n, m));

    let total_cells = n.saturating_mul(m);
    if total_cells < PAR_THRESHOLD {
        for j in 0..m {
            let start = sites_offsets[j];
            let end = sites_offsets[j + 1];
            let idx = &sites_flat[start..end];
            fill_column(encoded, idx, out.column_mut(j));
        }
    } else {
        out.axis_iter_mut(Axis(1))
            .into_par_iter()
            .enumerate()
            .for_each(|(j, col)| {
                let start = sites_offsets[j];
                let end = sites_offsets[j + 1];
                let idx = &sites_flat[start..end];
                fill_column(encoded, idx, col);
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    #[test]
    fn intercept_column_copies_ones() {
        let enc = array![[1i8, 1, 1], [1, -1, 1], [1, 1, -1], [1, -1, -1]];
        let sites_flat = vec![0i64, 1, 2];
        let sites_offsets = vec![0usize, 1, 2, 3];
        let mut out = Array2::<i8>::zeros((4, 3));
        build_into(enc.view(), &sites_flat, &sites_offsets, out.view_mut());
        assert_eq!(out.column(0).to_vec(), vec![1, 1, 1, 1]);
        assert_eq!(out.column(1).to_vec(), vec![1, -1, 1, -1]);
        assert_eq!(out.column(2).to_vec(), vec![1, 1, -1, -1]);
    }

    #[test]
    fn pair_is_product() {
        let enc = array![[1i8, 1, 1], [1, -1, 1], [1, 1, -1], [1, -1, -1]];
        let sites_flat = vec![1i64, 2];
        let sites_offsets = vec![0usize, 2];
        let mut out = Array2::<i8>::zeros((4, 1));
        build_into(enc.view(), &sites_flat, &sites_offsets, out.view_mut());
        assert_eq!(out.column(0).to_vec(), vec![1, -1, -1, 1]);
    }

    #[test]
    fn hadamard_2site_full() {
        let enc = array![
            [1i8, 1, 1],
            [1, 1, -1],
            [1, -1, 1],
            [1, -1, -1]
        ];
        let sites_flat = vec![0i64, 1, 2, 1, 2];
        let sites_offsets = vec![0usize, 1, 2, 3, 5];
        let mut out = Array2::<i8>::zeros((4, 4));
        build_into(enc.view(), &sites_flat, &sites_offsets, out.view_mut());
        let expected = array![
            [1i8, 1, 1, 1],
            [1, 1, -1, -1],
            [1, -1, 1, -1],
            [1, -1, -1, 1]
        ];
        assert_eq!(out, expected);
    }

    #[test]
    fn validate_rejects_empty_site() {
        assert!(validate_sites(&[], &[0, 0], 3).is_err());
    }

    #[test]
    fn validate_rejects_out_of_range() {
        assert!(validate_sites(&[5], &[0, 1], 3).is_err());
    }

    #[test]
    fn validate_rejects_negative() {
        assert!(validate_sites(&[-1], &[0, 1], 3).is_err());
    }

    #[test]
    fn validate_accepts_good_sites() {
        assert!(validate_sites(&[0, 1, 2], &[0, 1, 2, 3], 3).is_ok());
    }
}
