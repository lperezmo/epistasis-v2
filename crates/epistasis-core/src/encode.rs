//! encode_vectors kernel: uint8 binary_packed -> int8 encoded with intercept column.

use ndarray::{ArrayView2, ArrayViewMut2, Axis};
use rayon::prelude::*;

#[derive(Clone, Copy)]
pub enum ModelType {
    Global,
    Local,
}

pub fn validate_binary(binary_packed: ArrayView2<'_, u8>) -> Result<(), String> {
    for &x in binary_packed.iter() {
        if x > 1 {
            return Err("binary_packed entries must be 0 or 1.".into());
        }
    }
    Ok(())
}

/// Rayon parallel threshold (in total output cells). Encode is a memory-bound
/// linear transform: NumPy's vectorized `1 - 2*x` is hard to beat without a
/// chunky input, so the threshold is set high to keep the serial fast path
/// for typical epistasis workloads (L up to ~14).
const PAR_THRESHOLD: usize = 1 << 18;

#[inline]
fn encode_row(in_row: ndarray::ArrayView1<'_, u8>, mut out_row: ndarray::ArrayViewMut1<'_, i8>, model: ModelType) {
    let n_bits = in_row.len();
    out_row[0] = 1;
    match model {
        ModelType::Global => {
            for k in 0..n_bits {
                out_row[k + 1] = 1 - 2 * (in_row[k] as i8);
            }
        }
        ModelType::Local => {
            for k in 0..n_bits {
                out_row[k + 1] = in_row[k] as i8;
            }
        }
    }
}

pub fn encode_into(
    binary_packed: ArrayView2<'_, u8>,
    mut out: ArrayViewMut2<'_, i8>,
    model: ModelType,
) {
    let (n, n_bits) = binary_packed.dim();
    debug_assert_eq!(out.dim(), (n, n_bits + 1));

    let total_cells = n.saturating_mul(n_bits + 1);
    if total_cells < PAR_THRESHOLD {
        for (in_row, out_row) in binary_packed.axis_iter(Axis(0)).zip(out.axis_iter_mut(Axis(0))) {
            encode_row(in_row, out_row, model);
        }
    } else {
        out.axis_iter_mut(Axis(0))
            .into_par_iter()
            .zip(binary_packed.axis_iter(Axis(0)).into_par_iter())
            .for_each(|(out_row, in_row)| {
                encode_row(in_row, out_row, model);
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    #[test]
    fn global_all_wt() {
        let bp = array![[0u8, 0, 0]];
        let mut out = Array2::<i8>::zeros((1, 4));
        encode_into(bp.view(), out.view_mut(), ModelType::Global);
        assert_eq!(out.row(0).to_vec(), vec![1, 1, 1, 1]);
    }

    #[test]
    fn global_all_mut() {
        let bp = array![[1u8, 1, 1]];
        let mut out = Array2::<i8>::zeros((1, 4));
        encode_into(bp.view(), out.view_mut(), ModelType::Global);
        assert_eq!(out.row(0).to_vec(), vec![1, -1, -1, -1]);
    }

    #[test]
    fn local_copies() {
        let bp = array![[0u8, 1, 1], [1, 0, 0]];
        let mut out = Array2::<i8>::zeros((2, 4));
        encode_into(bp.view(), out.view_mut(), ModelType::Local);
        assert_eq!(out.row(0).to_vec(), vec![1, 0, 1, 1]);
        assert_eq!(out.row(1).to_vec(), vec![1, 1, 0, 0]);
    }

    #[test]
    fn validate_rejects_two() {
        let bp = array![[0u8, 2]];
        assert!(validate_binary(bp.view()).is_err());
    }

    #[test]
    fn validate_accepts_binary() {
        let bp = array![[0u8, 1, 0, 1]];
        assert!(validate_binary(bp.view()).is_ok());
    }
}
