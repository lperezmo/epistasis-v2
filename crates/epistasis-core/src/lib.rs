//! epistasis-core: Rust hot-path kernels for epistasis-v2.
//!
//! Exposed to Python as `epistasis._core` via PyO3.

mod encode;
mod fwht;
mod matrix;

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn parse_model_type(s: &str) -> PyResult<encode::ModelType> {
    match s {
        "global" => Ok(encode::ModelType::Global),
        "local" => Ok(encode::ModelType::Local),
        other => Err(PyValueError::new_err(format!(
            "model_type must be 'global' or 'local'; got {:?}.",
            other
        ))),
    }
}

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyfunction]
#[pyo3(signature = (binary_packed, model_type = "global"))]
fn encode_vectors<'py>(
    py: Python<'py>,
    binary_packed: PyReadonlyArray2<'py, u8>,
    model_type: &str,
) -> PyResult<Bound<'py, PyArray2<i8>>> {
    let model = parse_model_type(model_type)?;
    let input = binary_packed.as_array();
    encode::validate_binary(input).map_err(PyValueError::new_err)?;

    let (n, n_bits) = input.dim();
    let out = PyArray2::<i8>::zeros(py, [n, n_bits + 1], false);
    {
        let output_view = unsafe { out.as_array_mut() };
        py.allow_threads(move || {
            encode::encode_into(input, output_view, model);
        });
    }
    Ok(out)
}

#[pyfunction]
fn build_model_matrix<'py>(
    py: Python<'py>,
    encoded: PyReadonlyArray2<'py, i8>,
    sites_flat: PyReadonlyArray1<'py, i64>,
    sites_offsets: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray2<i8>>> {
    let enc_view = encoded.as_array();
    let (n, vec_len) = enc_view.dim();

    let flat = sites_flat.as_slice().map_err(|_| {
        PyValueError::new_err("sites_flat must be a C-contiguous 1D int64 array.")
    })?;
    let offsets_i64 = sites_offsets.as_slice().map_err(|_| {
        PyValueError::new_err("sites_offsets must be a C-contiguous 1D int64 array.")
    })?;
    let offsets: Vec<usize> = offsets_i64
        .iter()
        .map(|&x| {
            if x < 0 {
                Err(PyValueError::new_err("sites_offsets entries must be non-negative."))
            } else {
                Ok(x as usize)
            }
        })
        .collect::<PyResult<Vec<_>>>()?;

    matrix::validate_sites(flat, &offsets, vec_len).map_err(PyValueError::new_err)?;

    let m = offsets.len() - 1;
    let out = PyArray2::<i8>::zeros(py, [n, m], false);
    {
        let output_view = unsafe { out.as_array_mut() };
        py.allow_threads(move || {
            matrix::build_into(enc_view, flat, &offsets, output_view);
        });
    }
    Ok(out)
}

#[pyfunction(name = "fwht")]
fn fwht_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let input = data.as_slice().map_err(|_| {
        PyValueError::new_err("data must be a C-contiguous 1D float64 array.")
    })?;
    let n = input.len();
    if n == 0 || !n.is_power_of_two() {
        return Err(PyValueError::new_err(format!(
            "fwht requires a length that is a power of two; got {}.",
            n
        )));
    }

    let mut buf = input.to_vec();
    py.allow_threads(|| {
        fwht::fwht_inplace_f64(&mut buf);
    });

    let out = PyArray1::<f64>::from_vec(py, buf);
    Ok(out)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(encode_vectors, m)?)?;
    m.add_function(wrap_pyfunction!(build_model_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(fwht_py, m)?)?;
    Ok(())
}
