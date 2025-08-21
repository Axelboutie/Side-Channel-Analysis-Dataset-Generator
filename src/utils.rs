use modular_arithmetic::mod_mul;
use pyo3::pyfunction;
use rand::Rng;
use rand::distr::Uniform;
use rand_distr::{Distribution, Normal};
use statrs::statistics::Statistics;
use std::collections::HashSet;

/// Utility functions for data processing and statistical analysis
/// This module contains various helper functions for matrix operations, statistical calculations,
/// and random number generation used in side-channel analysis.

/// Extracts values at a specific position from all arrays in the matrix
///
/// # Arguments
/// * `matrix` - A slice of vectors containing elements of type T
/// * `col` - The position to extract values from
///
/// # Returns
/// A vector containing all values at the specified position
pub(crate) fn extract_column<T: std::marker::Copy>(matrice: &[Vec<T>], col: usize) -> Vec<T> {
    matrice
        .iter()
        .filter_map(|tableau| tableau.get(col).copied())
        .collect()
}
/// Extracts values at a specific position from selected arrays based on indices
///
/// # Arguments
/// * `arrays` - Vector of vectors containing f64 values
/// * `index` - Slice of index specifying which arrays to use
/// * `col` - Position to extract values from
pub(crate) fn extract_column_index_labels(
    arrays: &Vec<Vec<f64>>,
    index: &[usize],
    col: usize,
) -> Vec<f64> {
    index.iter().map(|&idx| arrays[idx][col]).collect()
}

/// Finds all index where a specific value occurs in a vector
pub(crate) fn index_of<T: PartialEq>(vec: &[T], value: &T) -> Vec<usize> {
    vec.iter()
        .enumerate()
        .filter_map(|(i, v)| if v == value { Some(i) } else { None })
        .collect()
}

/// Returns unique values from a vector of f64 numbers
/// Uses bit representation for float comparison
pub(crate) fn unique(matrix: &[f64]) -> Vec<f64> {
    let mut uniques = Vec::new();
    let mut seen = HashSet::new();

    for &val in matrix.iter() {
        let val_bits = val.to_bits();
        if seen.insert(val_bits) {
            uniques.push(val);
        }
    }

    uniques
}

/// Calculates the mean of each vector in the matrix
pub(crate) fn mean(matrix: &Vec<Vec<f64>>) -> Vec<f64> {
    matrix.iter().map(|x| x.mean()).collect()
}

/// Calculates the standard deviation of each vector in the matrix
pub(crate) fn std_deviation(matrix: &Vec<Vec<f64>>) -> Vec<f64> {
    matrix.iter().map(|x| x.std_dev()).collect()
}

use rayon::prelude::*;

/// Performs parallel matrix multiplication using Rayon
///
/// # Arguments
/// * `a` - First matrix
/// * `b` - Second matrix
///
/// # Returns
/// Result of matrix multiplication as f32
pub(crate) fn parallel_matrix_multiplication(
    a: &Vec<Vec<f32>>,
    b: &Vec<Vec<f64>>,
) -> Vec<Vec<f32>> {
    let n = a.len();
    let m = b[0].len();

    let result: Vec<Vec<i32>> = (0..n)
        .into_par_iter()
        .map(|i| {
            (0..m)
                .map(|j| {
                    (0..a[0].len())
                        .map(|k| (a[i][k] as i32) * (b[k][j] as i32))
                        .sum()
                })
                .collect()
        })
        .collect();
    let mut res = vec![Vec::new()];
    for (i, v) in result.iter().enumerate() {
        res[i] = convert(v);
    }

    res
}

/// Converts a vector of i32 values to f32 values
///
/// # Arguments
/// * `b` - Slice of i32 values to convert
///
/// # Returns
/// A vector containing the converted f32 values
fn convert(b: &[i32]) -> Vec<f32> {
    b.iter().map(|x| *x as f32).collect()
}

/// Calculates voltage based on signal power and SNR
///
/// # Arguments
/// * `p_s` - Signal power
/// * `snr` - Signal-to-noise ratio in dB
pub(crate) fn sigma_snr(p_s: f64, snr: f64) -> f32 {
    let div: f32 = (snr / 10.0) as f32;

    let dem = f32::powf(10.0, div);

    p_s as f32 / dem
}

/// Generates a uniform random integer between 0 and 255
pub(crate) fn uni_u8() -> i32 {
    let mut rng = rand::rng();
    let test = Uniform::new(0, 255).unwrap();
    let x = rng.sample(test);

    x as i32
}

/// Generates a matrix of plaintext to use for the traces
///
/// # Arguments
/// * `nbt` - Number of traces in the set
/// * `nbb` - Number of bytes that needs plaintext
pub(crate) fn gen_plaintext(nbt: usize, nbb: usize) -> Vec<Vec<u8>> {
    (0..nbt)
        .map(|_f| (0..nbb).map(|_val| uni_u8() as u8).collect())
        .collect()
}

/// Generates a normally distributed random float
///
/// # Arguments
/// * `std` - Standard deviation for the normal distribution
pub(crate) fn norm_f(std: f32) -> f32 {
    let normal = Normal::new(0.0, std).unwrap();
    let v = normal.sample(&mut rand::rng());
    v
}

/// Calculates Hamming weight of an integer
pub(crate) fn hw(n: i32) -> i32 {
    let s = format!("{n:b}");
    let m = s.matches("1");

    m.count() as i32
}

/// Generates points of interest based on sample size and number of points
#[pyfunction]
pub(crate) fn gen_poi(nbs: usize, nbb: usize) -> Vec<usize> {
    let intervalle = nbs / (nbb + 1);
    (1..=nbb).map(|i| intervalle * i).collect()
}

/// Calculates weighted Hamming weight using provided weights
pub(crate) fn w_hw(n: i32, weights: &Vec<f32>) -> f32 {
    let s = format!("{n:b}");
    s.chars()
        .zip(weights.iter())
        .fold(0.0, |acc, (bit, weight)| {
            let bit_value = bit
                .to_digit(2)
                .expect("The string's value needs to be either 0 or 1");
            let a = acc as f32 + (bit_value as f32 * weight);
            a
        })
}

/// Calculates the Hamming distance between two integers
///
/// # Arguments
/// * `a` - First integer value
/// * `b` - Second integer value
///
/// # Returns
/// The Hamming distance between a and b
pub(crate) fn hd(a: i32, b: i32) -> i32 {
    hw(a ^ b)
}

/// Applies delays to points of interest within sample size constraints
pub(crate) fn delays(d: i32, poi: &Vec<usize>, nbs: usize) -> Vec<usize> {
    let mut rng = rand::rng();
    let distr = Uniform::new(0, d).unwrap();
    let x = rng.sample(distr);

    poi.iter()
        .map(|p| {
            if *p as i32 + x > nbs as i32 || *p as i32 + x < 0 {
                panic!("The delay is too big");
            } else {
                (*p as i32 + x) as usize
            }
        })
        .collect::<Vec<_>>()
}

pub(crate) fn delays_fixed(
    d: i32,
    ref_trace: &mut Vec<f32>,
    poi: &Vec<usize>,
    nbs: usize,
) -> Vec<usize> {
    let mut rng = rand::rng();
    let distr = Uniform::new(0, d).unwrap();
    let x = rng.sample(distr);

    poi.iter()
        .map(|p| {
            if *p as i32 + x > nbs as i32 || *p as i32 + x < 0 {
                panic!("The delay is too big");
            } else {
                let temp = (*p as i32 + x) as usize;
                let temp2 = ref_trace[*p - 1];
                ref_trace[*p] = ref_trace[temp];
                ref_trace[*p - 1] = ref_trace[temp - 1];
                ref_trace[temp - 1] = temp2;

                temp
            }
        })
        .collect::<Vec<_>>()
}

/// Performs a bitwise XOR operation on all elements in the vector
///
/// # Arguments
/// * `t` - Vector of integers to XOR together
///
/// # Returns
/// The result of XORing all elements
pub(crate) fn xor(t: &Vec<i32>) -> i32 {
    let mut res = 0;
    for i in t.iter() {
        res = res ^ i;
    }
    res
}

/// Performs a modular multiplication of all elements in the vector
///
/// # Arguments
/// * `t` - Vector of integers to multiply together
///
/// # Returns
/// The result of multiplying all elements with modulo 256
pub(crate) fn mult_mod256(t: &Vec<i32>) -> i32 {
    let mut res = 1;
    for i in t.iter() {
        res = mod_mul(res as u64, *i as u64, 256) as i32
    }
    res
}

/// Generates a logarithmically spaced list of integers
///
/// # Arguments
/// * `max_value` - Maximum value in the list
/// * `num_points` - Number of points to generate
pub(crate) fn logspace(max_value: f64, num_points: usize) -> Vec<i32> {
    let mut logarithmic_list = Vec::new();
    let log_max = max_value.ln();
    let log_min = 10.0_f64.ln();

    for i in 0..num_points {
        let log_value = log_min + (log_max - log_min) * (i as f64) / ((num_points - 1) as f64);
        let value = log_value.exp();
        logarithmic_list.push(value);
    }

    logarithmic_list.iter().map(|x| *x as i32).collect()
}
