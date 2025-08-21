use pyo3::prelude::*;
use rand::random_range;
use statrs::statistics::Statistics;
/// Correlation Power Analysis (CPA) implementation module
/// Provides functions for different types of power analysis attacks using correlation
use std::f64;
use std::vec;

use super::aes_sbox;
use super::utils::{hw, mean, parallel_matrix_multiplication, std_deviation, w_hw};

/// Calculates Pearson's correlation coefficient using a numerically stable algorithm
///
/// # Arguments
/// * `h` - Hypothetical power consumption values
/// * `t` - Matrix of measured power traces
///
/// # Returns
/// Vector of correlation coefficients
fn equi(h: Vec<f64>, t: &Vec<Vec<f64>>) -> Vec<f64> {
    let h_bar = h.clone().mean();
    let t_bar = mean(t);

    let coef: f32 = 1.0 / (t.len() as f32 - 1.0);

    let s_t = std_deviation(t);
    let s_h = h.clone().std_dev();

    let mut p1s: Vec<Vec<f64>> = vec![];

    for (i, v) in t.iter().enumerate() {
        let p: Vec<f64> = v
            .iter()
            .map(|x| x - t_bar[i])
            .map(|x| (x / s_t[i]) as f64)
            .collect();

        p1s.push(p);
    }

    let p2: Vec<f32> = h
        .iter()
        .map(|y| y - h_bar)
        .map(|y| (y / s_h) as f32)
        .collect();
    let sum = parallel_matrix_multiplication(&vec![p2], &p1s);

    let result: Vec<f64> = sum.iter().flatten().map(|x| (*x * coef) as f64).collect();

    result
}

// All the CPA under are using the same algorithms the only difference is on the leakage model

///cpa(trace: float's matrix, plaintext = Array of plaintext) -> (Key: u8, Correlation: Float's Array)
/// --
/// This function perform a Correlation Power Analysis on the trace that were given with the plaintext given
/// It is using the "Online" formula with the Pearson's Correlation define above
/// The Leakage model here is the plain value of the AES

#[pyfunction]
pub(crate) fn cpa(trace: Vec<Vec<f64>>, plaintext: Vec<u8>) -> (u8, [f64; 256]) {
    let mut maxcpa = [0.0; 256];
    for (kguess, mc) in maxcpa.iter_mut().enumerate() {
        let guess = plaintext
            .iter()
            .map(|ptext| (aes_sbox::aes_8(ptext, &(kguess as u8))) as f64)
            .collect();
        let cpaoutput: &Vec<f64> = &equi(guess, &trace); //.iter().map(|x| x.abs()).collect();
        *mc = cpaoutput.iter().abs_max();
    }

    let mut test = maxcpa;
    test.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let key: u8 = maxcpa.iter().position(|x| *x == test[0]).unwrap() as u8;

    (key, maxcpa)
}

///cpa(trace: float's matrix, plaintext = Array of plaintext) -> (Key: u8, Correlation: Float's Array)
/// --
/// This function perform a Correlation Power Analysis on the trace that were given with the plaintext given
/// It is using the "Online" formula with the Pearson's Correlation define above
/// The Leakage model here is the Hamming_Weight value of the AES

#[pyfunction]
pub(crate) fn cpa_hw(trace: Vec<Vec<f64>>, plaintext: Vec<u8>) -> (u8, [f64; 256]) {
    let mut maxcpa = [0.0; 256];
    for (kguess, mc) in maxcpa.iter_mut().enumerate() {
        let guess = plaintext
            .iter()
            .map(|ptext| hw(aes_sbox::aes_8(ptext, &(kguess as u8)) as i32) as f64)
            .collect();
        let cpaoutput: &Vec<f64> = &equi(guess, &trace); //.iter().map(|x| x.abs()).collect();

        *mc = cpaoutput.iter().abs_max();
    }

    let mut test = maxcpa;
    test.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let key: u8 = maxcpa.iter().position(|x| *x == test[255]).unwrap() as u8;

    (key, maxcpa)
}

///cpa(trace: float's matrix, plaintext = Array of plaintext, pond: Float's Array) -> (Key: u8, Correlation: Float's Array)
/// --
/// This function perform a Correlation Power Analysis on the trace that were given with the plaintext given
/// It is using the "Online" formula with the Pearson's Correlation define above
/// The Leakage model here is the ponderate Hamming_Weight value of the AES
/// The value in the ponderation array needs to be between 0 and 1 and the sum needs to be equal to 1

#[pyfunction]
pub(crate) fn cpa_whw(
    trace: Vec<Vec<f64>>,
    plaintext: Vec<u32>,
    weights: Vec<f32>,
) -> (u8, [f64; 256]) {
    let mut maxcpa = [0.0; 256];
    for (kguess, mc) in maxcpa.iter_mut().enumerate() {
        let guess = plaintext
            .iter()
            .map(|ptext| {
                w_hw(
                    aes_sbox::aes_8(&((*ptext).try_into().unwrap()), &(kguess as u8)) as i32,
                    &weights,
                ) as f64
            })
            .collect();
        let cpaoutput: &Vec<f64> = &equi(guess, &trace); //.iter().map(|x| x.abs()).collect();
        *mc = cpaoutput.iter().abs_max();
    }

    let mut test = maxcpa;
    test.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let key: u8 = maxcpa.iter().position(|x| *x == test[255]).unwrap() as u8;

    (key, maxcpa)
}

#[pyfunction]
pub(crate) fn socpa_hw(trace: Vec<Vec<f64>>, plaintext: Vec<u8>) -> (u8, [f64; 256]) {
    let mut maxcpa = [0.0; 256];
    for (kguess, mc) in maxcpa.iter_mut().enumerate() {
        let mut corr: Vec<f64> = Vec::new();
        let i = random_range(1..trace.len());
        for pt in trace[i].iter() {
            let guess = plaintext
                .iter()
                .map(|ptext| {
                    hw((aes_sbox::aes_8(ptext, &(kguess as u8)) ^ *pt as u8) as i32) as f64
                })
                .collect();
            let cpaoutout: &Vec<f64> = &equi(guess, &trace); //.iter().map(|x| x.abs()).collect();
            corr.push(cpaoutout.iter().abs_max())
        }
        *mc = corr.mean();
    }

    let mut test = maxcpa;
    test.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let key: u8 = maxcpa.iter().position(|x| *x == test[255]).unwrap() as u8;

    (key, maxcpa)
}
