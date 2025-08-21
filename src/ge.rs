/// Guessing Entropy implementation module
/// Provides functions for evaluating the effectiveness of side-channel attacks
use pyo3::pyfunction;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use statrs::statistics::Statistics;

use crate::{
    cpa::{cpa_hw, socpa_hw},
    utils::logspace,
};

/// Performs Guessing Entropy analysis using Hamming Weight CPA
///
/// # Arguments
/// * `trace` - Matrix of power measurement traces
/// * `plaintext` - Vector of plaintext bytes
/// * `int` - Number of CPA iterations to perform
/// * `key` - Correct key value for reference
/// * `nb_pts` - Number of points for logarithmic sampling
///
/// # Returns
/// Tuple containing rank progression and number of traces used
#[pyfunction]
pub(crate) fn ge(
    trace: Vec<Vec<f64>>,
    plaintext: Vec<u8>,
    int: i32,
    key: u8,
    nb_pts: usize,
) -> (Vec<i32>, Vec<i32>) {
    let it = logspace(trace.len() as f64 - 1.0, nb_pts);
    let mut rank: Vec<i32> = Vec::new();

    for i in it.iter() {
        dbg!(i);
        let mut rn: Vec<i32> = (0..int).collect();
        rn.par_iter_mut().enumerate().for_each(|(_j, val)| {
            let (_k, corel) = cpa_hw(
                trace.clone()[..(*i as usize)].to_vec(),
                plaintext.clone()[..(*i as usize)].to_vec(),
            );
            let mut test = corel;
            test.sort_by(|a, b| a.partial_cmp(b).unwrap());
            test.reverse();
            *val = test.iter().position(|x| *x == corel[key as usize]).unwrap() as i32;
            // Finding the position of the right key
        });
        let res = rn.iter().map(|x| *x as f64).mean();
        rank.push((rn.iter().map(|x| *x as f64).mean()) as i32);
        if res == 0.0 {
            break;
        }
    }
    (rank, it)
}

/// Performs Guessing Entropy analysis using Hamming Weight CPA Second Order
///
/// # Arguments
/// * `trace` - Matrix of power measurement traces
/// * `plaintext` - Vector of plaintext bytes
/// * `int` - Number of CPA iterations to perform
/// * `key` - Correct key value for reference
/// * `nb_pts` - Number of points for logarithmic sampling
///
/// # Returns
/// Tuple containing rank progression and number of traces used

#[pyfunction]
pub(crate) fn soge(
    trace: Vec<Vec<f64>>,
    plaintext: Vec<u8>,
    int: i32,
    key: u8,
    nb_pts: usize,
) -> (Vec<i32>, Vec<i32>) {
    let it = logspace(trace.len() as f64 - 1.0, nb_pts);
    let mut rank: Vec<i32> = Vec::new();

    for i in it.iter() {
        // dbg!(i);
        let mut rn: Vec<i32> = (0..int).collect();
        rn.par_iter_mut().enumerate().for_each(|(_j, val)| {
            let (_k, corel) = socpa_hw(
                trace.clone()[..(*i as usize)].to_vec(),
                plaintext.clone()[..(*i as usize)].to_vec(),
            );
            let mut test = corel;
            test.sort_by(|a, b| a.partial_cmp(b).unwrap());
            test.reverse();
            *val = test.iter().position(|x| *x == corel[key as usize]).unwrap() as i32;
            // Finding the position of the right key
        });
        let res = rn.iter().map(|x| *x as f64).mean();
        rank.push((rn.iter().map(|x| *x as f64).mean()) as i32);
        if res == 0.0 {
            break;
        }
    }
    (rank, it)
}
