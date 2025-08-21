/// Signal-to-Noise Ratio (SNR) analysis module
/// Implements SNR calculation methods for evaluating side-channel leakage
use average::WeightedMean;
use ndarray::Array1;
use pyo3::prelude::*;
use statrs::statistics::Statistics;

use super::utils;

/// Calculates Signal-to-Noise Ratio (SNR) using ANOVA methodology
///
/// # Arguments
/// * `labels` - Vector of labels (typically key-dependent values)
/// * `traces` - Matrix of power measurement traces
///
/// # Returns
/// Vector of SNR values for each time sample
#[pyfunction]
pub(crate) fn anova_snr(labels: Vec<f64>, traces: Vec<Vec<f64>>) -> Vec<f64> {
    let mut mean = vec![Vec::new()];
    let mut var = vec![Vec::new()];
    //let mut vm = vec![Vec::new()];

    let mut weight = Vec::new();

    //This loop will go around all the value of the labels one time

    for v in utils::unique(&labels) {
        let idx = utils::index_of(&labels, &v);
        let mut meani = Vec::new();
        let mut vari = Vec::new();
        //let mut vmi = Vec::new();
        for i in 0..traces[0].len() {
            meani.push(utils::extract_column_index_labels(&traces, &idx, i).mean());
            vari.push(utils::extract_column_index_labels(&traces, &idx, i).variance());

            //vmi.push(VarianceMean::from_traces(&traces, &idx, i));
        }
        mean.push(meani);
        var.push(vari);
        //vm.push(vmi);
        weight.push(idx.len()) //The weight is the occurence of every labels' value
    }

    //let mean = &mean[1..0];
    mean.remove(0);
    var.remove(0);
    // let var = &var[1..0];

    let mut o_x = Vec::new();
    let mut o_b = Vec::new();

    for j in 0..mean[0].len() {
        let values = utils::extract_column(&var, j);
        let mut o_bi = WeightedMean::new();
        o_x.push(utils::extract_column(&mean, j).variance());
        for (&value, &w) in values.iter().zip(weight.iter()) {
            o_bi.add(value, w as f64)
        }
        o_b.push(o_bi.mean());
    }
    let a = Array1::from(o_x);
    let b = Array1::from(o_b);
    let c = &a / &b;

    Array1::to_vec(&c)
}
