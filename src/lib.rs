/// Side-Channel Analysis Dataset Generation Library
///
/// This library provides tools and utilities for generating datasets used in
/// side-channel analysis research and experimentation. It implements various
/// attack models, countermeasures, and analysis techniques.

/// All the external crate that is use in this project
use hdf5::File;
use ndarray::Array;
use pyo3::prelude::*;
use rand::Rng;
use rand::rng;
use rand::seq::SliceRandom;
use std::any::Any;
use std::any::TypeId;
use std::fmt;

// Reference modules for the crate structure
mod aes_sbox;
mod anova_snr;
mod cpa;
mod ge;
mod trace;
mod utils;

/// Model types for dataset generation
///
/// Enumerates all the models that can be used to generate datasets
/// for side-channel analysis experiments
#[derive(Debug, Copy, Clone, PartialEq)]
enum LeakageModel {
    HW,
    WHW, // Weighted Hamming_Weight
    ID,
    HD,
}

/// Display implementation for leakage_model enum
/// Provides string representation of model types
impl fmt::Display for LeakageModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Leakage Model : {}", self)
    }
}

/// Hiding countermeasure configuration
///
/// Represents the hiding countermeasure settings applied to the dataset
/// including various protection mechanisms against side-channel attacks
#[derive(Debug)]
struct Hiding {
    delay: bool,
    shuffle: bool,
    shuffle_f: bool,
    shuffle_m: bool,
}

/// Display implementation for Hiding configuration
impl fmt::Display for Hiding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "delay: {}, shuffle: {}, shuffle_full: {}, shuffle_m: {}",
            self.delay, self.shuffle, self.shuffle_f, self.shuffle_m
        )
    }
}

impl Hiding {
    /// The new function implement a new method that sets all the hiding CMs that needs to be set
    /// It takes a vector of String in argument and return a Hiding struct
    /// This method is only use with a python's list of string
    /// The shuffle full CM is an option for shuffling during a masking CM
    /// Shuffle full will shuffle every poi (even the mask part) rather than
    /// only the sbox poi for the shuffle

    fn new(v: Vec<String>) -> Self {
        let mut a = Hiding {
            delay: false,
            shuffle: false,
            shuffle_f: false,
            shuffle_m: false,
        };

        // Conversion from a String to a &str type because python's string is a String in rust
        // And then we use a for with a match to take all the value independently and set the hiding struct

        let value: Vec<&str> = v.iter().map(|x| x.as_str()).collect();
        for &item in value.iter() {
            match item {
                "delay" => a.delay = true,
                "shuffle" => a.shuffle = true,
                "shuffle_full" => a.shuffle_f = true,
                "shuffle_mask" => a.shuffle_m = true,
                "" => (),
                _ => println!("Check the value in your Hiding array"),
            }
        }
        a
    }
}

/// Masking scheme types
///
/// Defines the available masking countermeasure schemes
/// Nothing is the default value indicating no masking is applied
#[derive(Debug, Copy, Clone, PartialEq)]
enum Scheme {
    Nothing,
    Boolean,
    Multiplicative,
    Affine,
}

impl fmt::Display for Scheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

/// Masking countermeasure configuration
///
/// Configures the masking countermeasure parameters:
/// * order: Protection order of the masking scheme
/// * fixed: Use same mask for all traces when true
#[derive(Debug)]
struct Masking {
    software: (Scheme, bool),
    hardware: (Scheme, bool), //Enlevez possiblement l'hardware
    order: i32,
    fixed: bool,
}

/// Display implementation for Masking configuration
impl fmt::Display for Masking {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Masking's Scheme Software : {:?}, Masking's Scheme Hardware : {:?}, Order : {}, Fixed : {}",
            self.software.0, self.hardware.0, self.order, self.fixed
        )
    }
}

impl Masking {
    /// The new function implement a new method that sets all the masking CMs that needs to be set
    /// It takes a vector of String in argument and return a Masking struct
    /// This method is only use with a python's list of string
    /// A particular case is done with the parse method to convert the number of order from a string to a vec.

    fn new(v: Vec<String>) -> Self {
        let mut a = Masking {
            software: (Scheme::Nothing, false),
            hardware: (Scheme::Nothing, false),
            order: 0,
            fixed: false,
        };
        let value: Vec<&str> = v.iter().map(|x| x.as_str()).collect();
        for &item in value.iter() {
            match item {
                "Boolean, Software" => a.software = (Scheme::Boolean, true),
                "Boolean, Hardware" => a.hardware = (Scheme::Boolean, true),
                "Affine, Software" => a.software = (Scheme::Affine, true),
                "Affine, Hardware" => a.hardware = (Scheme::Affine, true),
                "Multiplicative, Software" => a.software = (Scheme::Multiplicative, true),
                "Multiplicative, Hardware" => a.hardware = (Scheme::Multiplicative, true),
                "Fixed" => a.fixed = true,
                "" => (),
                _ => {
                    //The item.parse convert the string value to a i32
                    //In the match case that I use, I verify if we are in the case of the number or an error case,
                    //that's why we compare the type of the item.parse

                    if item.parse::<i32>().expect("Parse error").type_id() == TypeId::of::<i32>() {
                        a.order = item.parse::<i32>().expect("Parse error")
                    } else {
                        println!("Check the value in your masking array")
                    }
                }
            }
        }
        a
    }
}

/// Dataset generation configuration and interface
/// Main structure for configuring and generating side-channel analysis datasets.
/// Provides the primary interface for Python users to interact with the library.
/// Combines both hiding and masking countermeasures with trace generation parameters.
#[pyclass]
#[pyo3(str)]
pub(crate) struct Dataset {
    leakage_model: LeakageModel,
    hiding: Hiding,
    masking: Masking,
    nb_traces: usize,
    nb_samples: usize,
    nb_bytes: usize,
    fixed_pattern: bool,
}

/// Implementation of the Trait Diplay to use the print in python

impl fmt::Display for Dataset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Number of Traces : {}, Number of Samples : {}, Number of Bytes : {}, {:?}, {:?}, Leakage Model : {:?}, Fixed Pattern : {} ",
            self.nb_traces,
            self.nb_samples,
            self.nb_bytes,
            self.hiding,
            self.masking,
            self.leakage_model,
            self.fixed_pattern
        )
    }
}

#[pymethods]
impl Dataset {
    /// New method to create the dataset via the __init__ method of python
    /// It uses the new method that we set for the struct

    #[new]
    fn new(
        leakage_model: &str,
        hiding: Vec<String>,
        masking: Vec<String>,
        nb_traces: usize,
        nb_samples: usize,
        nb_bytes: usize,
        fixed_pattern: bool,
    ) -> Self {
        let mut a = Dataset {
            leakage_model: LeakageModel::HW,
            hiding: Hiding::new(hiding),
            masking: Masking::new(masking),
            nb_traces: nb_traces,
            nb_samples: nb_samples,
            nb_bytes: nb_bytes,
            fixed_pattern: fixed_pattern,
        };
        match leakage_model {
            "Hamming_Weight" => a.leakage_model = LeakageModel::HW,
            "Weighted_HW" => a.leakage_model = LeakageModel::WHW,
            "ID" => a.leakage_model = LeakageModel::ID,
            "Hamming_Distance" => a.leakage_model = LeakageModel::HD,
            "" => (),
            _ => println!("Check the spelling for the leakage model."),
        }
        a
    }

    /// Series of setter and getter for all the usize value of the struct Dataset that is adapted to use with the setter/getter norm of python
    /// We decided to not implement any setter or getter for the struct of CM because on set of dataset will have his CMs
    /// and there is no need to be changed during all the execution

    #[setter]
    fn set_nb_traces(&mut self, value: usize) -> PyResult<()> {
        self.nb_traces = value;
        Ok(())
    }

    #[getter]
    fn get_nb_traces(&self) -> PyResult<usize> {
        Ok(self.nb_traces)
    }

    #[setter]
    fn set_nb_samples(&mut self, value: usize) -> PyResult<()> {
        self.nb_samples = value;
        Ok(())
    }

    #[getter]
    fn get_nb_samples(&self) -> PyResult<usize> {
        Ok(self.nb_samples)
    }

    #[setter]
    fn set_nb_bytes(&mut self, value: usize) -> PyResult<()> {
        self.nb_bytes = value;
        Ok(())
    }

    #[getter]
    fn get_nb_bytes(&self) -> PyResult<usize> {
        Ok(self.nb_bytes)
    }

    /// This method generates a HDF5 file for the dataset
    /// The file generate is using the exact same format as the ASCADv1 dataset from ANSSI-fr
    /// It is using the same names, groupe and idea
    /// The only difference for the moment is that the metadata dataset is not build around a compound type but around a group named metadata
    /// Therefore we have two set of traces that is created one for the profiling and the other for the attack
    /// IDtext1 is the IDtext for the attack set
    /// The Nb_att and nb_prof represent the number of traces that we want for the attack group and the profiling group respectively
    /// To generate the set of traces, labels and keys used, this method uses the method traces that is implemented below

    pub(crate) fn t_ref(&self, poi: Vec<usize>, random_rates: f64) -> Vec<f32> {
        let mut rng = rng();

        if random_rates > 1.0 || random_rates < 0.0 {
            panic!("You need to choose a rate between 0 and 1");
        }

        let r = 1.0 - random_rates;

        let a: Vec<f32> = (0..self.nb_samples as i32)
            .map(|val| {
                let mut x = if rng.random_bool(r) {
                    utils::uni_u8() as f32
                } else {
                    -1.0f32
                };
                x = if poi.contains(&(val as usize + 1)) {
                    utils::uni_u8() as f32
                } else {
                    x
                };
                if poi.contains(&(val as usize)) {
                    -2.0f32
                } else {
                    x
                }
            })
            .collect();

        a
    }

    fn gen_file(
        &mut self,
        snr: f64,
        weights: Option<Vec<f32>>,
        delay: Option<i32>,
        filename: String,
        nb_prof: i32,
        nb_att: i32,
        random_rates: f64,
        testing_set: bool,
        nb_testing: usize,
    ) {
        let file = File::create(&filename).expect("Error to create file");
        let mut grp = "Attack_traces";
        let mut plaintext = utils::gen_plaintext(self.nb_traces, self.nb_bytes);

        // This loop for will do the two groupe (Attack and profiling)
        let mask = if self.masking.fixed && self.masking.software.1 {
            self.mask()
        } else {
            Vec::new()
        };

        let ref_trace = {
            //References Trace
            if self.fixed_pattern {
                Some(self.t_ref(
                    utils::gen_poi(
                        self.nb_samples,
                        self.nb_bytes + (self.nb_bytes * self.masking.order as usize),
                    ),
                    random_rates,
                ))
            } else {
                Some(vec![0.0f32])
            }
        };

        let iter = if testing_set { 3usize } else { 2 };

        for i in 0..iter {
            if i == 0 {
                self.nb_traces = nb_att as usize;
            } else if i == 1 {
                self.nb_traces = nb_prof as usize;
                grp = "Profiling_traces";
                plaintext = utils::gen_plaintext(self.nb_traces, self.nb_bytes);
            } else {
                self.nb_traces = nb_testing;
                grp = "Testing_traces";
                plaintext = utils::gen_plaintext(self.nb_traces, self.nb_bytes);
            }

            let attack = file.create_group(grp).expect("group create failed");

            //Generation of the traces, labels and keys with the methode

            let (t, l, secret, ma) = self.traces(
                plaintext.clone(),
                snr,
                weights.clone(),
                delay,
                mask.clone(),
                ref_trace.clone(),
            );

            let rows = self.nb_traces;
            let cols = self.nb_samples;

            //Conversion of the set of traces into a Array type from ndarray
            //It is also done for the labels and keys

            let arr22 =
                Array::from_shape_vec((rows, cols), t.iter().flatten().copied().collect()).unwrap();

            let traces = attack
                .new_dataset::<f32>()
                .shape([rows, cols])
                .create("traces")
                .expect("Trace dataset");

            traces.write(&arr22).expect("write failed");

            let labels = attack
                .new_dataset::<f32>()
                .shape((self.nb_traces, self.nb_bytes))
                .create("labels")
                .expect("Labels DATASET");
            labels
                .write(
                    &Array::from_shape_vec(
                        (rows, self.nb_bytes),
                        l.iter().flatten().copied().collect(),
                    )
                    .unwrap(),
                )
                .expect("write failed");

            let metadata = attack.create_group("Metadata").unwrap();

            let pt = metadata
                .new_dataset::<u8>()
                .shape((self.nb_traces, self.nb_bytes))
                .create("plaintext")
                .unwrap();
            pt.write(
                &Array::from_shape_vec(
                    (rows, self.nb_bytes),
                    plaintext.iter().flatten().copied().collect(),
                )
                .unwrap(),
            )
            .expect("write failed");

            let k = metadata
                .new_dataset::<i32>()
                .shape(self.nb_bytes)
                .create("key")
                .unwrap();
            k.write(
                &Array::from_shape_vec(self.nb_bytes, secret.clone().into_iter().collect())
                    .unwrap(),
            )
            .expect("write failed");

            dbg!(ma[1].len());

            if self.masking.software.1 {
                let m = metadata
                    .new_dataset::<i32>()
                    .shape((self.nb_traces, self.nb_bytes))
                    .create("masks")
                    .unwrap();
                m.write(
                    &Array::from_shape_vec(
                        (self.nb_traces, self.nb_bytes),
                        ma.clone().into_iter().flatten().collect(),
                    )
                    .unwrap(),
                )
                .expect("write failed");
            }
        }
    }

    /// This method is internal for the dataset and create a vector with all the mask part we need for a masking CM
    /// The number of mask part that is created depends on the order of the mask and the number of bytes generated

    fn mask(&self) -> Vec<Vec<i32>> {
        (0..self.nb_bytes)
            .map(|_i| (0..self.masking.order).map(|_x| utils::uni_u8()).collect())
            .collect()
    }

    fn mask1d(&self) -> Vec<i32> {
        (0..self.masking.order).map(|_x| utils::uni_u8()).collect()
    }

    /// This method create the set of traces, labels and keys that it is generated in the dataset
    /// It can be used independently in python or it is use in the gen_file method
    /// This function is develop in a way to be modular and use with all the case, with all CMs

    fn traces(
        &self,
        plaintext: Vec<Vec<u8>>,
        snr: f64,
        weights: Option<Vec<f32>>,
        delay: Option<i32>,
        mask: Vec<Vec<i32>>,
        ref_traces: Option<Vec<f32>>,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<i32>, Vec<Vec<i32>>) {
        let mut poi = utils::gen_poi(
            self.nb_samples,
            self.nb_bytes + (self.nb_bytes * self.masking.order as usize),
        );
        //dbg!(&poi);
        let mut poi_c = poi.clone();
        let ref_t = ref_traces.unwrap();
        let mut ref_t_temp = ref_t.clone();

        // Each key follow a uniform distribution on a 8bits value in i32

        let s: Vec<i32> = (0..self.nb_bytes).map(|_i| utils::uni_u8()).collect();

        let w = weights.unwrap();

        let mut rng = rng();

        let mut maskage = mask;

        let mut mask_nf: Vec<Vec<i32>> = (0..self.nb_traces).map(|_i| vec![0]).collect();

        let mut p_b = -1.0f32;

        let traces: Vec<Vec<f32>> = (0..self.nb_traces)
            .map(|i| {
                ref_t_temp = ref_t.clone();
                if self.hiding.shuffle {
                    poi[(self.nb_bytes * self.masking.order as usize)..].shuffle(&mut rng); // This line will shuffle each poi associated with each value of bytes you are using, use in the shuffle CM
                    poi_c = poi.clone();
                } else if self.hiding.shuffle_f {
                    poi[(self.nb_bytes * self.masking.order as usize)..].shuffle(&mut rng);
                    let temp = poi.len();
                    poi[..(temp - self.nb_bytes)].shuffle(&mut rng);
                    // If you have masking CM on, it will also shuffle with the mask part
                    poi_c = poi.clone();
                } else {
                    poi[..(self.nb_bytes * self.masking.order as usize)].shuffle(&mut rng);
                    poi_c = poi.clone();
                }
                if self.hiding.delay {
                    // This function add/substract the random delay on the range set by the user
                    // And if the delay set exceed the range of the samples an error will be raise
                    if self.fixed_pattern {
                        poi_c = utils::delays_fixed(
                            delay.unwrap(),
                            &mut ref_t_temp,
                            &poi,
                            self.nb_samples,
                        );
                    } else {
                        poi_c = utils::delays(delay.unwrap(), &poi, self.nb_samples);
                    }
                }
                if self.masking.fixed {
                    // Masking fixed

                    if self.fixed_pattern {
                        let (a, p) = self.trace_fixed(
                            ref_t_temp.clone(),
                            &s,
                            snr,
                            &poi_c,
                            &plaintext[i],
                            &maskage,
                            &w,
                            p_b,
                        );
                        p_b = p;
                        a
                    } else {
                        // This trace function generates one trace for the Hamming Weight leakage model
                        let (a, p) = self.trace(
                            self.nb_samples,
                            &poi_c,
                            &s,
                            snr,
                            &plaintext[i],
                            &maskage,
                            &w,
                            p_b,
                        );
                        p_b = p;
                        a
                    }
                } else {
                    if self.fixed_pattern {
                        let temp = self.mask();
                        mask_nf[i] = if self.masking.software.0 == Scheme::Boolean {
                            temp.iter().map(|x| utils::xor(x)).collect()
                        } else if self.masking.software.0 == Scheme::Multiplicative {
                            temp.iter().map(|x| utils::mult_mod256(x)).collect()
                        } else {
                            temp.iter().map(|x| utils::mult_mod256(x)).collect()
                        };
                        let (a, p) = self.trace_fixed(
                            ref_t_temp.clone(),
                            &s,
                            snr,
                            &poi_c,
                            &plaintext[i],
                            &temp,
                            &w,
                            p_b,
                        );
                        p_b = p;
                        a
                    } else {
                        let temp = self.mask();
                        mask_nf[i] = if self.masking.software.0 == Scheme::Boolean {
                            temp.iter().map(|x| utils::xor(x)).collect()
                        } else if self.masking.software.0 == Scheme::Multiplicative {
                            temp.iter().map(|x| utils::mult_mod256(x)).collect()
                        } else {
                            temp.iter().map(|x| utils::mult_mod256(x)).collect()
                        };
                        // This trace function generates one trace for the Hamming Weight leakage model
                        let (a, p) = self.trace(
                            self.nb_samples,
                            &poi_c,
                            &s,
                            snr,
                            &plaintext[i],
                            &temp,
                            &w,
                            p_b,
                        );
                        p_b = p;
                        a
                    }
                }
            })
            .collect();
        // dbg!(poi);
        let labels: Vec<Vec<f32>> = (0..self.nb_traces)
            .map(|i| {
                (0..self.nb_bytes)
                    .map(|o| aes_sbox::aes_8(&plaintext[i][o], &(s[o] as u8)) as f32)
                    .collect()
            })
            .collect();

        if self.masking.fixed {
            let maks: Vec<i32> = maskage.iter().map(|x| utils::xor(x)).collect();

            maskage = (0..self.nb_traces).map(|_i| maks.clone()).collect();
        } else {
            maskage = mask_nf
        }
        (traces, labels, s, maskage)
    }
}

///This function takes all the Rust Object that is referenced inside and convert them to python Object to be use in python
/// Only object that have #[py...] before them can be added in this function
/// If you want more information on how this works, you can check the pyo3 documentation

#[pymodule]
fn sca_dataset(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();

    m.add_function(wrap_pyfunction!(anova_snr::anova_snr, m)?)?;
    m.add_function(wrap_pyfunction!(cpa::cpa, m)?)?;
    m.add_function(wrap_pyfunction!(cpa::cpa_hw, m)?)?;
    m.add_function(wrap_pyfunction!(cpa::cpa_whw, m)?)?;
    m.add_function(wrap_pyfunction!(cpa::socpa_hw, m)?)?;
    m.add_function(wrap_pyfunction!(ge::ge, m)?)?;
    m.add_function(wrap_pyfunction!(ge::soge, m)?)?;
    m.add_function(wrap_pyfunction!(utils::gen_poi, m)?)?;

    m.add_class::<Dataset>()?;
    Ok(())
}
