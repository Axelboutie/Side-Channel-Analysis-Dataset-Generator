use super::aes_sbox;
use super::utils::*;
use crate::Dataset;
use crate::LeakageModel;
use crate::Scheme;

use modular_arithmetic::mod_mul;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use statrs::statistics::Statistics;

impl Dataset {
    ///trace(n: integer, poi: Integer array, s(Secret): Integer array(8bits), sn(Noise value in dB): float, p(plaintext): Integer array(8bits)) -> f32 array
    /// --
    /// This function performs a trace (in the form of an integer array) with the hamming weight of SBOX[p^s] at the given poi or the hamming weight of a random between 0-255.

    pub(crate) fn trace(
        &self,
        n: usize,
        poi: &Vec<usize>,
        s: &Vec<i32>,
        sn: f64,
        p: &Vec<u8>,
        m: &Vec<Vec<i32>>,
        pond: &Vec<f32>,
        p_b: f32,
    ) -> (Vec<f32>, f32) {
        let mut a: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|_i| {
                let v = uni_u8();
                v as f32
            })
            .collect();
        if !self.masking.software.1 {
            for (j, val) in poi[(self.masking.order as usize * self.nb_bytes)..]
                .iter()
                .enumerate()
            {
                a[*val] = aes_sbox::aes_8(&p[j], &(s[j] as u8)) as f32;
            }
        } else {
            for (j, val) in poi[(self.masking.order as usize * self.nb_bytes)..]
                .iter()
                .enumerate()
            {
                a[*val] = self.fn_mask(aes_sbox::aes_8(&p[j], &(s[j] as u8)), j, m) as f32;
            }
        }

        if self.masking.software.1 {
            for (ind, mp) in m.iter().enumerate() {
                for (k, x) in poi[(ind * self.masking.order as usize)..(poi.len() - self.nb_bytes)]
                    .iter()
                    .enumerate()
                {
                    if k >= mp.len() {
                        break;
                    }
                    a[*x] = mp[k] as f32;
                }
            }
        } else if self.masking.hardware.1 {
            let temp: Vec<i32> = self.mask1d();

            a.iter_mut().for_each(|val| {
                if self.masking.hardware.0 == Scheme::Boolean {
                    let mut v = *val as i32 ^ xor(&temp);
                    for b in temp.iter() {
                        v = v << 8 | *b;
                    }
                    *val = v as f32;
                } else if self.masking.hardware.0 == Scheme::Multiplicative {
                    let mut v = mod_mul(*val as u64, mult_mod256(&temp) as u64, 256) as i32;
                    for b in temp.iter() {
                        v = v << 8 | *b;
                    }
                    *val = v as f32;
                } else {
                    let mut v = mod_mul(
                        *val as u64,
                        mult_mod256(&temp[..temp.len() - 1].to_vec()) as u64,
                        256,
                    ) as i32
                        ^ temp[temp.len() - 1];
                    for b in temp.iter() {
                        v = v << 8 | *b;
                    }
                    *val = v as f32;
                }
            });
        }
        let temp2 = a.clone();
        a.iter_mut().enumerate().for_each(|(i, val)| {
            *val = {
                if self.leakage_model == LeakageModel::HD {
                    if i == 0 {
                        hw(*val as i32) as f32
                    } else {
                        hd(temp2.clone()[i - 1] as i32, *val as i32) as f32
                    }
                } else if self.leakage_model == LeakageModel::HW {
                    hw(*val as i32) as f32
                } else if self.leakage_model == LeakageModel::WHW {
                    w_hw(*val as i32, pond)
                } else {
                    *val
                }
            }
        });

        let mut p_b1: f32 = p_b;
        if p_b1 == -1.0 {
            let moy = a
                .iter()
                .map(|x| x.powf(2.0) as f64)
                .collect::<Vec<_>>()
                .mean();
            p_b1 = sigma_snr(moy, sn);
            dbg!(p_b1);

            a.iter_mut().for_each(|x| *x += norm_f(p_b1.sqrt()) as f32);
        } else {
            a.iter_mut().for_each(|x| *x += norm_f(p_b.sqrt()) as f32);
        }

        (a, p_b1)
    }

    pub(crate) fn trace_fixed(
        &self,
        a: Vec<f32>,
        s: &Vec<i32>,
        sn: f64,
        poi: &Vec<usize>,
        p: &Vec<u8>,
        m: &Vec<Vec<i32>>,
        pond: &Vec<f32>,
        p_b: f32,
    ) -> (Vec<f32>, f32) {
        let mut temp = a.clone();

        temp.iter_mut()
            .for_each(|x| *x = if *x == -1.0f32 { uni_u8() as f32 } else { *x });

        if !self.masking.software.1 {
            for (j, val) in poi[(self.masking.order as usize * self.nb_bytes)..]
                .iter()
                .enumerate()
            {
                temp[*val] = aes_sbox::aes_8(&p[j], &(s[j] as u8)) as f32;
            }
        } else {
            for (j, val) in poi[(self.masking.order as usize * self.nb_bytes)..]
                .iter()
                .enumerate()
            {
                temp[*val] = self.fn_mask(aes_sbox::aes_8(&p[j], &(s[j] as u8)), j, m) as f32;
            }
        }

        if self.masking.software.1 {
            for (ind, mp) in m.iter().enumerate() {
                for (k, x) in poi[(ind * self.masking.order as usize)..(poi.len() - self.nb_bytes)]
                    .iter()
                    .enumerate()
                {
                    if k >= mp.len() {
                        break;
                    }
                    temp[*x] = mp[k] as f32;
                }
            }
        } else if self.masking.hardware.1 {
            let temp3: Vec<i32> = self.mask1d();

            temp.iter_mut().for_each(|val| {
                if self.masking.hardware.0 == Scheme::Boolean {
                    let mut v = *val as i32 ^ xor(&temp3);
                    for b in temp3.iter() {
                        v = v << 8 | *b;
                    }
                    *val = v as f32;
                } else if self.masking.hardware.0 == Scheme::Multiplicative {
                    let mut v = mod_mul(*val as u64, mult_mod256(&temp3) as u64, 256) as i32;
                    for b in temp3.iter() {
                        v = v << 8 | *b;
                    }
                    *val = v as f32;
                } else {
                    let mut v = mod_mul(
                        *val as u64,
                        mult_mod256(&temp3[..temp3.len() - 1].to_vec()) as u64,
                        256,
                    ) as i32
                        ^ temp3[temp3.len() - 1];
                    for b in temp3.iter() {
                        v = v << 8 | *b;
                    }
                    *val = v as f32;
                }
            });
        }
        let temp2 = temp.clone();
        temp.iter_mut().enumerate().for_each(|(i, val)| {
            *val = {
                if self.leakage_model == LeakageModel::HD {
                    if i == 0 {
                        hw(*val as i32) as f32
                    } else {
                        hd(temp2.clone()[i - 1] as i32, *val as i32) as f32
                    }
                } else if self.leakage_model == LeakageModel::HW {
                    hw(*val as i32) as f32
                } else if self.leakage_model == LeakageModel::WHW {
                    w_hw(*val as i32, pond)
                } else {
                    *val
                }
            }
        });

        let mut p_b1: f32 = p_b;
        if p_b1 == -1.0 {
            let moy = temp
                .iter()
                .map(|x| x.powf(2.0) as f64)
                .collect::<Vec<_>>()
                .mean();
            p_b1 = sigma_snr(moy, sn);
            dbg!(p_b1);
            temp.iter_mut()
                .for_each(|x| *x += norm_f(p_b1.sqrt()) as f32);
        } else {
            temp.iter_mut()
                .for_each(|x| *x += norm_f(p_b.sqrt()) as f32);
        }

        (temp, p_b1)
    }

    fn fn_mask(&self, val: u8, ind: usize, m: &Vec<Vec<i32>>) -> i32 {
        if self.masking.software.0 == Scheme::Boolean {
            val as i32 ^ xor(&m[ind])
        } else if self.masking.software.0 == Scheme::Multiplicative {
            mod_mul(val as u64, mult_mod256(&m[ind]) as u64, 256) as i32
        } else {
            mod_mul(
                val as u64,
                mult_mod256(&m[ind][..m[ind].len() - 1].to_vec()) as u64,
                256,
            ) as i32
                ^ m[ind][m[ind].len() - 1]
        }
    }
}
