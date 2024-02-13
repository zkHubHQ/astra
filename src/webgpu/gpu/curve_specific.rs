use ff::PrimeField;
use num_bigint::BigInt;
use num_traits::Num;

use crate::bn256::{Fq, Fr};

use super::wgsl::{
    BLS12_377_CURVE_BASE_WGSL, BLS12_377_PARAMS_WGSL, BN254_CURVE_BASE_WGSL, BN254_CURVE_HALO2_BASE_WGSL, BN254_PARAMS_WGSL
};

#[derive(Debug, Clone, Copy)]
pub enum CurveType {
    BLS12_377,
    BN254,
}

pub const WORKGROUP_SIZE: usize = 64;

pub fn get_modulus(curve: CurveType) -> BigInt {
    match curve {
        CurveType::BLS12_377 => BigInt::from_str_radix(Fq::MODULUS, 16).unwrap(), // TODO: This is not correct
        CurveType::BN254 => BigInt::from_str_radix(Fq::MODULUS, 16).unwrap(),
    }
}

pub fn get_scalar_modulus(curve: CurveType) -> BigInt {
    match curve {
        CurveType::BLS12_377 => BigInt::from_str_radix(Fr::MODULUS, 16).unwrap(), // TODO: This is not correct
        CurveType::BN254 => BigInt::from_str_radix(Fr::MODULUS, 16).unwrap(),
    }
}

pub fn get_curve_params_wgsl(curve: CurveType) -> &'static str {
    match curve {
        CurveType::BLS12_377 => BLS12_377_PARAMS_WGSL,
        CurveType::BN254 => BN254_PARAMS_WGSL,
    }
}

pub fn get_curve_base_functions_wgsl(curve: CurveType) -> &'static str {
    match curve {
        CurveType::BLS12_377 => BLS12_377_CURVE_BASE_WGSL,
        CurveType::BN254 => BN254_CURVE_HALO2_BASE_WGSL,
    }
}
