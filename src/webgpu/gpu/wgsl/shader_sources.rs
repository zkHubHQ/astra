pub const BLS12_377_PARAMS_WGSL: &str = include_str!("bls12_377_params.wgsl");
pub const BN254_PARAMS_WGSL: &str = include_str!("bn254_params.wgsl");
pub const BLS12_377_CURVE_BASE_WGSL: &str = include_str!("bls12_377_curve_base.wgsl");
pub const BN254_CURVE_BASE_WGSL: &str = include_str!("bn254_curve_base.wgsl");
pub const BN254_CURVE_HALO2_BASE_WGSL: &str = include_str!("bn254_curve_halo2_base.wgsl");
pub const CURVE_WGSL: &str = include_str!("curve.wgsl");
pub const FIELD_MODULUS_WGSL: &str = include_str!("field_modulus.wgsl");
pub const U256_WGSL: &str = include_str!("u256.wgsl");

// Ugly (but optimized) WGSL shaders
pub const UGLY_BN254_CURVE_HALO2_BASE_WGSL: &str = include_str!("ugly/bn254_curve_halo2_base.wgsl");
pub const UGLY_BN254_PARAMS_WGSL: &str = include_str!("ugly/bn254_params.wgsl");
pub const UGLY_CURVE_WGSL: &str = include_str!("ugly/curve.wgsl");
pub const UGLY_FIELD_MODULUS_WGSL: &str = include_str!("ugly/field_modulus.wgsl");
pub const UGLY_U256_WGSL: &str = include_str!("ugly/u256.wgsl");
