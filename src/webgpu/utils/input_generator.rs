use group::{prime::PrimeCurveAffine, Curve};
use pasta_curves::arithmetic::CurveAffine;

use crate::webgpu::gpu::utils::{generate_random_affine_point, generate_random_scalar_point};

pub struct PointScalarInput<C: CurveAffine> {
    pub point: C,
    pub scalar: C::Scalar,
}

pub fn point_scalar_generator<C: CurveAffine>(input_size: usize) -> Vec<PointScalarInput<C>> {
    let mut inputs = Vec::new();
    for _ in 0..input_size {
        let random_point = generate_random_affine_point::<C>();
        let random_scalar = generate_random_scalar_point::<C>();
        inputs.push(PointScalarInput {
            point: random_point,
            scalar: random_scalar,
        });
    }
    inputs
}
