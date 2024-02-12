use ff::{Field, PrimeField};
use group::{prime::PrimeCurveAffine, Curve, Group, GroupEncoding};
use num_bigint::{BigInt, ToBigInt};
use num_traits::{One, ToPrimitive, Zero};
use pasta_curves::arithmetic::CurveAffine;
use rand::Rng;
use rand_core::OsRng;

use crate::bn256::G1Affine;

use super::curve_specific::{get_modulus, get_scalar_modulus, CurveType};

pub fn bigints_to_u16_array(be_bigints: &[BigInt]) -> Vec<u16> {
    let mut u16_array = Vec::new();
    for big_int in be_bigints {
        u16_array.extend(big_int_to_u16_array(big_int));
    }
    u16_array
}

// Helper function to convert a single BigInt to a Vec<u16>
pub fn big_int_to_u16_array(be_bigint: &BigInt) -> Vec<u16> {
    let num_bits = 256;
    let bits_per_element = 16;
    let num_elements = num_bits / bits_per_element;
    let mask = (BigInt::one() << bits_per_element) - BigInt::one();

    (0..num_elements)
        .map(|i| {
            let shift_amount = bits_per_element * (num_elements - 1 - i);
            let value: BigInt = ((be_bigint >> shift_amount) & &mask);
            value.to_u16().expect("Value should fit into u16")
        })
        .collect()
}

pub fn bigints_to_u32_array(be_bigints: &[BigInt]) -> Vec<u32> {
    let mut u32_array = Vec::new();
    for big_int in be_bigints {
        u32_array.extend(big_int_to_u32_array(big_int));
    }
    u32_array
}

// Helper function to convert a single BigInt to a Vec<u32>
pub fn big_int_to_u32_array(be_bigint: &BigInt) -> Vec<u32> {
    let num_bits = 256;
    let bits_per_element = 32;
    let num_elements = num_bits / bits_per_element;
    let mask = (BigInt::one() << bits_per_element) - BigInt::one();

    (0..num_elements)
        .map(|i| {
            let shift_amount = bits_per_element * (num_elements - 1 - i);
            let value: BigInt = ((be_bigint >> shift_amount) & &mask);
            value.to_u32().unwrap()
        })
        .collect()
}

pub fn u32_array_to_bigints(u32_array: &[u32]) -> Vec<BigInt> {
    u32_array
        .chunks(8)
        .map(|chunk| {
            chunk.iter().fold(BigInt::zero(), |acc, &val| {
                (acc << 32) + val.to_bigint().unwrap()
            })
        })
        .collect()
}

// Struct to hold gpuU32Inputs equivalent data
#[derive(Debug, Clone)]
pub struct GpuU32Inputs {
    pub u32_inputs: Vec<u32>,
    pub individual_input_size: usize,
}

pub fn gpu_u32_puppeteer_string(gpu_u32_input: &GpuU32Inputs) -> String {
    let inputs_str = gpu_u32_input
        .u32_inputs
        .iter()
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(", ");

    format!(
        "{{ u32Inputs: Uint32Array.from([{}]), individualInputSize: {} }}",
        inputs_str, gpu_u32_input.individual_input_size
    )
}

pub fn chunk_array(inputs_array: &[GpuU32Inputs], batch_size: usize) -> Vec<Vec<GpuU32Inputs>> {
    let first_input_length = inputs_array.first().unwrap().u32_inputs.len()
        / inputs_array.first().unwrap().individual_input_size;
    (0..first_input_length)
        .step_by(batch_size)
        .map(|index| {
            inputs_array
                .iter()
                .map(|input| {
                    let start_index = index * input.individual_input_size;
                    let end_index = std::cmp::min(
                        input.u32_inputs.len(),
                        (index + batch_size) * input.individual_input_size,
                    );
                    let sliced_inputs = input.u32_inputs[start_index..end_index].to_vec();
                    GpuU32Inputs {
                        u32_inputs: sliced_inputs,
                        individual_input_size: input.individual_input_size,
                    }
                })
                .collect()
        })
        .collect()
}

pub fn chunk_gpu_inputs(
    inputs_array: &[GpuU32Inputs],
    batch_size: usize,
    inputs_to_batch: &[usize],
) -> Vec<Vec<GpuU32Inputs>> {
    let num_inputs = inputs_array.first().unwrap().u32_inputs.len()
        / inputs_array.first().unwrap().individual_input_size;
    let mut chunked_array = Vec::new();

    let mut num_batched = 0;
    while num_batched < num_inputs {
        let mut chunked_inputs = Vec::new();
        for (i, input) in inputs_array.iter().enumerate() {
            let should_batch = inputs_to_batch.is_empty() || inputs_to_batch.contains(&i);
            if should_batch {
                let start_index = num_batched * input.individual_input_size;
                let end_index = std::cmp::min(
                    input.u32_inputs.len(),
                    (num_batched + batch_size) * input.individual_input_size,
                );
                let sliced_inputs = input.u32_inputs[start_index..end_index].to_vec();
                chunked_inputs.push(GpuU32Inputs {
                    u32_inputs: sliced_inputs,
                    individual_input_size: input.individual_input_size,
                });
            } else {
                // If not batching this input, include it as is in every chunk
                chunked_inputs.push(input.clone());
            }
        }
        num_batched += batch_size;
        chunked_array.push(chunked_inputs);
    }

    chunked_array
}

#[derive(Debug, Clone)]
pub struct FieldU32Array {
    // Implicitly stores the field inputs in a u32 array in the little endian format
    pub u32_array: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct CurveU32Array {
    // Implicitly stores the curve inputs in a u32 array in the little endian format
    pub x_u32_array: Vec<u32>,
    pub y_u32_array: Vec<u32>,
}

// Returns the u32 array representation of a bn256 curve point in the big endian format
pub fn convert_bn256_curve_to_u32_array(curve: &G1Affine) -> CurveU32Array {
    let x_u32_array = curve
        .x
        .to_repr()
        .chunks(4)
        .rev() // Reverse the chunks to convert to big endian
        .map(|chunk| {
            let mut bytes = [0u8; 4];
            bytes.copy_from_slice(chunk);
            u32::from_le_bytes(bytes)
        })
        .collect();

    let y_u32_array = curve
        .y
        .to_repr()
        .chunks(4)
        .rev() // Reverse the chunks to convert to big endian
        .map(|chunk| {
            let mut bytes = [0u8; 4];
            bytes.copy_from_slice(chunk);
            u32::from_le_bytes(bytes)
        })
        .collect();

    CurveU32Array {
        x_u32_array,
        y_u32_array,
    }
}

// Returns the u32 array representation of a bn256 scalar in the big endian format
pub fn convert_bn256_scalar_to_u32_array(
    scalar: &<G1Affine as PrimeCurveAffine>::Scalar,
) -> FieldU32Array {
    let u32_array = scalar
        .to_repr()
        .chunks(4)
        .rev() // Reverse the chunks to convert to big endian
        .map(|chunk| {
            let mut bytes = [0u8; 4];
            bytes.copy_from_slice(chunk);
            u32::from_le_bytes(bytes)
        })
        .collect();

    FieldU32Array { u32_array }
}

// Convert big endian u32 array to bn256 curve point
pub fn convert_u32_array_to_bn256_curve(u32_array: &CurveU32Array) -> G1Affine {
    let mut x_bytes = [0u8; 32];
    for (i, chunk) in u32_array.x_u32_array.iter().rev().enumerate() {
        let chunk_bytes = &chunk.to_le_bytes();
        x_bytes[i * 4..(i + 1) * 4].copy_from_slice(chunk_bytes);
    }
    let x = crate::bn256::Fq::from_repr(x_bytes).unwrap();

    let mut y_bytes = [0u8; 32];
    for (i, chunk) in u32_array.y_u32_array.iter().rev().enumerate() {
        let chunk_bytes = &chunk.to_le_bytes();
        y_bytes[i * 4..(i + 1) * 4].copy_from_slice(chunk_bytes);
    }
    let y = crate::bn256::Fq::from_repr(y_bytes).unwrap();

    G1Affine::from_xy(x, y).unwrap()
}

// Convert big endian u32 array to bn256 scalar
pub fn convert_u32_array_to_bn256_scalar(
    u32_array: &Vec<u32>,
) -> <G1Affine as PrimeCurveAffine>::Scalar {
    let mut bytes = [0u8; 32];
    for (i, chunk) in u32_array.iter().rev().enumerate() {
        let chunk_bytes = &chunk.to_le_bytes();
        bytes[i * 4..(i + 1) * 4].copy_from_slice(chunk_bytes);
    }
    let scalar = crate::bn256::Fr::from_repr(bytes).unwrap();

    scalar
}

pub fn generate_random_field(curve: CurveType) -> BigInt {
    create_random_number(&get_modulus(curve))
}

pub fn generate_random_scalar(curve: CurveType) -> BigInt {
    create_random_number(&get_scalar_modulus(curve))
}

pub fn generate_random_affine_point<C: CurveAffine>() -> C {
    C::Curve::random(OsRng).to_affine()
}

pub fn generate_random_scalar_point<C: CurveAffine>() -> C::Scalar {
    C::Scalar::random(OsRng)
}

// Function to generate random BigInts within a curve's field modulus
pub fn generate_random_fields(input_size: usize, curve: CurveType) -> Vec<BigInt> {
    (0..input_size)
        .map(|_| create_random_number(&get_modulus(curve)))
        .collect()
}

pub fn generate_random_scalars(input_size: usize, curve: CurveType) -> Vec<BigInt> {
    (0..input_size)
        .map(|_| create_random_number(&get_scalar_modulus(curve)))
        .collect()
}

pub fn concatenate_vectors(vec1: &Vec<u32>, vec2: &Vec<u32>) -> Vec<u32> {
    let mut combined = Vec::with_capacity(vec1.len() + vec2.len());
    combined.extend(vec1.iter().cloned());
    combined.extend(vec2.iter().cloned());
    combined
}

// Helper function to create a random BigInt number
fn create_random_number(modulus: &BigInt) -> BigInt {
    let mut rng = rand::thread_rng();

    // Use `rng.gen::<u32>()` to generate a random u32 value, avoiding overflow.
    let big_int_value: BigInt = (0..8)
        .map(|_| {
            // Directly convert each random u32 to a BigInt and shift it
            BigInt::from(rng.gen::<u32>())
        })
        // Accumulate into a single BigInt, shifting each by 32 bits to the left to concatenate them
        .enumerate()
        .fold(BigInt::zero(), |acc, (i, val)| acc + (val << (32 * i)));

    big_int_value % modulus
}

// Converts bigints to wasm fields format
pub fn convert_bigints_to_wasm_fields(bigints: &[BigInt]) -> Vec<String> {
    bigints
        .iter()
        .map(|bigint| format!("{}field", bigint))
        .collect()
}

// Strips the "field" suffix from a string
pub fn strip_field_suffix(field: &str) -> String {
    field
        .get(..field.len().checked_sub(5).unwrap_or_else(|| field.len()))
        .unwrap_or_default()
        .to_string()
}

// Strips the "group" suffix from a string
pub fn strip_group_suffix(group: &str) -> String {
    group
        .get(..group.len().checked_sub(5).unwrap_or_else(|| group.len()))
        .unwrap_or_default()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    // Helper function to convert string representations of big integers and arrays to their corresponding values.
    fn bigint_from_str(s: &str) -> BigInt {
        BigInt::parse_bytes(s.as_bytes(), 10).unwrap()
    }

    // Test data as specified in the TypeScript tests.
    fn test_data() -> Vec<(BigInt, Vec<u32>)> {
        vec![
            (bigint_from_str("0"), vec![0, 0, 0, 0, 0, 0, 0, 0]),
            (bigint_from_str("1"), vec![0, 0, 0, 0, 0, 0, 0, 1]),
            (bigint_from_str("33"), vec![0, 0, 0, 0, 0, 0, 0, 33]),
            (bigint_from_str("4294967297"), vec![0, 0, 0, 0, 0, 0, 1, 1]),
            (bigint_from_str("115792089237316195423570985008687907853269984665640564039457584007913129639935"), vec![4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295]),
            (bigint_from_str("6924886788847882060123066508223519077232160750698452411071850219367055984476"), vec![256858326, 3006847798, 1208683936, 2370827163, 3854692792, 1079629005, 1919445418, 2787346268]),
            (bigint_from_str("60001509534603559531609739528203892656505753216962260608619555"), vec![0, 9558, 3401397337, 1252835688, 2587670639, 1610789716, 3992821760, 136227]),
            (bigint_from_str("30000754767301779765804869764101946328252876608481130304309778"), vec![0, 4779, 1700698668, 2773901492, 1293835319, 2952878506, 1996410880, 68114]),
            // (bigint_from_str("5472060717959818805561601436314318772174077789324455915672259473661306552146"), vec![0, 0, 0, 0, 0, 0, 0, 0]),
        ]
    }

    #[test]
    fn test_big_int_to_u32_array() {
        for (big_int, expected) in test_data() {
            assert_eq!(
                big_int_to_u32_array(&big_int),
                expected,
                "Testing big_int_to_u32_array for {:?}",
                big_int
            );
        }
    }

    #[test]
    fn test_bigints_to_u32_array() {
        let (bigints, expected): (Vec<_>, Vec<_>) = test_data().into_iter().unzip();
        let result = bigints_to_u32_array(&bigints);
        assert_eq!(result, expected.concat(), "Testing bigints_to_u32_array");
    }

    #[test]
    fn test_u32_array_to_bigints() {
        for (expected_bigint, u32_array) in test_data() {
            let result = u32_array_to_bigints(&u32_array);
            assert_eq!(
                result,
                vec![expected_bigint],
                "Testing u32_array_to_bigints for {:?}",
                u32_array
            );
        }
    }
}
