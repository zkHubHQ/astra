# Astra: Accelerate Client-Side ZK with WebGPU

## About

This repository is an ambitious effort to accelerate Zero Knowledge cryptography in the browser using WebGPU.

Right now, most operations are supported on BLS12-377 & BN-254.

Support operations include:
 * Field Math
 * Curve Math
 * Multi-Scalar Multiplications (MSMs)
 * Number Theoretic Transforms (NTTs aka FFTs)

 You can try out the library by running the [pippengen_msm file](src/webgpu/gpu/entries/bn256_algorithms/pippenger_msm_entry.rs).

 This library is very much a work in progress and is not yet ready for production use. It is also not yet optimized for performance, but will be pretty soon. highly recommend following this repository to stay updated on the progress!
