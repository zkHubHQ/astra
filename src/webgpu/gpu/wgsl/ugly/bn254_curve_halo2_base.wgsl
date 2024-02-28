const ZERO_POINT = UglyPoint (U256_ZERO, U256_ONE, U256_ZERO, U256_ZERO);
const ZERO_AFFINE = UglyAffinePoint (U256_ZERO, U256_ONE);

fn mul_by_3b(input: array<u32, 8>) -> array<u32, 8> {
    let doubled = field_double(input); // Double once
    let doubledTwice = field_double(doubled); // Double twice
    let doubledThrice = field_double(doubledTwice); // Double thrice
    return field_add(doubledThrice, input); // Add the original input
}

fn add_points(p1: UglyPoint, p2: UglyPoint) -> UglyPoint {
    var T0 = field_multiply(p1.x, p2.x);
    var T1 = field_multiply(p1.y, p2.y);
    var T2 = field_multiply(p1.z, p2.z);
    var T3 = field_add(p1.x, p1.y);
    var T4 = field_add(p2.x, p2.y);
    T3 = field_multiply(T3, T4);
    T4 = field_add(T0, T1);
    T3 = field_sub(T3, T4);
    T4 = field_add(p1.y, p1.z);
    var X3 = field_add(p2.y, p2.z);
    T4 = field_multiply(T4, X3);
    X3 = field_add(T1, T2);
    T4 = field_sub(T4, X3);
    X3 = field_add(p1.x, p1.z);
    var Y3 = field_add(p2.x, p2.z);
    X3 = field_multiply(X3, Y3);
    Y3 = field_add(T0, T2);
    Y3 = field_sub(X3, Y3);
    X3 = field_add(T0, T0);
    T0 = field_add(X3, T0);
    T2 = mul_by_3b(T2);
    var Z3 = field_add(T1, T2);
    T1 = field_sub(T1, T2);
    Y3 = mul_by_3b(Y3);
    X3 = field_multiply(T4, Y3);
    T2 = field_multiply(T3, T1);
    X3 = field_sub(T2, X3);
    Y3 = field_multiply(Y3, T0);
    T1 = field_multiply(T1, Z3);
    Y3 = field_add(T1, Y3);
    T0 = field_multiply(T0, T3);
    Z3 = field_multiply(Z3, T4);
    Z3 = field_add(Z3, T0);
    var T5 = field_multiply(X3, Y3);
    return UglyPoint(X3, Y3, T5, Z3);
}

fn double_point(p: UglyPoint) -> UglyPoint {
    var T0 = field_multiply(p.y, p.y);
    var Z3 = field_add(T0, T0);
    Z3 = field_add(Z3, Z3);
    Z3 = field_add(Z3, Z3);
    var T1 = field_multiply(p.y, p.z);
    var T2 = field_multiply(p.z, p.z);
    T2 = mul_by_3b(T2);
    var X3 = field_multiply(T2, Z3);
    var Y3 = field_add(T0, T2);
    Z3 = field_multiply(T1, Z3);
    T1 = field_add(T2, T2);
    T2 = field_add(T1, T2);
    T0 = field_sub(T0, T2);
    Y3 = field_multiply(T0, Y3);
    Y3 = field_add(X3, Y3);
    T1 = field_multiply(p.x, p.y);
    X3 = field_multiply(T0, T1);
    X3 = field_add(X3, X3);
    var T3 = field_multiply(X3, Y3);
    return UglyPoint(X3, Y3, p.t, Z3);
}

// fn normalize_x(x: Field, z: Field) -> Field {
//     var z_inverse = field_inverse(z);
//     var z_inv_squared = field_multiply(z_inverse, z_inverse);
//     return field_multiply(x, z_inv_squared);
// }