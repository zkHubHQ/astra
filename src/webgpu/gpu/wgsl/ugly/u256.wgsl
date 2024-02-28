// big endian
struct u256 {
  components: array<u32, 8>
}

// 115792089237316195423570985008687907853269984665640564039457584007913129639935
const U256_MAX: array<u32, 8> = 
  array<u32, 8>(4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295)
;

const U256_SEVENTEEN: array<u32, 8> =
  array<u32, 8>(0, 0, 0, 0, 0, 0, 0, 17)
;

const U256_ONE: array<u32, 8> =
  array<u32, 8>(0, 0, 0, 0, 0, 0, 0, 1)
;

const U256_TWO: array<u32, 8> =
  array<u32, 8>(0, 0, 0, 0, 0, 0, 0, 2)
;

const U256_THREE: array<u32, 8> =
  array<u32, 8>(0, 0, 0, 0, 0, 0, 0, 3)
;

const U256_FOUR: array<u32, 8> =
  array<u32, 8>(0, 0, 0, 0, 0, 0, 0, 4)
;

const U256_EIGHT: array<u32, 8> = 
  array<u32, 8>(0, 0, 0, 0, 0, 0, 0, 8)
;

const U256_ZERO: array<u32, 8> = 
  array<u32, 8>(0, 0, 0, 0, 0, 0, 0, 0)
;

// adds u32s together, returns a vector of the result and the carry (either 0 or 1)
fn add_components(a: u32, b: u32, carry_in: u32) -> vec2<u32> {
  var sum: vec2<u32>;
  let total = a + b + carry_in;
  // potential bitwise speed ups here
  sum[0] = total;
  sum[1] = 0u;
  // if the total is less than a, then we know there was a carry
  // need to subtract the carry_in for the edge case though, where a or b is 2^32 - 1 and carry_in is 1
  if (total < a || (total - carry_in) < a) {
    sum[1] = 1u;
  }
  return sum;
}

// subtracts u32s together, returns a vector of the result and the carry (either 0 or 1)
fn sub_components(a: u32, b: u32, carry_in: u32) -> vec2<u32> {
  var sub: vec2<u32>;
  let total = a - b - carry_in;
  sub[0] = total;
  sub[1] = 0u;
  // if the total is greater than a, then we know there was a carry from a less significant component.
  // need to add the carry_in for the edge case though, where a carry_in of 1 causes a component of a to underflow
  if (total > a || (total + carry_in) > a) {
    sub[1] = 1u;
  }
  return sub;
}

// no overflow checking for u256
fn u256_add(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
  var components = array<u32, 8>(0, 0, 0, 0, 0, 0, 0, 0);
  var carry: u32 = 0u;

  // for (var i = 7i; i >= 0i; i--) {
  //   let componentResult = add_components(a[i], b[i], carry);
  //   components[i] = componentResult[0];
  //   carry = componentResult[1];
  // }

  // unroll the loop to avoid naga issues in wgpu
  var componentResult = add_components(a[7], b[7], carry);
  components[7] = componentResult[0];
  carry = componentResult[1];

  componentResult = add_components(a[6], b[6], carry);
  components[6] = componentResult[0];
  carry = componentResult[1];

  componentResult = add_components(a[5], b[5], carry);
  components[5] = componentResult[0];
  carry = componentResult[1];

  componentResult = add_components(a[4], b[4], carry);
  components[4] = componentResult[0];
  carry = componentResult[1];

  componentResult = add_components(a[3], b[3], carry);
  components[3] = componentResult[0];
  carry = componentResult[1];

  componentResult = add_components(a[2], b[2], carry);
  components[2] = componentResult[0];
  carry = componentResult[1];

  componentResult = add_components(a[1], b[1], carry);
  components[1] = componentResult[0];
  carry = componentResult[1];

  componentResult = add_components(a[0], b[0], carry);
  components[0] = componentResult[0];
  carry = componentResult[1];

  return components;
}

fn u256_rs1(a: array<u32, 8>) -> array<u32, 8> {
  var right_shifted: array<u32, 8> = 
    array<u32, 8>(0, 0, 0, 0, 0, 0, 0, 0)
  ;
  var carry: u32 = 0u;
  for (var i = 0u; i < 8u; i++) {
    var componentResult = a[i] >> 1u;
    componentResult = componentResult | carry;
    right_shifted[i] = componentResult;
    carry = a[i] << 31u;
  }

  return right_shifted;
}


fn is_even(a: array<u32, 8>) -> bool {
  return (a[7u] & 1u) == 0u;
}

fn is_odd(a: array<u32, 8>) -> bool {
  return (a[7u] & 1u) == 1u;
}

fn is_odd_32_bits(a: u32) -> bool {
  return (a & 1u) == 1u;
}

fn u256_sub(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
  var components = array<u32, 8>(0, 0, 0, 0, 0, 0, 0, 0);
  var carry: u32 = 0u;
  // for (var i = 7i; i >= 0i; i--) {
  //   let componentResult = sub_components(a[i], b[i], carry);
  //   components[i] = componentResult[0];
  //   carry = componentResult[1];
  // }

  // Unroll the above loop to avoid the naga issues in wgpu
  var componentResult = sub_components(a[7], b[7], carry);
  components[7] = componentResult[0];
  carry = componentResult[1];

  componentResult = sub_components(a[6], b[6], carry);
  components[6] = componentResult[0];
  carry = componentResult[1];

  componentResult = sub_components(a[5], b[5], carry);
  components[5] = componentResult[0];
  carry = componentResult[1];

  componentResult = sub_components(a[4], b[4], carry);
  components[4] = componentResult[0];
  carry = componentResult[1];

  componentResult = sub_components(a[3], b[3], carry);
  components[3] = componentResult[0];
  carry = componentResult[1];

  componentResult = sub_components(a[2], b[2], carry);
  components[2] = componentResult[0];
  carry = componentResult[1];

  componentResult = sub_components(a[1], b[1], carry);
  components[1] = componentResult[0];
  carry = componentResult[1];

  componentResult = sub_components(a[0], b[0], carry);
  components[0] = componentResult[0];
  carry = componentResult[1];

  return components;
}

// underflow allowed u256 subtraction
fn u256_subw(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
  var sub: array<u32, 8>;
  if (gte(a, b)) {
    sub = u256_sub(a, b);
  } else {
    var b_minus_a: array<u32, 8> = u256_sub(b, a);
    var b_minus_a_minus_one: array<u32, 8> = u256_sub(b_minus_a, U256_ONE);
    sub = u256_sub(U256_MAX, b_minus_a_minus_one);
  }

  return sub;
}

fn equal(a: array<u32, 8>, b: array<u32, 8>) -> bool {
  for (var i = 0u; i < 8u; i++) {
    if (a[i] != b[i]) {
      return false;
    }
  }

  return true;
}

fn gt(a: array<u32, 8>, b: array<u32, 8>) -> bool {
  // for (var i = 0u; i < 8u; i++) {
  //   if (a[i] != b[i]) {
  //     return a[i] > b[i];
  //   }
  // }

  // Unroll the above loop to avoid the naga issues in wgpu

  if (a[0] != b[0]) {
    return a[0] > b[0];
  }

  if (a[1] != b[1]) {
    return a[1] > b[1];
  }

  if (a[2] != b[2]) {
    return a[2] > b[2];
  }

  if (a[3] != b[3]) {
    return a[3] > b[3];
  }

  if (a[4] != b[4]) {
    return a[4] > b[4];
  }

  if (a[5] != b[5]) {
    return a[5] > b[5];
  }

  if (a[6] != b[6]) {
    return a[6] > b[6];
  }

  if (a[7] != b[7]) {
    return a[7] > b[7];
  }

  // if a's components are never greater, than a is equal to b
  return false;
}

// returns whether a >= b
fn gte(a: array<u32, 8>, b: array<u32, 8>) -> bool {
  // for (var i = 0u; i < 8u; i++) {
  //   if (a[i] != b[i]) {
  //     return a[i] > b[i];
  //   }
  // }

  // Unroll the above loop to avoid the naga issues in wgpu

  if (a[0] != b[0]) {
    return a[0] > b[0];
  }

  if (a[1] != b[1]) {
    return a[1] > b[1];
  }

  if (a[2] != b[2]) {
    return a[2] > b[2];
  }

  if (a[3] != b[3]) {
    return a[3] > b[3];
  }

  if (a[4] != b[4]) {
    return a[4] > b[4];
  }

  if (a[5] != b[5]) {
    return a[5] > b[5];
  }

  if (a[6] != b[6]) {
    return a[6] > b[6];
  }

  if (a[7] != b[7]) {
    return a[7] > b[7];
  }

  // if none of the components are greater or smaller, a is equal to b
  return true;
}

fn component_double(a: u32, carry: u32) -> vec2<u32> {
  var double: vec2<u32>;
  let total = a << 1u;

  double[0] = total + carry;
  double[1] = 0u;

  if (total < a) {
    double[1] = 1u;
  }
  return double;
}

fn u256_double(a: array<u32, 8>) -> array<u32, 8> {
  var components = array<u32, 8>(0, 0, 0, 0, 0, 0, 0, 0);
  var carry: u32 = 0u;
  // for (var i = 7i; i >= 0i; i--) {
  //   let componentResult = component_double(a[i], carry);
  //   components[i] = componentResult[0];
  //   carry = componentResult[1];
  // }

  // unroll the loop to avoid naga issues in wgpu

  var componentResult = component_double(a[7], carry);
  components[7] = componentResult[0];
  carry = componentResult[1];

  componentResult = component_double(a[6], carry);
  components[6] = componentResult[0];
  carry = componentResult[1];

  componentResult = component_double(a[5], carry);
  components[5] = componentResult[0];
  carry = componentResult[1];

  componentResult = component_double(a[4], carry);
  components[4] = componentResult[0];
  carry = componentResult[1];

  componentResult = component_double(a[3], carry);
  components[3] = componentResult[0];
  carry = componentResult[1];

  componentResult = component_double(a[2], carry);
  components[2] = componentResult[0];
  carry = componentResult[1];

  componentResult = component_double(a[1], carry);
  components[1] = componentResult[0];
  carry = componentResult[1];

  componentResult = component_double(a[0], carry);
  components[0] = componentResult[0];
  carry = componentResult[1];

  return components;
}

fn component_right_shift(a: u32, shift: u32, carry: u32) -> vec2<u32> { 
  var shifted: vec2<u32>;
  shifted[0] = (a >> shift) + carry;
  shifted[1] = a << (32u - shift);

  return shifted;
}

fn u256_right_shift(a: array<u32, 8>, shift: u32) -> array<u32, 8> {
  // var components_to_drop = shift / 32u;
  var components_to_drop = 0;
  // if (components_to_drop >= 8u) {
  //   return U256_ZERO;
  // }

  var big_shift_components = array<u32, 8>(0, 0, 0, 0, 0, 0, 0, 0);

  // Shift out the components that need dropping
  // for (var i = components_to_drop; i < 8u; i++) {
  //   big_shift_components[i] = a[i-components_to_drop];
  // }

  // Unroll the above loop to avoid the naga issues in wgpu

  big_shift_components[0] = a[0];
  big_shift_components[1] = a[1];
  big_shift_components[2] = a[2];
  big_shift_components[3] = a[3];
  big_shift_components[4] = a[4];
  big_shift_components[5] = a[5];
  big_shift_components[6] = a[6];
  big_shift_components[7] = a[7];

  var big_shift: array<u32, 8>;
  big_shift = big_shift_components;

  var shift_within_component = shift % 32u;

  if (shift_within_component == 0u) {
    return big_shift;
  }

  var carry: u32 = 0u;
  big_shift_components = big_shift;
  // for (var i = components_to_drop; i < 8u; i++) {
  //   let componentResult = component_right_shift(big_shift_components[i], shift_within_component, carry);
  //   big_shift_components[i] = componentResult[0];
  //   carry = componentResult[1];
  // }

  // Unroll the above loop to avoid the naga issues in wgpu
  var componentResult = component_right_shift(big_shift_components[0], shift_within_component, carry);
  big_shift_components[0] = componentResult[0];
  carry = componentResult[1];

  componentResult = component_right_shift(big_shift_components[1], shift_within_component, carry);
  big_shift_components[1] = componentResult[0];
  carry = componentResult[1];

  componentResult = component_right_shift(big_shift_components[2], shift_within_component, carry);
  big_shift_components[2] = componentResult[0];
  carry = componentResult[1];

  componentResult = component_right_shift(big_shift_components[3], shift_within_component, carry);
  big_shift_components[3] = componentResult[0];
  carry = componentResult[1];

  componentResult = component_right_shift(big_shift_components[4], shift_within_component, carry);
  big_shift_components[4] = componentResult[0];
  carry = componentResult[1];

  componentResult = component_right_shift(big_shift_components[5], shift_within_component, carry);
  big_shift_components[5] = componentResult[0];
  carry = componentResult[1];

  componentResult = component_right_shift(big_shift_components[6], shift_within_component, carry);
  big_shift_components[6] = componentResult[0];
  carry = componentResult[1];

  componentResult = component_right_shift(big_shift_components[7], shift_within_component, carry);
  big_shift_components[7] = componentResult[0];
  carry = componentResult[1];

  big_shift = big_shift_components;

  return big_shift;
}
