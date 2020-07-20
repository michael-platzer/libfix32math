/*
 * Copyright (c) 2020 Michael Platzer (TU Wien)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 * SPDX-License-Identifier: MIT
 */


/**
 * Math library for 32-bit fixed-point computation
 *
 * Hosted at: https://github.com/michael-platzer/libfix32math
 */


/**
 * This header must be included only once per translation unit (ideally from
 * the source file) to avoid conflicting definitions of the macros controlling
 * the rounding and overflow actions of fixed point arithmetic.  Avoid
 * including it from header files.
 */
#ifdef FIX32MATH_H
#error "ERROR: `fix32math.h' must not be included more than once"
#endif
#define FIX32MATH_H

#include <stdint.h>


/**
 * Scale down a signed 32-bit or 64-bit fixed point number (equivalent to a
 * division by 2^n) with rounding to nearest in following flavours (see
 * https://en.wikipedia.org/wiki/Rounding#Rounding_to_the_nearest_integer ):
 *
 *  - RHU: Round Half Up (i.e. 0.5 becomes 1 and -0.5 becomes 0),
 *    by adding 2^(n-1) to val before shifting.
 *
 *  - RHD: Round Half Down (i.e. 0.5 becomes 0 and -0.5 becomes -1),
 *    by adding 2^(n-1) - 1 to val before shifting.
 *
 *  - RHAZ: Round Half Away from Zero (i.e. 0.5 becomes 1 and -0.5 becomes -1),
 *    by adding 2^(n-1) if val is positive and 2^(n-1) - 1 if val is negative.
 *
 *  - RHTZ: Round Half Towards Zero (i.e. 0.5 becomes 0 and -0.5 becomes 0),
 *    by adding 2^(n-1) - 1 if val is positive and 2^(n-1) if val is negative.
 */
// scale function template; allows to specify integer data type, function name
// extension and what else to add to val besides 2^(n-1) before shifting:
#define FIX32_MATH_SCALE_FUNCTION(DTYPE, NAME_SUFFIX, ADD_TO_VAL_BESIDE_HALF) \
static DTYPE fix32_scale_##NAME_SUFFIX (DTYPE val, int n) {                   \
    return (val + ((1LL << (n - 1)) ADD_TO_VAL_BESIDE_HALF)) >> n;            \
}
FIX32_MATH_SCALE_FUNCTION(int32_t, rhu_32, )                    // 32-bit RHU
FIX32_MATH_SCALE_FUNCTION(int32_t, rhd_32, - 1)                 // 32-bit RHD
FIX32_MATH_SCALE_FUNCTION(int32_t, rhaz_32, + (val >> 31))      // 32-bit RHAZ
FIX32_MATH_SCALE_FUNCTION(int32_t, rhtz_32, + (~(val >> 31)))   // 32-bit RHTZ
FIX32_MATH_SCALE_FUNCTION(int64_t, rhu_64, )                    // 64-bit RHU
FIX32_MATH_SCALE_FUNCTION(int64_t, rhd_64, - 1)                 // 64-bit RHD
FIX32_MATH_SCALE_FUNCTION(int64_t, rhaz_64, + (val >> 63))      // 64-bit RHAZ
FIX32_MATH_SCALE_FUNCTION(int64_t, rhtz_64, + (~(val >> 63)))   // 64-bit RHTZ


/**
 * Multiply two fixed point numbers with scaling factor 2^n.
 *
 * The two 32-bit operands 'a' and 'b' are multiplied and the 64-bit result is
 * right-shifted (with sign extension) by 'n' bits (equivalent to a division by
 * 2^n).  The final result is rounded to the nearest value with half rounded
 * away from zero by default.  The macro FIX32_MATH_MUL_ROUND_FUNC can be used
 * to choose a different rounding function from the 'fix32_scale_*_64()' group.
 *
 * Arithmetic overflow is silently ignored by default, with higher bits being
 * lost if they do not fit the 32-bit integer type.  However, overflow checking
 * can be enabled and an action to be executed in case of overflow can be
 * specified by defining a function-like macro FIX32_MATH_MUL_OVERFLOW_ACTION,
 * which accepts as single argument the 64-bit variant of the multiplication of
 * 'a' and 'b'.
 */
static int32_t fix32_mul(int32_t a, int32_t b, int n)
{
    // use RHAZ rounding function by default
#ifndef FIX32_MATH_MUL_ROUND_FUNC
#define FIX32_MATH_MUL_ROUND_FUNC   fix32_scale_rhaz_64
#endif

    // multiply a and b and round according to the desired scheme:
    int64_t prod = FIX32_MATH_MUL_ROUND_FUNC((int64_t)a * b, n);

    // check for overflow if an overflow action was specified; skip otherwise:
#ifdef FIX32_MATH_MUL_OVERFLOW_ACTION
    // overflow occurs if any of the upper 33 bits are not equal (either 0 for
    // a positive number or 1 for a negative number); we use (-1LL << 31) as a
    // mask for those bits:
    if (((prod & (-1LL << 31)) != ((prod >> 32) & (-1LL << 31)))) {
        FIX32_MATH_MUL_OVERFLOW_ACTION(prod);
    }
#endif
    return prod;
}


/**
 * Approximate the inverse square root of a 32-bit fixed point value with a
 * scaling factor of 2^scale, where scale is even.  Undefined for val = 0.
 *
 * The approximation is calculated using cubic interpolation and improved with
 * one or two iterations of Newton's method (see FIX32_INVSQRT_NEWTON_ITERS).
 * The relative error is less than 1 % with one iteration and less than 0.01 %
 * with two iterations.  The result is well-conditioned and smooth with
 * continuous first derivative.
 *
 * @param val   32-bit fixed point input value with scaling factor 2^scale
 * @param scale scaling factor power; input and output parameter
 * @return      32-bit fixed point inverse square root of val with a scaling
 *              factor of 2^scale, where scale has been modified in order to
 *              retain high precision; the result can safely be cast to signed.
 */
uint32_t fix32_invsqrt(uint32_t val, int *scale);


/**
 * Rough approximation of atan2, i.e. the arcus tangens of y/x .
 *
 * @param x, y  32-bit fixed point input coordinates
 * @param scale scaling factor power of 2 of x and y
 * @return      32-bit fixed point arcus tangens of y/x with a scaling factor
 *              of 2^28
 */
int32_t fix32_atan2(int32_t x, int32_t y, int scale);
