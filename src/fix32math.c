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


#include "fix32math.h"

#define ASSERT(cond, msg)

#define FIX32_INVSQRT_NEWTON_ITERS    2

/**
 * Approximate the inverse square root using cubic interpolation refined with
 * Newton's method.  Well-conditioned and smooth with continuous first
 * derivative.  Accepts and returns unsigned 32-bit fixed point values with a
 * scaling factor of 2^scale, where scale must be even.  Undefined for val = 0.
 * Modifies scale to return a value with high precision.
 */
uint32_t fix32_invsqrt(uint32_t val, int *scale)
{
    ASSERT(((*scale) & 1) == 0, "scale is odd")
    //if ((scale & 1))
    //    val = (val + 1) >> 1;

    uint32_t val_copy = val; // save a copy of val for later

    // Let: val = a * 2^(2n) , with 1 <= a < 4
    // then: sqrt(val) = sqrt(a) * 2^n

    // Let's start by extracting a; get the index of the highest set bit:
    int msb = 0;
#ifdef __riscv
    asm("li     t0, 0xffff\n\t"
        "sltu   t0, t0, %1\n\t"
        "slli   %0, t0, 4\n\t"
        "srl    t1, %1, %0\n\t"

        "li     t0, 0xff\n\t"
        "sltu   t0, t0, t1\n\t"
        "slli   t0, t0, 3\n\t"
        "add    %0, %0, t0\n\t"
        "srl    t1, t1, t0\n\t"

        "li     t0, 0xf\n\t"
        "sltu   t0, t0, t1\n\t"
        "slli   t0, t0, 2\n\t"
        "add    %0, %0, t0\n\t"
        "srl    t1, t1, t0\n\t"

        "li     t0, 0x3\n\t"
        "sltu   t0, t0, t1\n\t"
        "slli   t0, t0, 1\n\t"
        "add    %0, %0, t0\n\t"

        : "=r"(msb) : "r"(val) : "t0", "t1");
#else
    if (val & 0xFFFF0000) {
        val &= 0xFFFF0000;
        msb += 16;
    }
    if (val & 0xFF00FF00) {
        val &= 0xFF00FF00;
        msb += 8;
    }
    if (val & 0xF0F0F0F0) {
        val &= 0xF0F0F0F0;
        msb += 4;
    }
    if (val & 0xCCCCCCCC)
        msb += 2;
#endif

    // extract 'a' by correctly shifting val; since 1 <= a < 4, it can be
    // stored with a scaling factor of 2^30 for maximum precision:
    uint32_t a = val_copy << (30 - msb);

    // 'n' can be calculated from 'scale' and the highest bit index 'msb':
    // n = (msb - scale) / 2 ; n is therefore in the range [-15,15]
    int n = (msb - *scale) >> 1; // works for negative n since values are even

    // Next, we approximate 1/sqrt(a); this is done by cubic interpolation in
    // order to get smooth transitions between interpolation intervals.
    // Since 1 <= a < 4 we interpolate in the interval [1,4].
    // The derivative of 1/sqrt(a) is: d/da 1/sqrt(a) = -1 / (2 a sqrt(a)),
    // therefore these are the boundary conditions:
    // 1/sqrt(1) = 1, 1/sqrt(4) = 0.5,
    // d/da 1/sqrt(a=1) = -0.5, d/da 1/sqrt(a=4) = -0.0625
    // which yields following cubic polynomial:
    // p(a) = -11/432 a^3 + 19/72 a^2 - 137/144 a + 185/108

    // Polynomial constant fractions with scaling factor of 2^30:
    const uint32_t frac_11_432  = 0x01A12F68,
                   frac_19_72   = 0x10E38E39,
                   frac_137_144 = 0x3CE38E39,
                   frac_185_108 = 0x6DA12F68;

    // Calculate a^2 and a^3; use scaling factor of 2^28 and 2^26 respectively,
    // to accomodate for larger ranges (i.e. 1 <= a^2 < 16 and 1 <= a^3 < 64 ):
    uint32_t a_squ = ((uint64_t)a * a     + (1u<<31)) >> 32,
             a_cub = ((uint64_t)a * a_squ + (1u<<31)) >> 32;

    // Do additions before subtractions and use a scaling factor of 2^28 for
    // intermediate results to avoid overflow of unsigned integers:
    uint32_t res = ((frac_185_108 + (1u<<1)) >> 2)
                + (uint32_t)(((uint64_t)frac_19_72 * a_squ + (1u<<29)) >> 30)
                - (uint32_t)(((uint64_t)frac_137_144 * a + (1u<<31)) >> 32)
                - (uint32_t)(((uint64_t)frac_11_432 * a_cub + (1u<<27)) >> 28);

    // 0.5 < res <= 1 ; scale res up to a scaling factor of 2^30 (we could use
    // 2^31, but it should be possible to cast the final res to a signed 32-bit
    // integer without issues, thus we use 2^30 to keep the sign bit clear):
    res <<= 2;

#ifdef FIX32_INVSQRT_NEWTON_ITERS
    // Now let us refine this with Newton's method:

    const uint32_t threehalfs = 3u<<30; // 1.5 with a scaling factor of 2^31
    int i;
    for (i = 0; i < FIX32_INVSQRT_NEWTON_ITERS; i++) {
        // 0.25 < res^2 <= 1 ; store res^2 with a scaling factor of 2^31:
        uint32_t res_squ = ((uint64_t)res * res + (1u<<28)) >> 29;

        // Since 1 <= a < 4 , 0.125 <= a * res^2 / 2 < 2 ; keep 2^31 scaling;
        // 'a' has a scaling factor of 2^30; also, we need to divide by 2:
        uint32_t half_a_res_squ = ((uint64_t)a * res_squ + (1u<<30)) >> 31;

        // For a > 2, res < 0.8 , thus res^2 < 0.75 , hence a * res^2 / 2 < 1.5
        // therefore 1.5 - a * res^2 / 2 is always positive:
        res = ((uint64_t)res * (threehalfs - half_a_res_squ) + (1<<30)) >> 31;
    }
#endif

    // Finally, 1/sqrt(val) = 1/sqrt(a) * 2^(-n)
    // The intermediate result has a scaling factor of 2^30; thus the scaling
    // factor of the final result is 2^(30 + n) ; modify scale accordingly:
    *scale = 30 + n;

    return res;
}


/**
 * Rough approximation of atan2, i.e. the arcus tangens of y/x
 */
int32_t fix32_atan2(int32_t y, int32_t x, int scale)
{
    int32_t abs_x = (x >= 0) ? x : -x,
            abs_y = (y >= 0) ? y : -y;

    int octant = (abs_x > abs_y) ? 0 : 1;
    if (x < 0)
        octant = 3 - octant;
    if (y < 0)
        octant = 7 - octant;

    // product of x and y, with a scaling factor of 2^(scale + scale - 32)
    int32_t x_y = fix32_mul(x, y, 32);

    // squares of x and y, also with a scaling factor of 2^(scale + scale - 32)
    int32_t sq_x = fix32_mul(x, x, 32),
            sq_y = fix32_mul(y, y, 32);

    int sq_scale = scale + scale - 32;

    int32_t _28125 = 0x48000000; // 0.28125 with a scaling factor of 2^32

    int32_t denum;
    switch (octant) {
        case 7:
        case 0:
        case 3:
        case 4:
            denum = sq_x + fix32_mul(sq_y, _28125, 32);
            break;

        default: // 1, 2, 5, 6
            denum = sq_y + fix32_mul(sq_x, _28125, 32);
    }

    int den_scale = sq_scale;
    int32_t inv_sqrt = fix32_invsqrt(denum, &den_scale); // den_scale altered

    // inverse has scaling factor of 2^(2*den_scale - 32)
    int32_t inv = fix32_mul(inv_sqrt, inv_sqrt, 32);

    int shift = sq_scale + (2 * den_scale - 32) - 28; // target scale: 2^28

    int32_t pi_half = 0x1921FB54, // pi/2 with a scaling factor of 2^28
            pi      = 0x3243F6A9; // pi with a scaling factor of 2^28

    switch (octant) {
        case 7:
        case 0:
            return fix32_mul(x_y, inv, shift);

        case 1:
        case 2:
            return pi_half - fix32_mul(x_y, inv, shift);

        case 3:
            return pi + fix32_mul(x_y, inv, shift);

        case 4:
            return -pi + fix32_mul(x_y, inv, shift);

        case 5:
        case 6:
            return -pi_half - fix32_mul(x_y, inv, shift);
    }

    // not reached
    return 0;
}
