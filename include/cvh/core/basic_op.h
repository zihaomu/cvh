//
// Created by mzh on 2024/3/27.
//

#ifndef CVH_CORE_BASIC_OP_H
#define CVH_CORE_BASIC_OP_H

#include "mat.h"

namespace cvh
{

/*
TODO: all binary op 都需要对齐到opencv。
*/

// BinaryOp
enum BinaryOp
{
    AND = 0,
    EQUAL,
    GREATER,
    GREATER_EQUAL,
    LESS,
    LESS_EQUAL,
    OR,
    POW,
    XOR,
    BITSHIFT,
    MOD,  // Integer Mod. Reminder's sign = Divisor's sign.
    MUL,
    SUB,
    ADD,
    DIV,
    MAX,
    MIN,
    ATAN2,
    HYPOT,
    NOT,  // Bitwise and-not: a & (~b), integral depth only.
    SUM,  // Alias of ADD.
    FMOD, // Floating-point mod. Remainder keeps dividend sign.
    MEAN, // Element-wise arithmetic mean: (a + b) / 2.
};

void binaryFunc(BinaryOp op, const Mat& a, const Mat& b, Mat& c);

// a + b = c
void add(const Mat& a, const Mat& b, Mat& c);
void add(const Mat& a, const Scalar& b, Mat& c);
void add(const Scalar& a, const Mat& b, Mat& c);

// a * alpha + b * beta = c
void addWeighted(const Mat& a, double alpha, const Mat& b, double beta, Mat& c);

// -a = c
void subtract(const Mat& a, Mat& c);

// a - b = c
void subtract(const Mat& a, const Mat& b, Mat& c);
void subtract(const Mat& a, const Scalar& b, Mat& c);
void subtract(const Scalar& a, const Mat& b, Mat& c);
void subtract(const Mat& a, double b, Mat& c);
void subtract(double a, const Mat& b, Mat& c);

// a * b = c
void multiply(const Mat& a, const Mat& b, Mat& c);
void multiply(const Mat& a, const Scalar& b, Mat& c);
void multiply(const Scalar& a, const Mat& b, Mat& c);

// a / b = c
void divide(const Mat& a, const Mat& b, Mat& c);
void divide(const Mat& a, const Scalar& b, Mat& c);
void divide(const Scalar& a, const Mat& b, Mat& c);

void compare(const Mat& a, const Mat& b, Mat& c, int op);
void compare(const Mat& a, const Scalar& b, Mat& c, int op);
void compare(const Scalar& a, const Mat& b, Mat& c, int op);

// Per-element absolute difference.
void absdiff(const Mat& a, const Mat& b, Mat& c);
void absdiff(const Mat& a, const Scalar& b, Mat& c);
void absdiff(const Scalar& a, const Mat& b, Mat& c);

// Per-element bit operations. Floating-point inputs are processed by raw bit pattern.
void bitwise_and(const Mat& a, const Mat& b, Mat& c, const Mat& mask = Mat());
void bitwise_and(const Mat& a, const Scalar& b, Mat& c, const Mat& mask = Mat());
void bitwise_and(const Scalar& a, const Mat& b, Mat& c, const Mat& mask = Mat());
void bitwise_or(const Mat& a, const Mat& b, Mat& c, const Mat& mask = Mat());
void bitwise_or(const Mat& a, const Scalar& b, Mat& c, const Mat& mask = Mat());
void bitwise_or(const Scalar& a, const Mat& b, Mat& c, const Mat& mask = Mat());
void bitwise_xor(const Mat& a, const Mat& b, Mat& c, const Mat& mask = Mat());
void bitwise_xor(const Mat& a, const Scalar& b, Mat& c, const Mat& mask = Mat());
void bitwise_xor(const Scalar& a, const Mat& b, Mat& c, const Mat& mask = Mat());
void bitwise_not(const Mat& src, Mat& dst, const Mat& mask = Mat());

// Inclusive per-pixel range test. All source channels must pass.
void inRange(const Mat& src, const Mat& lower, const Mat& upper, Mat& dst);
void inRange(const Mat& src, const Scalar& lower, const Scalar& upper, Mat& dst);

// Per-element minimum and maximum.
void min(const Mat& a, const Mat& b, Mat& c);
void min(const Mat& a, const Scalar& b, Mat& c);
void max(const Mat& a, const Mat& b, Mat& c);
void max(const Mat& a, const Scalar& b, Mat& c);

// Merge multiple source Mats into a multi-channel destination Mat.
void merge(const Mat* src, size_t nsrc, Mat& dst);
void merge(const std::vector<Mat>& src, Mat& dst);

// Split multi-channel source Mat into single-channel destination Mats.
void split(const Mat& src, Mat* dst);
void split(const Mat& src, std::vector<Mat>& dst);

// Transpose Mat last two dimension, if the Mat is one dimension, add the axis to the shape.
Mat transpose(const Mat& input);

// transpose Mat according to the input mat and the given new order.
Mat transposeND(const Mat& input, const std::vector<int> order);

// reshape Mat according to the given shape.
void reshape(const Mat& input, const std::vector<int>& shape, Mat& out);

#define MAT_AUG_OPERATOR1(op, cvop) \

#define MAT_AUG_OPERATOR(op, cvop) \
static inline Mat &operator op (Mat& a, const Mat& b) {cvop; return a;} \
static inline const Mat &operator op (const Mat& a, const Mat& b) {cvop; return a;}

MAT_AUG_OPERATOR(+=, add(a, b, (Mat &) a))
MAT_AUG_OPERATOR(-=, subtract(a, b, (Mat &) a))
MAT_AUG_OPERATOR(*=, multiply(a, b, (Mat &) a))
MAT_AUG_OPERATOR(/=, divide(a, b, (Mat &) a))

}

#include "detail/basic_op_impl.hpp"
#include "detail/mat_expr_impl.hpp"
#include "reduce.h"

#endif //CVH_BASIC_OP_H
