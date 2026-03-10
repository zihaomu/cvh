//
// Created by mzh on 2024/3/27.
//

#ifndef CVH_MAT_INL_H
#define CVH_MAT_INL_H

namespace cvh
{

inline
MatSize::MatSize(int* _p) : p0(_p)
{
    if (_p)
        p = _p + 1;
    else
        p = nullptr;
}

inline
int MatSize::dims() const
{
    return p0[0];
}

inline
const int& MatSize::operator[](int i) const
{
    return p[i];
}

inline
int& MatSize::operator[](int i)
{
    return p[i];
}

inline
bool MatSize::operator!=(const MatSize& sz) const
{
    return !(*this == sz);
}

inline
bool MatSize::operator==(const MatSize& sz) const
{
    int d = dims();
    int dsz = sz.dims();

    if (d != dsz)
        return false;

    for (int i = 0; i < d; i++)
    {
        if (p[i] != sz[i])
            return false;
    }

    return true;
}

inline
size_t total(const Mat& m)
{
    return m.total();
}

inline
size_t total(const Mat& m, int startDim, int endDim)
{
    if (endDim == -1)
    {
        endDim = m.dims;
    }
    return m.total(startDim, endDim);
}

inline
size_t total(const MatShape shape)
{
    return total(shape, 0, static_cast<int>(shape.size()));
}

inline
size_t total(const MatShape shape, int startDim, int endDim)
{
    if (endDim == -1)
    {
        endDim = static_cast<int>(shape.size());
    }

    assert(startDim >= 0 && startDim <= endDim);
    size_t p = 1;
    int dims = static_cast<int>(shape.size());

    int endDim_ = endDim <= dims ? endDim : dims;
    for (int i = startDim; i < endDim_; i++)
    {
        p *= static_cast<size_t>(shape[i]);
    }
    return p;
}

inline
MatExpr::MatExpr()
        : op(0), flags(0), a(Mat()), b(Mat()), c(Mat()), alpha(0), beta(0)
{}

inline
MatExpr::MatExpr(const MatOp* _op, int _flags, const Mat& _a, const Mat& _b,
                 const Mat& _c, double _alpha, double _beta)
        : op(_op), flags(_flags), a(_a), b(_b), c(_c), alpha(_alpha), beta(_beta)
{}

inline
MatExpr::operator Mat() const
{
    Mat m;
    op->assign(*this, m);
    return m;
}

inline
Mat& Mat::operator = (const MatExpr& e)
{
    e.op->assign(e, *this);
    return *this;
}

static inline
Mat& operator += (Mat& a, const MatExpr& b)
{
    b.op->augAssginAdd(b, a);
    return a;
}

static inline
const Mat& operator += (const Mat& a, const MatExpr& b)
{
    b.op->augAssginAdd(b, (Mat&)a);
    return a;
}

static inline
Mat& operator -= (Mat& a, const MatExpr& b)
{
    b.op->augAssginSubtract(b, a);
    return a;
}

static inline
const Mat& operator -= (const Mat& a, const MatExpr& b)
{
    b.op->augAssginSubtract(b, (Mat&)a);
    return a;
}

static inline
Mat& operator *= (Mat& a, const MatExpr& b)
{
    b.op->augAssginMultiply(b, a);
    return a;
}

static inline
const Mat& operator *= (const Mat& a, const MatExpr& b)
{
    b.op->augAssginMultiply(b, (Mat&)a);
    return a;
}

static inline
Mat& operator /= (Mat& a, const MatExpr& b)
{
    b.op->augAssginDivide(b, a);
    return a;
}

static inline
const Mat& operator /= (const Mat& a, const MatExpr& b)
{
    b.op->augAssginDivide(b, (Mat&)a);
    return a;
}

}

#endif //CVH_MAT_INL_H
