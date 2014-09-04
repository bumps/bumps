/* This program is public domain. */

// MSVC 2008 doesn't define erf()
#if defined(_MSC_VER) && _MSC_VER<=1600
extern "C" double erf(double);
#endif /* NEED_ERF */

#include <Python.h>
#include <math.h>
#include <stdio.h>
#include <iostream>

#include "methods.h"

extern "C" void
convolve(size_t Nin, const double xin[], const double yin[],
         size_t N, const double x[], const double dx[], double y[]);
extern "C" void
convolve_sampled(size_t Nin, const double xin[], const double yin[],
         size_t Np, const double xp[], const double yp[],
         size_t N, const double x[], const double dx[], double y[]);

#if defined(PY_VERSION_HEX) &&  (PY_VERSION_HEX < 0x02050000)
typedef int Py_ssize_t;
#endif


#undef BROKEN_EXCEPTIONS


PyObject* Pconvolve(PyObject *obj, PyObject *args)
{
  PyObject *xi_obj,*yi_obj,*x_obj,*dx_obj,*y_obj;
  const double *xi, *yi, *x, *dx;
  double *y;
  Py_ssize_t nxi, nyi, nx, ndx, ny;

  if (!PyArg_ParseTuple(args, "OOOOO:convolve",
			&xi_obj,&yi_obj,&x_obj,&dx_obj,&y_obj)) return NULL;
  INVECTOR(xi_obj,xi,nxi);
  INVECTOR(yi_obj,yi,nyi);
  INVECTOR(x_obj,x,nx);
  INVECTOR(dx_obj,dx,ndx);
  OUTVECTOR(y_obj,y,ny);
  if (nxi != nyi) {
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "convolve: xi and yi have different lengths");
#endif
    return NULL;
  }
  if (nx != ndx || nx != ny) {
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "convolve: x, dx and y have different lengths");
#endif
    return NULL;
  }
  convolve(nxi,xi,yi,nx,x,dx,y);
  return Py_BuildValue("");
}

PyObject* Pconvolve_sampled(PyObject *obj, PyObject *args)
{
  PyObject *xi_obj,*yi_obj,*xp_obj,*yp_obj,*x_obj,*dx_obj,*y_obj;
  const double *xi, *yi, *xp, *yp, *x, *dx;
  double *y;
  Py_ssize_t nxi, nyi, nxp, nyp, nx, ndx, ny;

  if (!PyArg_ParseTuple(args, "OOOOOOO:convolve_sampled",
	   &xi_obj,&yi_obj,&xp_obj,&yp_obj,&x_obj,&dx_obj,&y_obj)) return NULL;
  INVECTOR(xi_obj,xi,nxi);
  INVECTOR(yi_obj,yi,nyi);
  INVECTOR(xp_obj,xp,nxp);
  INVECTOR(yp_obj,yp,nyp);
  INVECTOR(x_obj,x,nx);
  INVECTOR(dx_obj,dx,ndx);
  OUTVECTOR(y_obj,y,ny);
  if (nxi != nyi) {
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "convolve_sampled: xi and yi have different lengths");
#endif
    return NULL;
  }
  if (nxp != nyp) {
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "convolve_sampled: xp and yp have different lengths");
#endif
    return NULL;
  }
  if (nx != ndx || nx != ny) {
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "convolve_sampled: x, dx and y have different lengths");
#endif
    return NULL;
  }
  convolve_sampled(nxi,xi,yi,nxp,xp,yp,nx,x,dx,y);
  return Py_BuildValue("");
}


PyObject* Perf(PyObject*obj,PyObject*args)
{
  PyObject *data_obj, *result_obj;
  const double *data;
  double *result;
  Py_ssize_t ndata, nresult;

  if (!PyArg_ParseTuple(args, "OO:erf",
			&data_obj, &result_obj))
    return NULL;
  INVECTOR(data_obj,data, ndata);
  OUTVECTOR(result_obj, result, nresult);
  if (ndata != nresult) {
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "len(data) != nresult");
#endif
    return NULL;
  }
  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for(int i=0; i < ndata; i++) result[i] = erf(data[i]);
  return Py_BuildValue("");
}


