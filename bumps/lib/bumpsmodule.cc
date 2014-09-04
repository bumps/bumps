/* This program is public domain. */
// MSVC 2008 doesn't define cstdint
#if defined(_MSC_VER) && _MSC_VER<=1500
 #define MISSING_STDINT
#endif

#include <Python.h>

/* Visual Studio 2008 is missing cstdint */
#ifndef MISSING_STDINT
  #include <stdint.h>
#else
  typedef signed char        int8_t;
  typedef short              int16_t;
  typedef int                int32_t;
  typedef long long          int64_t;
  typedef unsigned char      uint8_t;
  typedef unsigned short     uint16_t;
  typedef unsigned int       uint32_t;
  typedef unsigned long long uint64_t;
#endif /* !MISSING_STDINT */

#include "methods.h"
#include "rebin.h"
#include "rebin2D.h"

//PyObject* pyvector(int n, double v[])
//{
//   int dims[1];
//   dims[0] = n;
//   return PyArray_FromDimsAndData(1,dims,PyArray_DOUBLE,(char *)v);
//}


template <typename T>
PyObject* Prebin(PyObject *obj, PyObject *args)
{
  PyObject *in_obj,*Iin_obj,*out_obj,*Iout_obj;
  Py_ssize_t nin,nIin, nout, nIout;
  double *in, *out;
  T *Iin, *Iout;

  if (!PyArg_ParseTuple(args, "OOOO:rebin",
                        &in_obj,&Iin_obj,&out_obj,&Iout_obj)) return NULL;
  INVECTOR(in_obj,in,nin);
  INVECTOR(Iin_obj,Iin,nIin);
  INVECTOR(out_obj,out,nout);
  OUTVECTOR(Iout_obj,Iout,nIout);
  if (nin-1 != nIin || nout-1 != nIout) {
    PyErr_SetString(PyExc_ValueError,
        "_reduction.rebin: must have one more bin edges than bins");
    return NULL;
  }
  rebin_counts<T>(nin-1,in,Iin,nout-1,out,Iout);
  return Py_BuildValue("");
}

template <typename T>
PyObject* Prebin2d(PyObject *obj, PyObject *args)
{
  PyObject *xin_obj, *yin_obj, *Iin_obj;
  PyObject *xout_obj, *yout_obj, *Iout_obj;
  Py_ssize_t nxin, nyin, nIin;
  Py_ssize_t nxout, nyout, nIout;
  double *xin,*yin,*xout,*yout;
  T *Iin, *Iout;

  if (!PyArg_ParseTuple(args, "OOOOOO:rebin",
                        &xin_obj, &yin_obj, &Iin_obj,
                        &xout_obj, &yout_obj, &Iout_obj))
        return NULL;

  INVECTOR(xin_obj,xin,nxin);
  INVECTOR(yin_obj,yin,nyin);
  INVECTOR(Iin_obj,Iin,nIin);
  INVECTOR(xout_obj,xout,nxout);
  INVECTOR(yout_obj,yout,nyout);
  OUTVECTOR(Iout_obj,Iout,nIout);
  if ((nxin-1)*(nyin-1) != nIin || (nxout-1)*(nyout-1) != nIout) {
    /* printf("%ld %ld %ld %ld %ld %ld\n",nxin,nyin,nIin,nxout,nyout,nIout); */
    PyErr_SetString(PyExc_ValueError,
        "_reduction.rebin2d: must have one more bin edges than bins");
    return NULL;
  }
  rebin_counts_2D<T>(nxin-1,xin,nyin-1,yin,Iin,
      nxout-1,xout,nyout-1,yout,Iout);
  return Py_BuildValue("");
}

static PyMethodDef methods[] = {

	{"convolve",
	 Pconvolve,
	 METH_VARARGS,
	 "convolve(xi,yi,x,dx,y): compute convolution of width dx[k] at points x[k],\nreturned in y[k]"},

	{"convolve_sampled",
	 Pconvolve_sampled,
	 METH_VARARGS,
	 "convolve_sampled(xi,yi,xp,yp,x,dx,y): compute convolution with sampled\ndistribution of width dx[k] at points x[k], returned in y[k]"},

	{"erf",
	 Perf,
	 METH_VARARGS,
	 "erf(data, result): get the erf of a set of data points"},


	{"rebin_uint8",
	 &Prebin<uint8_t>,
	 METH_VARARGS,
	 "rebin_uint8(xi,Ii,xo,Io): rebin from bin edges xi to bin edges xo"
	},
	{"rebin2d_uint8",
	 &Prebin2d<uint8_t>,
	 METH_VARARGS,
	 "rebin2d_uint8(xi,yi,Ii,xo,yo,Io): 2-D rebin from (xi,yi) to (xo,yo)"
	},
	{"rebin_uint16",
	 &Prebin<uint16_t>,
	 METH_VARARGS,
	 "rebin_uint16(xi,Ii,xo,Io): rebin from bin edges xi to bin edges xo"
	},
	{"rebin2d_uint16",
	 &Prebin2d<uint16_t>,
	 METH_VARARGS,
	 "rebin2d_uint16(xi,yi,Ii,xo,yo,Io): 2-D rebin from (xi,yi) to (xo,yo)"
	},
	{"rebin_uint32",
	 &Prebin<uint32_t>,
	 METH_VARARGS,
	 "rebin_uint32(xi,Ii,xo,Io): rebin from bin edges xi to bin edges xo"
	},
	{"rebin2d_uint32",
	 &Prebin2d<uint32_t>,
	 METH_VARARGS,
	 "rebin2d_uint32(xi,yi,Ii,xo,yo,Io): 2-D rebin from (xi,yi) to (xo,yo)"
	},
	{"rebin_uint64",
	 &Prebin<uint64_t>,
	 METH_VARARGS,
	 "rebin_uint64(xi,Ii,xo,Io): rebin from bin edges xi to bin edges xo"
	},
	{"rebin2d_uint64",
	 &Prebin2d<uint64_t>,
	 METH_VARARGS,
	 "rebin2d_uint64(xi,yi,Ii,xo,yo,Io): 2-D rebin from (xi,yi) to (xo,yo)"
	},
	{"rebin_float32",
	 &Prebin<float>,
	 METH_VARARGS,
	 "rebin_float32(xi,Ii,xo,Io): rebin from bin edges xi to bin edges xo"
	},
	{"rebin2d_float32",
	 &Prebin2d<float>,
	 METH_VARARGS,
	 "rebin2d_float32(xi,yi,Ii,xo,yo,Io): 2-D rebin from (xi,yi) to (xo,yo)"
	},
	{"rebin_float64",
	 &Prebin<double>,
	 METH_VARARGS,
	 "rebin_float64(xi,Ii,xo,Io): rebin from bin edges xi to bin edges xo"
	},
	{"rebin2d_float64",
	 &Prebin2d<double>,
	 METH_VARARGS,
	 "rebin2d_float64(xi,yi,Ii,xo,yo,Io): 2-D rebin from (xi,yi) to (xo,yo)"
	},

	{0}
} ;


#define MODULE_DOC "C library for convolving models and rebinning data."
#define MODULE_NAME "_reduction"
#define MODULE_INIT2 init_reduction
#define MODULE_INIT3 PyInit__reduction
#define MODULE_METHODS methods

/* ==== boilerplate python 2/3 interface bootstrap ==== */

#if PY_MAJOR_VERSION >= 3

  PyMODINIT_FUNC MODULE_INIT3(void)
  {
    static struct PyModuleDef moduledef = {
      PyModuleDef_HEAD_INIT,
      MODULE_NAME,         /* m_name */
      MODULE_DOC,          /* m_doc */
      -1,                  /* m_size */
      MODULE_METHODS,      /* m_methods */
      NULL,                /* m_reload */
      NULL,                /* m_traverse */
      NULL,                /* m_clear */
      NULL,                /* m_free */
    };
    return PyModule_Create(&moduledef);
  }

#else /* PY_MAJOR_VERSION >= 3 */

  PyMODINIT_FUNC MODULE_INIT2(void)
  {
    Py_InitModule4(MODULE_NAME,
		 MODULE_METHODS,
		 MODULE_DOC,
		 0,
		 PYTHON_API_VERSION
		 );
  }

#endif /* PY_MAJOR_VERSION >= 3 */
