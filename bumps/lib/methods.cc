/* This program is public domain. */

#include "reflcalc.h"
#include <math.h>
#include <stdio.h>
#include <iostream>
#include "methods.h"


#if defined(PY_VERSION_HEX) &&  (PY_VERSION_HEX < 0x02050000)
typedef int Py_ssize_t;
#endif


#undef BROKEN_EXCEPTIONS


PyObject* Pconvolve(PyObject *obj, PyObject *args)
{
  PyObject *Qi_obj,*Ri_obj,*Q_obj,*dQ_obj,*R_obj;
  const double *Qi, *Ri, *Q, *dQ;
  double *R;
  Py_ssize_t nQi, nRi, nQ, ndQ, nR;

  if (!PyArg_ParseTuple(args, "OOOOO:resolution",
			&Qi_obj,&Ri_obj,&Q_obj,&dQ_obj,&R_obj)) return NULL;
  INVECTOR(Qi_obj,Qi,nQi);
  INVECTOR(Ri_obj,Ri,nRi);
  INVECTOR(Q_obj,Q,nQ);
  INVECTOR(dQ_obj,dQ,ndQ);
  OUTVECTOR(R_obj,R,nR);
  if (nQi != nRi) {
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "_librefl.convolve: Qi and Ri have different lengths");
#endif
    return NULL;
  }
  if (nQ != ndQ || nQ != nR) {
#ifndef BROKEN_EXCEPTIONS
    PyErr_SetString(PyExc_ValueError, "_librefl.convolve: Q, dQ and R have different lengths");
#endif
    return NULL;
  }
  resolution(nQi,Qi,Ri,nQ,Q,dQ,R);
  return Py_BuildValue("");
}


PyObject* Perf(PyObject*obj,PyObject*args)
{
  PyObject *data_obj, *result_obj;
  const double *data;
  double *result;
  int i;
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
  for(i=0; i < ndata; i++)
    result[i] = erf(data[i]);
  return Py_BuildValue("");
}
