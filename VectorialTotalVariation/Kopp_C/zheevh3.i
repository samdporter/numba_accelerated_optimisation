/* File: kopp.i */
%module kopp

%{
#define SWIG_FILE_WITH_INIT
#include "dsytrd3.c"
#include "dsyev2.c"
#include "dsyevc3.c"
#include "dsyevd3.c"
#include "dsyevh3.c"
#include "dsyevj3.c"
#include "dsyevq3.c"
#include "dsyevv3.c"
#include "slvsec3.c"
#include "zheevc3.c"
#include "zheevd3.c"
#include "zheevh3.h"
#include "zheevj3.c"
#include "zheevq3.c"
#include "zheevv3.c"
#include "zhetrd3.c"
%}

%include numpy.i
%include "typemaps.i"
%include <complex.i>

%init %{
import_array();
%}

%numpy_typemaps(double complex, NPY_CDOUBLE, int)
%numpy_typemaps(double, NPY_DOUBLE, int)
%numpy_typemaps(int,    NPY_INT   , int)

%apply (double complex IN_ARRAY2[ANY][ANY]) {(double complex A[3][3])}
%apply (double complex ARGOUT_ARRAY2[ANY][ANY]) {(double complex Q[3][3])}
%apply (double ARGOUT_ARRAY1[ANY]) {(double w[3])}

%include "dsyev2.h"
%include "dsyevc3.h"
%include "dsyevd3.h"
%include "dsyevh3.h"
%include "dsyevj3.h"
%include "dsyevq3.h"
%include "dsyevv3.h"
%include "dsytrd3.h"
%include "slvsec3.h"
%include "zheevc3.h"
%include "zheevd3.h"
%include "zheevh3.h"
%include "zheevj3.h"
%include "zheevq3.h"
%include "zheevv3.h"
%include "zhetrd3.h"