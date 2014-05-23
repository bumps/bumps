/* This program is public domain. */

// MSVC 2008 doesn't define erf()
#if defined(_MSC_VER) && _MSC_VER<=1600
  #define const
  #define __LITTLE_ENDIAN
  #include "erf.c"
#endif

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <math.h>

#ifdef SGI
#include <ieeefp.h>
#endif

/* What to do at the endpoints --- USE_TRUNCATED_NORMALIZATION will
 * avoid the assumption that the data is zero where it hasn't been
 * measured.
 */
#define USE_TRUNCATED_NORMALIZATION

/* Computed using extended precision with Octave's symbolic toolbox. */
#define PI4          12.56637061435917295385
#define PI_180        0.01745329251994329576
#define LN256         5.54517744447956247533
#define SQRT2         1.41421356237309504880
#define SQRT2PI       2.50662827463100050241

/* Choose the resolution limit based on relative contribution to resolution
 * rather than absolute contribution.  So if we ignore everything below,
 * e.g. 0.1% of the peak, that limit occurs when G(x)/G(0) = 0.001 for
 * gaussian G of width sima, or when x = sqrt(-2 sigma^2 log(0.001)). */
#define LOG_RESLIMIT -6.90775527898213703123

/** \file
The resolution function returns the convolution of the curve with a
x-dependent gaussian.

We provide the following function:
   resolution(Nin, xin, yin, N, x, dx, y)  returns convolution
   resolution_padding(step,dx)             returns \#points (see below)

where
   Nin is the number of theory points
   xin,yin are the computed theory points
   N is the number of x points to calculate
   x are the locations of the measured data points
   dx are the width (sigma) of the convolution at each measured point
   y is the returned convolution.

Note that FWHM = sqrt(8 ln 2) dx, so scale dx appropriately.

The contribution of x to a resolution of width dxo at point xo is:

   p(x) = 1/sqrt(2 pi dxo^2) exp ( (x-xo)^2/(2 dxo^2) )

We are approximating the convolution at xo using a numerical
approximation to the integral over the measured points.  For
efficiency, the integral is limited to p(x_i)/p(0)>=0.001.

Note that the function we are convoluting may be falling off rapidly.
That means the correct convolution should uniformly sample across
the entire width of the Gaussian.  This is not possible at the
end points unless you calculate the curve beyond what is
strictly needed for the data. The function resolution_pad(dx,step)
returns the number of additional steps of size step required to
go beyond this limit for the given width dx.  This occurs when:

    (n*step)^2 < -2 dx^2 * ln 0.001

The choice of sampling density is particularly important near the
critical points.  This is where the resolution calculation has the
largest effect on the curve. In one particular model, calculating x
at every 0.001 rather than every 0.02 changed a value
above the critical point by 15%.  This is likely to be a problem for
any system with rapidly changing behaviour.  The solution is to
compute the theory function over a finer mesh where the derivative
is changing rapidly.

For systems in which the theory function oscillates rapidly around the
measured points there are problems when the period of the oscillation is on
the order of the width of the resolution function. In these systems, the
theory function should be oversampled around the measured points x.

FIXME is it better to use random sampling or strictly
regular spacing when you are undersampling?

===============================================================
*/

#undef USE_TRAPEZOID_RULE
#ifdef USE_TRAPEZOID_RULE
#warning This code does strange things with small sigma and large spacing
/* FIXME trapezoid performs very badly for large spacing unless
   we normalize to the unit width.  For very small sigma, the gaussian
   is a spike, but we are approximating it by a triangle so it is
   not surprising it works so poorly.  A slightly better solution is
   to use the inner limits rather than the outer limits, but this will
   still break down if the x spacing is approximately equal to limit.
   Best is to not use trapezoid.
*/

/* Trapezoid rule for numerical integration of convolution */
double
convolve_point(const double xin[], const double yin[], size_t k, size_t n,
        double xo, double limit, double sigma)
{
  const double two_sigma_sq = 2. * sigma * sigma;
  double z, Glo, yGlo, y, norm;

  z = xo - xin[k];
  Glo = exp(-z*z/two_sigma_sq);
  yGlo = yin[k]*Glo;
  norm = y = 0.;
  while (++k < n) {
    /* Compute the next endpoint */
    const double zhi = xo - xin[k];
    const double Ghi = exp(-zhi*zhi/two_sigma_sq);
    const double yGhi = yin[k] * Ghi;
    const double halfstep = 0.5*(xin[k] - xin[k-1]);

    /* Add the trapezoidal area. */
    norm += halfstep * (Ghi + Glo);
    y += halfstep * (yGhi + yGlo);

    /* Save the endpoint for next trapezoid. */
    Glo = Ghi;
    yGlo = yGhi;

    /* Check if we've calculated far enough */
    if (xin[k] >= xo+limit) break;
  }

  /* Scale to area of the linear spline distribution we actually used. */
  return y / norm;

  /* Scale to gaussian of unit area */
  /* Fails badly for small sigma or large x steps---do not use. */
  /* return y / sigma*SQRT2PI; */
}

#else /* !USE_TRAPEZOID_RULE */

/* Analytic convolution of gaussian with linear spline */
/* More expensive but more reliable */
double
convolve_point(const double xin[], const double yin[], size_t k, size_t n,
               double xo, double limit, double sigma)
{
  const double two_sigma_sq = 2. * sigma * sigma;
  double z, Glo, erflo, erfmin, y;

  z = xo - xin[k];
  Glo = exp(-z*z/two_sigma_sq);
  erfmin = erflo = erf(-z/(SQRT2*sigma));
  y = 0.;
  /* printf("%5.3f: (%5.3f,%11.5g)",xo,xin[k],yin[k]); */
  while ((++k < n) && (xin[k] != xin[k-1])) {
  	/* No additional contribution from duplicate points. */
  	//if (xin[k] == xin[k-1]) continue;

    /* Compute the next endpoint */
    const double zhi = xo - xin[k];
    const double Ghi = exp(-zhi*zhi/two_sigma_sq);
    const double erfhi = erf(-zhi/(SQRT2*sigma));
    const double m = (yin[k]-yin[k-1])/(xin[k]-xin[k-1]);
    const double b = yin[k] - m * xin[k];

    /* Add the integrals. */
    y += 0.5*(m*xo+b)*(erfhi-erflo) - sigma/SQRT2PI*m*(Ghi-Glo);

    /* Debug computation failures. */
    /*
    if isnan(y) {
    	printf("NaN from %d: zhi=%g, Ghi=%g, erfhi=%g, m=%g, b=%g\n",
    	       k,zhi,Ghi,erfhi,m,b);
    }
    */

    /* Save the endpoint for next trapezoid. */
    Glo = Ghi;
    erflo = erfhi;

    /* Check if we've calculated far enough */
    if (xin[k] >= xo+limit) break;
  }
  /* printf(" (%5.3f,%11.5g)",xin[k<n?k:n-1],yin[k<n?k:n-1]); */

#ifdef USE_TRUNCATED_NORMALIZATION
  /* Normalize by the area of the truncated gaussian */
  /* At this point erflo = erfmax */
  /* printf ("---> %11.5g\n",2*y/(erflo-erfmin)); */
  return 2 * y / (erflo - erfmin);
#else
  /* Return unnormalized (we used a gaussian of unit area) */
  /* printf ("---> %11.5g\n",y); */
  return y;
#endif
}

#endif /* !USE_TRAPEZOID_RULE */

void
convolve(size_t Nin, const double xin[], const double yin[],
         size_t N, const double x[], const double dx[], double y[])
{
  size_t in,out;

  /* FIXME fails if xin are not sorted; slow if x not sorted */
  assert(Nin>1);

  /* Scan through all x values to be calculated */
  /* Re: omp, each thread is going through the entire input array,
   * independently, computing the resolution from the neighbourhood
   * around its individual output points.  The firstprivate(in)
   * clause sets each thread to keep its own copy of in, initialized
   * at in's initial value of zero.  The "schedule(static,1)" clause
   * puts neighbouring points in separate threads, which is a benefit
   * since there will be less backtracking if resolution width increases
   * from point to point.  Because the schedule is static, this does not
   * significantly increase the parallelization overhead.  Because the
   * threads are operating on interleaved points, there should be fewer cache
   * misses than if each thread were given different stretches of x to
   * convolve.
   */
  in = 0;
  #ifdef _OPENMP
  #pragma omp parallel for firstprivate(in) schedule(static,1)
  #endif
  for (out=0; out < N; out++) {
    /* width of resolution window for x is w = 2 dx^2. */
    const double sigma = dx[out];
    const double xo = x[out];
    const double limit = sqrt(-2.*sigma*sigma* LOG_RESLIMIT);

    /* if (out%20==0) printf("%d: x,dx = %g,%g\n",out,xo,sigma); */

    /* Line up the left edge of the convolution window */
    /* It is probably forward from the current position, */
    /* but if the next dx is a lot higher than the current */
    /* dx or if the x are not sorted, then it may be before */
    /* the current position. */
    /* FIXME verify that the convolution window is just right */
    while (in < Nin-1 && xin[in] < xo-limit) in++;
    while (in > 0 && xin[in] > xo-limit) in--;

    /* Special handling to avoid 0/0 for w=0. */
    if (sigma > 0.) {
      y[out] = convolve_point(xin,yin,in,Nin,xo,limit,sigma);
    } else if (in < Nin-1) {
      /* Linear interpolation */
      double m = (yin[in+1]-yin[in])/(xin[in+1]-xin[in]);
      double b = yin[in] - m*xin[in];
      y[out] = m*xo + b;
    } else if (in > 0) {
      /* Linear extrapolation */
      double m = (yin[in]-yin[in-1])/(xin[in]-xin[in-1]);
      double b = yin[in] - m*xin[in];
      y[out] = m*xo + b;
    } else {
      /* Can't happen because there is more than one point in xin. */
      assert(Nin>1);
    }
  }

}
