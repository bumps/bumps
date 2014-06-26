/* This program is public domain. */

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <math.h>

#ifdef SGI
#include <ieeefp.h>
#endif

/* Convolution of two linear splines. */
double convolve_point_sampled(
    size_t Nin, const double xin[], const double yin[],
    size_t Np, const double xp[], const double yp[],
    double xo, double dx, size_t in)
{
    // Walk the theory spline and the resolution spline together, computing
    // the integral of the pairs of line segments.  Since the spline knots
    // do not necessarily correspond, we need to treat each individual segment
    // in both curves piece-wise, integrating from knot to knot in the union
    // of the two knot sets.

    // Need an undefined value in case the algorithm is broken and we don't
    // initialize the initial lines.  This would ideal be NaN, but that isn't
    // available in MSVC, so set it to something very large.  In order to avoid
    // normalizing that very large value to 1., only set the theory line to
    // undefined, and set the resolution line to 0.
    //const double undefined = 0./0.;
    const double undefined = 1e308;
    double m1=undefined, b1=undefined, m2=0., b2=0.;
    double sum = 0.;
    double norm = 0.;
    size_t p;
    double delta,delta2,delta3;
    double x, next_x, next_xin, next_xp;

    // Set the target start of the integral to xin[in].  This may or may
    // not be beyond the end of the resolution function, depending on whether
    // we are at the left edge of the data, or somewhere inside.
    next_xin = xin[in];

    // Set p to the point just before next_xin
    for (p=1; p < Np; p++) {
        if (xo + dx*xp[p] > next_xin) break;
    }
    next_xp = xo + dx*xp[--p];

    // Choose the larger of next_xp and next_xin as the starting point of the
    // integral.  This will force us to step both xin and xp in the first
    // iteration of the loop, computing new slope/intercepts for both lines.
    // If the theory line extends beyond the resolution function, then we will
    // be called with xin[in] before the first resolution point, and next_xp
    // will be bigger than next_xin.  If the resolution extends beyond the
    // end of the theory, then the next_xp will be set before next_xin, and
    // next_xin will be the start of the integral.  The integral ends when
    // either the resolution or the theory runs out.
    //
    // We are tracking the area under the resolution as well as the area
    // under the product of theory and resolution so that we can properly
    // normalize the data when less than the full resolution is included.
    // This means that we do not need to provide a resolution function
    // normalized to a total area of 1 as our input.
    x = (next_xp > next_xin ? next_xp : next_xin);
    //printf("  point xo:%g dx:%g x:%g, p:%ld, in:%ld, xp:%g, xin:%g\n",
    //       xo,dx,x,p,in,next_xp,next_xin);
    while (1) {
        // Step xin if we are at the next theory point
        if (next_xin <= x) {
            in++;
            if (in >= Nin) break; // At the right edge of the data
            next_xin = xin[in];
            m1 = (yin[in] - yin[in-1]) / (xin[in] - xin[in-1]);
            b1 = yin[in] - m1*xin[in];
        }
        // Step xp if we are at the next resolution point
        if (next_xp <= x) {
            p++;
            if (p >= Np) break; // At the right edge of the resolution
            next_xp = xo + dx*xp[p];
            m2 = (yp[p] - yp[p-1]) / (xp[p] - xp[p-1]) / dx;
            b2 = yp[p] - m2*next_xp;
        }
        // Find the next node
        next_x = (next_xin < next_xp  ? next_xin : next_xp);
        // Compute the convolution and norm between the current and next node
        delta = next_x - x;
        delta2 = next_x*next_x - x*x;
        delta3 = next_x*next_x*next_x - x*x*x;
        norm += 0.5*m2*delta2 + b2*delta;
        sum += m1*m2/3.0*delta3 + 0.5*(m1*b2+m2*b1)*delta2 + b1*b2*delta;
        // printf("  delta:%g delta2:%g delta3:%g norm:%g sum:%g m1:%g b1:%g m2:%g b2:%g x:%g nx:%g ni:%g np:%g\n",
        //  delta, delta2, delta3, norm, sum, m1, b1, m2, b2, x, next_x, next_xin, next_xp);
        // Move to the next node
        x = next_x;
    }
    return sum/norm;
}


void
convolve_sampled(size_t Nin, const double xin[], const double yin[],
         size_t Np, const double xp[], const double yp[],
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
    const double limit = -dx[out]*xp[0];
    const double xo = x[out];

    /* Line up the left edge of the convolution window */
    /* It is probably forward from the current position, */
    /* but if the next dx is a lot higher than the current */
    /* dx or if the x are not sorted, then it may be before */
    /* the current position. */
    /* FIXME verify that the convolution window is just right */
    while (in < Nin-1 && xin[in] < xo-limit) in++;
    while (in > 0 && xin[in] > xo-limit) in--;

    /* Special handling to avoid 0/0 for w=0. */
    if (dx[out] > 0.) {
      // printf("convolve in:%ld out:%ld, xo:%g dx:%g\n",in,out,xo,dx[out]);
      y[out] = convolve_point_sampled(Nin,xin,yin,Np,xp,yp,xo,dx[out],in);
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
