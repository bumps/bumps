#ifndef _REBIN2D_H
#define _REBIN2D_H

#include <iostream>
#include <vector>

#include "rebin.h"

template <typename T> void
print_bins(const std::string &message,
	   size_t nx, size_t ny, const T z[])
{
  std::cout << message << std::endl;
  for (size_t i=0; i < nx; i++) {
    for (size_t j=0; j < ny; j++) {
      std::cout << " " << z[i+j*nx];
    }
    std::cout << std::endl;
  }
}

template <typename T> void
print_bins(const std::string &message,
	   const std::vector<double> &x,
	   const std::vector<double> &y,
	   const std::vector<T> &z)
{
  size_t nx = x.size()-1;
  size_t ny = y.size()-1;
  assert(nx*ny == z.size());
  print_bins(message,nx,ny,&z[0]);
}

// rebin_counts(Nxold, xold, Nyold, yold, Iold,
//              Nxnew, xnew, Nynew, ynew, Inew)
// Rebin from old to new where:
//    Nxold,Nyold number of original bin edges
//    xold[Nxold+1], yold[Nyold+1] bin edges in x,y
//    Iold[Nxold*Nyold] input array
//    Nxnew,Nynew number of desired bin edges
//    xnew[Nxnew+1], ynew[Nynew+1] desired bin edges
//    Inew[Nxnew*Nynew] result array
template <typename T> void
rebin_counts_2D(
        const size_t Nxold, const double xold[],
	const size_t Nyold, const double yold[],
	const T Iold[],
	const size_t Nxnew, const double xnew[],
	const size_t Nynew, const double ynew[],
	T Inew[])
{

  // Clear the new bins
  for (size_t i=0; i < Nxnew*Nynew; i++) Inew[i] = 0;

  // Traverse both sets of bin edges; if there is an overlap, add the portion
  // of the overlapping old bin to the new bin.  Scale this by the portion
  // of the overlap in y.
  BinIter<double> from(Nxold, xold);
  BinIter<double> to(Nxnew, xnew);
  while (!from.atend && !to.atend) {
    if (to.hi <= from.lo) ++to; // new must catch up to old
    else if (from.hi <= to.lo) ++from; // old must catch up to new
    else {
      const double overlap = std::min(from.hi,to.hi) - std::max(from.lo,to.lo);
      const double portion = overlap/(from.hi-from.lo);
      rebin_counts_portion(Nyold, yold, Iold+from.bin*Nyold,
                           Nynew, ynew, Inew+to.bin*Nynew,
                           portion);
      if (to.hi > from.hi) ++from;
      else ++to;
    }
 }

}

template <typename T> inline void
rebin_counts_2D(const std::vector<double> &xold,
	const std::vector<double> &yold,
	const std::vector<T> &Iold,
	const std::vector<double> &xnew,
	const std::vector<double> &ynew,
	std::vector<T> &Inew)
{
  assert( (xold.size()-1)*(yold.size()-1) == Iold.size());
  Inew.resize( (xnew.size()-1)*(ynew.size()-1) );
  rebin_counts_2D(xold.size()-1, &xold[0], yold.size()-1, &yold[0], &Iold[0],
                  xnew.size()-1, &xnew[0], ynew.size()-1, &ynew[0], &Inew[0]);
}

#endif
