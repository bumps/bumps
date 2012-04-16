#ifndef _REBIN_H
#define _REBIN_H

#include <iostream>
#include <vector>
#include <stdexcept>

// Define a bin iterator to adapt to either forward or reversed inputs.
template <typename T>
class BinIter {
  bool forward;
  size_t n;
  const T *edges;
public:
  size_t bin;     // Index of the corresponding bin
  double lo, hi; // Low and high values for the bin edges.
  bool atend;  // True when we increment beyond the final bin.
  BinIter(size_t _n, const T *_edges) {
    // n is number of bins, which is #edges-1
    // edges are the values of the bin edges.
    n = _n; edges = _edges;
    forward = edges[0] < edges[n];
    if (forward) {
      bin = 0;
      lo = edges[0];
      hi = edges[1];
    } else {
      bin = n - 1;
      lo = edges[n];
      hi = edges[n-1];
    }
    atend = n < 1;
  }
  BinIter& operator++() {
    if (atend) {
      throw std::out_of_range("moving beyond final bin");
    }
    lo = hi;
    if (forward) {
      bin++;
      atend = (bin >= n);
      if (!atend) hi = edges[bin+1];
    } else {
      bin--;
      atend = (bin < 0);
      if (!atend) hi = edges[bin];
    }
    return *this;
  }
};

// TODO: with integer counts, the total counts after rebinning is not
// TODO: preserved.  Rounding would improve the situation somewhat,
// TODO: but a better algorithm would keep track of accumulated error
// TODO: and keep it between [-0.5,0.5].  This is more complicated
// TODO: with multidimensional rebinning.

// Do the rebinning.  The ND_portion parameter scales each bin by the
// given amount, which is useful when doing multidimensional rebinning.
template <typename T> void
rebin_counts_portion(const size_t Nold, const double vold[], const T Iold[],
                     const size_t Nnew, const double vnew[], T Inew[],
                     const double ND_portion)
{
  // Note: inspired by rebin from OpenGenie, but using counts per bin
  // rather than rates.

  // Does not work in place
  assert(Iold != Inew);

  // Traverse both sets of bin edges; if there is an overlap, add the portion
  // of the overlapping old bin to the new bin.
  BinIter<double> from(Nold, vold);
  BinIter<double> to(Nnew, vnew);
  while (!from.atend && !to.atend) {
    //std::cout << "from " << from.bin << ": [" << from.lo << ", " << from.hi << "]\n";
    //std::cout << "to " << to.bin << ": [" << to.lo << ", " << to.hi << "]\n";
    if (to.hi <= from.lo) ++to; // new must catch up to old
    else if (from.hi <= to.lo) ++from; // old must catch up to new
    else {
      const double overlap = std::min(from.hi,to.hi) - std::max(from.lo,to.lo);
      const double portion = overlap/(from.hi-from.lo);
      Inew[to.bin] += T(Iold[from.bin]*portion*ND_portion);
      if (to.hi > from.hi) ++from;
      else ++to;
    }
  }
}

// rebin_counts(Nx, x, Ix, Ny, y, Iy)
// Rebin from x to y where:
//    Nx is the number of bins in the data
//    x[Nx+1] is a vector of bin edges
//    I[Nx] is a vector of counts
//    Ny is the number of bins desired
//    y[Ny+1] is a vector of bin edges
//    I[Ny] is a vector of counts
template <typename T> void
rebin_counts(const size_t Nold, const double xold[], const T Iold[],
             const size_t Nnew, const double xnew[], T Inew[])
{
  // Note: inspired by rebin from OpenGenie, but using counts per bin
  // rather than rates.

  // Clear the new bins
  for (size_t i=0; i < Nnew; i++) Inew[i] = 0;

  rebin_counts_portion(Nold, xold, Iold, Nnew, xnew, Inew, 1.);
}

template <typename T> inline void
rebin_counts(const std::vector<double> &xold, const std::vector<T> &Iold,
             const std::vector<double> &xnew, std::vector<T> &Inew)
{
  assert(xold.size()-1 == Iold.size());
  Inew.resize(xnew.size()-1);
  rebin_counts(Iold.size(), &xold[0], &Iold[0],
               Inew.size(), &xnew[0], &Inew[0]);
}

// rebin_intensity(Nx, x, Ix, dIx, Ny, y, Iy, dIy)
// Like rebin_counts, but includes uncertainty.  This could of course be
// done separately, but it will be faster to rebin both at the same time.
template <typename T> void
rebin_intensity(const size_t Nold, const double xold[],
		const T Iold[], const T dIold[],
		const size_t Nnew, const double xnew[],
		T Inew[], T dInew[])
{
  // Note: inspired by rebin from OpenGenie, but using counts per bin rather than rates.

  // Clear the new bins
  for (size_t i=0; i < Nnew; i++) dInew[i] = Inew[i] = 0;

  // Traverse both sets of bin edges; if there is an overlap, add the portion
  // of the overlapping old bin to the new bin.
  BinIter<double> from(Nold, xold);
  BinIter<double> to(Nnew, xnew);
  while (!from.atend && !to.atend) {
    //std::cout << "from " << from.bin << ": [" << from.lo << ", " << from.hi << "]\n";
    //std::cout << "to " << to.bin << ": [" << to.lo << ", " << to.hi << "]\n";
    if (to.hi <= from.lo) ++to; // new must catch up to old
    else if (from.hi <= to.lo) ++from; // old must catch up to new
    else {
      const double overlap = std::min(from.hi,to.hi) - std::max(from.lo,to.lo);
      const double portion = overlap/(from.hi-from.lo);

      Inew[to.bin] += Iold[from.bin]*portion;
      dInew[to.bin] += square(dIold[from.bin]*portion);  // add in quadrature
      if (to.hi > from.hi) ++from;
      else ++to;
    }
  }

  // Convert variance to standard deviation.
  for (size_t i=0; i < Nnew; i++) dInew[i] = sqrt(dInew[i]);
}

template <typename T> inline void
rebin_intensity(const std::vector<double> &xold,
		const std::vector<T> &Iold, const std::vector<T> &dIold,
		const std::vector<double> &xnew,
		std::vector<T> &Inew, std::vector<T> &dInew)
{
  assert(xold.size()-1 == Iold.size());
  assert(xold.size()-1 == dIold.size());
  Inew.resize(xnew.size()-1);
  dInew.resize(xnew.size()-1);
  rebin_intensity(Iold.size(), &xold[0], &Iold[0], &dIold[0],
		  Inew.size(), &xnew[0], &Inew[0], &dInew[0]);
}

template <typename T> inline void
compute_uncertainty(const std::vector<T> &counts,
		    std::vector<T> &uncertainty)
{
  uncertainty.resize(counts.size());
  for (size_t i=0; i < counts.size(); i++)
    uncertainty[i] = T(counts[i] != 0 ? sqrt(counts[i]) : 1);
}


#endif // _REBIN_H
