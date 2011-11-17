
#include <iostream>
#include <cmath>
#include <cassert>
#include "rebin2D.h"

const double tolerance = 1e-15;

int check_bins(const char msg[], int nx, int ny,
	       const double result[], const double expected[])
{
  // Check that result is a uniform field
  int retval = 0;
  std::cout << "Checking " << msg << "...";
  for (int i=0; i < nx; i++) {
    for (int j=0; j < ny; j++) {
      int k = i+j*nx;
      if (fabs(result[k]-expected[k]) >= tolerance) {
	std::cout << "["<<i<<","<<j<<"] = " << result[k]
		  << ", not " << expected[k] << "  " << std::endl;
	retval = 1;
      }
    }
  }
  if (retval) std::cout << "FAIL\n";
  else std::cout << "PASS\n";
  return retval;
}

int nonuniform_test(const char msg[],
		    int nx, const double x[],
		    int ny, const double y[],
		    int nz, const double z[],
		    int nox, const double ox[],
		    int noy, const double oy[],
		    int noz, const double oz[])
{
  assert((nx-1)*(ny-1) == nz);
  assert((nox-1)*(noy-1) == noz);
  std::vector<double> result(noz);
  rebin_counts_2D(nx-1, x, ny-1, y, z, nox-1, ox, noy-1, oy, &result[0]);
#ifdef DEBUG
  print_bins("Initial matrix",nx-1,ny-1,&z[0]);
  print_bins("rebinned matrix",nox-1,noy-1,&result[0]);
#endif
  return check_bins(msg,nox-1,noy-1,&result[0],oz);
}

int uniform_test(const char msg[],
		 int nx, const double x[], int ny, const double y[])
{
  // Create bins based on pixel area
  std::vector<double> bins(nx*ny), result(nx*ny);
  for (int i=0; i < nx; i++) {
    for (int j=0; j < ny; j++) {
      bins[i+j*nx] = x[i]*y[j];
    }
  }

  // create input bin edges
  std::vector<double> edges_x(nx+1);
  std::vector<double> edges_y(ny+1);
  edges_x[0] = edges_y[0] = 0;
  for (int i=0; i < nx; i++) edges_x[i+1] = edges_x[i]+x[i];
  for (int j=0; j < ny; j++) edges_y[j+1] = edges_y[j]+y[j];

  // create output bin edges
  int out_nx = int(floor(edges_x[nx]));
  int out_ny = int(floor(edges_y[ny]));
  std::vector<double> out_x(out_nx+1);
  for (int i=0; i < out_nx+1; i++) out_x[i] = i;
  std::vector<double> out_y(out_ny+1);
  for (int i=0; i < out_ny+1; i++) out_y[i] = i;

  // create output bins
  std::vector<double> out;
  std::vector<double> expected(out_nx*out_ny,1.);

  // rebin
  rebin_counts_2D(edges_x, edges_y, bins, out_x, out_y, out);

#ifdef DEBUG
  print_bins("Initial matrix",nx,ny,&bins[0]);
  print_bins("rebinned matrix",out_nx,out_ny,&out[0]);
#endif
  return check_bins(msg,out_nx,out_ny,&out[0],&expected[0]);
}

#define TEST_UNIFORM(MSG,X,Y) do {	\
    int nx = sizeof(X)/sizeof(*X);	\
    int ny = sizeof(Y)/sizeof(*Y);	\
    retval = retval || uniform_test(MSG,nx,X,ny,Y);	\
 } while (0)

#define TEST_NONUNIFORM(MSG,X,Y,Z,OX,OY,OZ) do {	\
    int nx = sizeof(X)/sizeof(*X);   \
    int ny = sizeof(Y)/sizeof(*Y);   \
    int nz = sizeof(Z)/sizeof(*Z);   \
    int nox = sizeof(OX)/sizeof(*OX);   \
    int noy = sizeof(OY)/sizeof(*OY);   \
    int noz = sizeof(OZ)/sizeof(*OZ);   \
    retval = retval || nonuniform_test(MSG,nx,X,ny,Y,nz,Z,nox,OX,noy,OY,noz,OZ); \
  } while (0)


int main(int argc, char *argv[])
{
  int retval = 0;

#ifdef SPEED_CHECK // cost to rebin a 250x300x1000 dataset
  {
    std::vector<double> xbin(251);
    std::vector<double> xrebin(251);
    std::vector<double> ybin(301);
    std::vector<double> yrebin(301);
    std::vector<double> val(250*300);
    std::vector<double> result(250*300);

    // shifted binning
    for (size_t i=1; i < xbin.size()-1; i++) xbin[i]=i+.2;
    for (size_t i=1; i < ybin.size()-1; i++) ybin[i]=i+.2;
    xbin[0]=ybin[0]=0.;
    xbin[xbin.size()-1] = xbin.size()-1;
    ybin[ybin.size()-1] = ybin.size()-1;

    // Flat field
    for (size_t i=0; i < val.size(); i++) val[i]=10;

    // regular rebinning
    for (size_t i=0; i < xbin.size(); i++) xbin[i]=i;
    for (size_t i=0; i < ybin.size(); i++) ybin[i]=i;
    //return retval;

    for (size_t i = 0; i < 10000; i++) {
      rebin_counts_2D(xbin, ybin, val, xrebin, yrebin, result);
    }
  }

#else // ! SPEED_CHECK

  {  double //
      x[] = {1,2.5,4,0.5},
      y[] = {3,1,2.5,1,3.5};
      TEST_UNIFORM("uniform1",x,y);
  }
  {  double //
      x[] = {3,2},
      y[] = {1,2};
      TEST_UNIFORM("uniform2",x,y);
  }
  {  double //
      x[] = {0,3,5},
      y[] = {0,1,3},
      z[] = {3,2,6,4},
      ox[]= {0,1,2,3,4,5},
      oy[]= {0,1,2,3},
      oz[]= {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
      TEST_NONUNIFORM("nonuniform2",x,y,z,ox,oy,oz);
  }
  {  double //
      x[] = {-1,2,4},
      y[] = {0,1,3},
      z[] = {3,2,6,4},
      ox[] = {1,2},
      oy[] = {1,2},
      oz[] = {1};
      TEST_NONUNIFORM("nonuniform3",x,y,z,ox,oy,oz);
  }

#endif // !SPEED_CHECK

  return retval;
}
