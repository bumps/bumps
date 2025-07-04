**************
Change History
**************

v1.0.0 2025-06-24
-----------------
* a new web-based GUI (webview) has been added as the default GUI
* a dataclass-based JSON serialization is available for bumps classes
  (and the serializer can be used for model classes as well, falling back to `dill`)
* `FitProblem` class was refactored to support simultaneous fitting of multiple models, and
  `MultiFitProblem` is deprecated (currently an alias to `FitProblem`)
* `Fitness` is a Protocol class defining the interface for models
* `FitProblem.models` is now a generator that returns Fitness-compatible objects
* `Parameter` was refactored and has an indirection attribute `slot`:
  * `Parameter.value` now returns `float(Parameter.slot)`
  * this allows `Parameter` to be a wrapper of other `Parameter` or `Expression` or `Calculation` or `Constant` objects
* `Expression` is a new class that replaces `Operation` for combining `Parameter` objects in expressions
  (can be used in `Parameter` slots, and in a model)
* Use `Parameter.equals(Expression|Parameter)` for equality constraints. This preserves bounds
  and limits on the parameter imposed by the model. Calling `Parameter.unlink()` clears the constraint.
* `Calculation` is a new class to represent a calculable property of a class, which can be used in `Expression` instances
  (e.g. the total thickness of a stack of layers)
* new `Constraint` class that can be created with e.g. `constraints = [par1 < par2, par2 < par3]` and passed directly to FitProblem
* the `DE` stepper (also used by `dream`) is now fully accelerated with numba when numba is present (C DLL is still faster)
* `setup.py` removed and the package is now defined in `pyproject.toml`
* when using `trim` option in `dream`, the calculated stable portion is stored in the Dream state (`MCMCDraw.portion`),
  * this portion is used by default in `MCMCDraw.draw()` and plotting methods

v0.9.3 2024-07-09
-----------------
* fixed issues with numpy > = 2.0
  (see #140, `numpy.NaN` deprecated and removed)
* fixed issues with refactor of scipy.stats (see #139)

v0.9.2 2024-03-05
-----------------
* added testing for python 3.12
* fixed issue with matplotlib >= 3.8.0 (see #129)
* added missing documents to dream manual
  (Convergence tests, Parallel coordinates plot)
* added numba.njit-accelerated fallback bounds.apply methods to dream
  (still uses compiled C DLL if available)
* provide MAX_CORR attribute on the CorrelationView; clear the figure
  if the number of variables exceeds MAX_CORR

v0.9.1 2023-04-10
-----------------
* added support for python 3.11, scipy 1.10, numpy 1.24, wx 4.1.1
* fixed covariance calculation for n-D datasets
* fixed batch mode I/O redirection cleanup
* fixed issue with DREAM bounds checker when running in parallel
* default to single precision derivatives with lm (fixes issue in SasView
  where OpenCL models failed with Levenberg-Marquardt)
* improved support for repeat fitting within scripts and notebooks
  (*start_mapper* should now work after *stop_mapper*)

v0.9.0 2022-03-15
-----------------
* use MPFit in place of scipy.leastsq for bounds-constrained Levenberg-Marquardt

Breaking change:
* simultaneous fit now scales individual nllfs by squared weight rather than weight

v0.8.1 2021-11-18
-----------------
* "apply parameters" action added to GUI menu (does the same as --pars flag in CLI)
* operators refactored (no more eval)
* BoundedNormal keywords renamed (sigma, mu) -> (std, mean)
* support for numba usage in models
* fixed Parameters view jumping to top after toggling fit (linux, Mac)
* fixed Summary view sliders disappearing in linux
* fixed uncertainty plots regenerating at each parameter update
* improved documentation of uncertainty analysis

Breaking change:
* python 2.7 support discontinued

v0.8.0 2020-12-16
-----------------
* add stopping conditions to DREAM, using *--alpha=p-value* to reject convergence
* require *--overwrite* or *--resume* when reusing a store directory
* enable outlier trimming in DREAM with --outliers=iqr
* add fitted slope and loglikelihood distribution to the loglikelihood plot
* display seed value used for fit so it can be rerun with *--seed*
* save MCMC files using gzip
* remove R stat from saved state
* restore *--pars* option, which was broken in 0.7.17
* terminate the MPI session when the fit is complete instead of waiting for the
  allocation to expire
* allow a series of fits in the same MPI session
* support newest matplotlib

v0.7.18 2020-11-16
------------------
* restore python 2.7 support

v0.7.17 2020-11-06
------------------
* restore DREAM fitter efficiency (it should now require fewer burn-in steps)
* errplot.reload_errors allows full path to model file
* clip values within bounds at start of fit so constraints aren't infinite
* allow *--entropy=gmm|mvn|wnn|llf* to specify entropy estimation algorithm
* allow duplicate parameter names in model on reload
* expand tilde in path names
* GUI: restore parallel processing
* GUI: suppress uncertainty updates during fit to avoid memory leak
* disable broken fitters: particle swarm, random lines, snobfit
* minor doc changes

v0.7.16 2020-06-11
------------------
* improved handling of parameters for to_dict() json pickling

v0.7.15 2020-06-09
------------------
* parallel fitting suppressed in GUI for now---need to reuse thread pool
* support *limits=(min, max)* for pm and pmp parameter ranges
* cleaner handling of single/multiple fit specification
* fix *--entropy* command line option
* better support for pathlib with virtual file system

v0.7.14 2020-01-03
------------------

* support for *--checkpoint=n*, which updates the .mc files every n hours
* fix bug for stuck fits on *--resume*: probabilities contain NaN
* better error message for missing store directory
* Python 3.8 support (time.clock no longer exists)


v0.7.13 2019-10-15
------------------

* fix pickle problem for parameterized functions
* support multi-valued functions in Curve, shown with a coupled ODE example
* update support for newer numpy and matplotlib

v0.7.12 2019-07-30
------------------

* --parallel defaults to using one process per CPU.
* --pop=-k sets population size to k rather than k times num parameters
* --resume=- resumes from --store=/path/to/store
* use expanded canvas for parameter histograms to make plots more readable
* use regular spaced tics for parameter histograms rather than 1- and 2-sigma
* improve consistency between values of cov, stderr and chisq
* fix handling of degenerate ranges on parameter output
* add entropy calculator using gaussian mixture models (default is still Kramer)
* vfs module allows loading of model and data from zip file (not yet enabled)
* warn when model has no fitted parameters
* update mpfit to support python 3
* support various versions of scipy and numpy

v0.7.11 2018-09-24
------------------

* add support for parameter serialization

v0.7.10 2018-06-15
------------------

* restructure parameter table in gui

v0.7.9 2018-06-14
-----------------

* full support for python 3 in wx GUI
* allow added or missing parameters in reloaded .par file
* add dream state to return from fit() call

v0.7.8 2018-05-18
-----------------

* fix source distribution (bin directory was missing)

v0.7.7 2018-05-17
-----------------

* merge in amdahl branch for improved performance
* update plot so that the displayed "chisq" is consistent with nllf
* slight modification to the DREAM DE crossover ratio so that no crossover
  weight ever goes to zero.
* par.dev(std) now uses the initial value of the parameter as the center of the
  distribution for a gaussian prior on par, as stated in the documentation. In
  older releases it was incorrectly defaulting to mean=0 if the mean was
  not specified.
* save parameters and uncertainties as JSON as well as text
* convert discrete variables to integer prior to computing DREAM statistics
* allow relative imports from model files
* support latest numpy/matplotlib stack
* initial support for wxPhoenix/python 4 GUI (fit ranges can't yet be set)

v0.7.6 2016-08-05
-----------------

* add --view option to command line which gets propagated to the model plotter
* add support for probability p(x) for vector x using VectorPDF(f,x0)
* rename DirectPDF to DirectProblem, and allow it to run in GUI
* data reader supports multi-part files, with parts separated by blank lines
* add gaussian mixture and laplace examples
* bug fix: plots were failing if model name contains a '.'
* miscellaneous code cleanup

v0.7.5.10 2016-05-04
--------------------

* gui: undo code cleaning operation which broke the user interface

v0.7.5.9 2016-04-22
-------------------

* population initializers allow indefinite bounds
* use single precision criterion for levenberg-marquardt and bfgs
* implement simple, faster, less accurate Hessian & Jacobian
* compute uncertainty estimate from Jacobian if problem is sum of squares
* gui: fit selection window acts like a dialog

v0.7.5.8 2016-04-18
-------------------

* accept model.par output from a different model
* show residuals with curve fit output
* only show correlations for selected variables
* show tics on correlations if small number
* improve handling of uncertainty estimate from curvature
* tweak dream algorithm -- maybe improve the acceptance ratio?
* allow model to set visible variables in output
* improve handling of arbitrary probability density functions
* simplify loading of pymc models
* update to numdifftools 0.9.14
* bug fix: improved handling of ill-conditioned fits
* bug fix: avoid copying mcmc chain during run
* bug fix: more robust handling of --time limit
* bug fix: support newer versions of matplotlib and numpy
* miscellaneous tweaks and fixes

v0.7.5.7 2015-09-21
-------------------

* add entropy calculator (still unreliable for high dimensional problems)
* adjust scaling of likelihood (the green line) to match histogram area
* use --samples to specify the number of samples from the distribution
* mark this and future releases with a DOI at zenodo.org

v0.7.5.6 2015-06-03
-------------------

* tweak uncertainty calculations so they don't fail on bad models

v0.7.5.5 2015-05-07
-------------------

* documentation updates

v0.7.5.4 2014-12-05
-------------------

* use relative rather than absolute noise in dream, which lets us fit target
  values in the order of 1e-6 or less.
* fix covariance population initializer

v0.7.5.3 2014-11-21
-------------------

* use --time to stop after a given number of hours
* Levenberg-Marquardt: fix "must be 1-d or 2-d" bug
* improve curvefit interface

v0.7.5.2 2014-09-26
-------------------

* pull numdifftools dependency into the repository

v0.7.5.1 2014-09-25
-------------------

* improve the load_model interface

v0.7.5 2014-09-10
-----------------

* Pure python release
