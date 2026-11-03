"""
Run a bumps model using a nested sampler.

Nested sampling is now a regular bumps fitter, so this script is no longer
needed. It is kept as a thin wrapper around that fitter and will be removed
in a future release.

Instead of this script, new scripts should use:

    bumps --fit=nested_sampling --export=results/path model.py

or, from python:

    from bumps.fitters import fit
    result = fit(problem, method="nested_sampling", export="results/path")

To use this script instead, call:

    python -m bumps.nested --pars=model.par --export=results/path model.py

where model.par is the path to a par file exported from a bumps fit, and
model.py is the script describing the model.
"""

import argparse
import warnings

DEPRECATION_MESSAGE = (
    "bumps.nested is deprecated and will be removed in a future release. "
    "Use the nested_sampling fitter instead: "
    "'bumps --fit=nested_sampling --export=results/path model.py' from the command line, "
    "or bumps.fitters.fit(problem, method='nested_sampling') from python."
)

warnings.warn(DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)


class Sampler:
    """
    Deprecated: Use the `nested_sampling` fitter via `bumps.fitters.fit`.

    Thin wrapper over `bumps.fitters.NestedSamplerFit` kept so that
    existing scripts continue to run. *store* is the directory for the exported
    fit results; use an empty string to skip the export.
    """

    def __init__(self, problem, store):
        warnings.warn(DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
        self.problem = problem
        self.save_name = store
        self.results = None

    def sample(self, verbose=True):
        """
        Run the fit.
        """
        from .fitters import fit

        self.results = fit(
            self.problem,
            method="nested_sampling",
            export=self.save_name if self.save_name else None,
            verbose=verbose,
        )
        return self.results


def main():
    from .fitproblem import load_pars, load_problem

    parser = argparse.ArgumentParser(
            description="Run nested sampling on a bumps model.\n\nThis is deprecated: new scripts should use 'bumps --fit=nested_sampling' instead.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument("-p", "--pars", type=str, default="", help="retrieve starting point from .par file")
    parser.add_argument("-x", "--export", type=str, default="", help="store folder for the fit results")
    parser.add_argument("modelfile", type=str, nargs=1, help="bumps model file")
    parser.add_argument("modelopts", type=str, nargs="*", help="options passed to the model")
    opts = parser.parse_args()

    # Make sure the user sees the deprecation even if warnings are suppressed.
    print(f"Warning: {DEPRECATION_MESSAGE}")

    problem = load_problem(opts.modelfile[0], args=opts.modelopts)
    if opts.pars:
        load_pars(problem, opts.pars)
    sampler = Sampler(problem, opts.export)
    sampler.sample()


if __name__ == "__main__":
    main()
