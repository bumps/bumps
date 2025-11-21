"""
Run a bumps model using a nested sampler.

Currently only ultranest is supported, though it uses some functions from dynesty.

Usage:

    pip install ultranest dynesty
    python -m bumps.nested  --pars=model.par  --export=results/path model.py

where model.par is the path to a par file exported from a bumps fit, and
model.py is the script describing the model.
"""

import argparse
import numpy as np
import csv
import json

# os.environ['OMP_NUM_THREADS'] = '7'
# export OMP_NUM_THREADS=7


def put01(v, low, high):
    return (high - low) * v + low


class Sampler:
    def __init__(self, problem, store):
        import ultranest
        import ultranest.stepsampler

        # TODO: need to use the --session option as done in bumps to parse the name
        self.save_name = store
        self.problem = problem
        self.problem.model_reset()
        # TODO: don't access problem privates
        self.parameters = self.problem._parameters
        print(type(self.parameters[0].name))
        # fit space dimension is length of parameter vector
        self.param_names = self.get_param_names()
        self.ndim = len(problem.getp())
        self.nested_sampler = ultranest.ReactiveNestedSampler(
            self.param_names, self.logl, self.prior_transform, log_dir=self.save_name, resume=True
        )

        # nsteps = 2 * len(self.param_names)
        self.nested_sampler.stepsampler = ultranest.stepsampler.RegionSliceSampler(
            nsteps=400, adaptive_nsteps="move-distance"
        )

    def get_param_names(self):
        param_names = []
        for i, v in enumerate(self.parameters):
            param_names.append(v.name)
        return param_names

    def logl(self, x):
        # Note: only using nllf from experiment; parameter_nllf is handled implicitly by prior_transform
        # and constraints_nllf is ignored for now.
        self.problem.setp(x)
        return -self.problem.model_nllf()

    def prior_transform(self, u):
        x = [put01(uk, *pk.bounds) for uk, pk in zip(u, self.parameters)]
        return np.array(x, "d")

    def sample(self, verbose=True):
        print("MPI:", self.nested_sampler.mpi_size, self.nested_sampler.mpi_rank)
        self.results = self.nested_sampler.run(frac_remain=0.5, min_ess=30000, min_num_live_points=3000)
        # results = self.nested_sampler.results
        # print(f"results = {self.results}")
        # print(f"results['samples'] type = {type(results['samples'])}")

        # Calculate parameter means.
        # weights = np.exp(results.logwt -  results.logz[-1])

        # mean, _ = dyfunc.mean_and_cov(results.samples, weights)
        mean = self.results["posterior"]["mean"]
        # print(type(mean))

        # Save bumps results
        self.save_bumps_pars(self.problem)
        self.problem.save(f"{self.save_name}")

        # Save UltraNest results
        # self.corner(results)
        self.save_as_csv(self.results)
        # self.save_as_json(self.results)
        # np.savetxt('mean.dat', mean)
        # cornerplot(self.results)
        self.nested_sampler.plot()

    def results_df(self):
        import pandas as pd

        df = pd.DataFrame(data=self.results["samples"], columns=self.results["paramnames"])
        df.describe()

        return df.loc["mean"]

    def corner(self, results):
        from dynesty import plotting as dyplot

        fig, _ = dyplot.cornerplot(
            results,
            color="blue",
            quantiles=None,
            show_titles=True,
            max_n_ticks=3,
            truths=np.zeros(self.ndim),
            truth_color="black",
        )

        # Label corner plot with parameter names.
        axes = np.reshape(np.array(fig.get_axes()), (self.ndim, self.ndim))
        for i in range(1, self.ndim):
            for j in range(self.ndim):
                if i == self.ndim - 1:
                    axes[i, j].set_xlabel(self.parameters[j].name)
                if j == 0:
                    axes[i, j].set_ylabel(self.parameters[i].name)
        axes[self.ndim - 1, self.ndim - 1].set_xlabel(self.parameters[-1].name)

        fig.savefig(f"{self.save_name}_corner_plot.png")
        return fig

    def save_as_csv(self, results):
        w = csv.writer(open(f"{self.save_name}_ultranest_results.csv", "w"))
        for key, val in results.items():
            w.writerow([key, val])

    def save_as_json(self, results):
        with open(f"{self.save_name}_ultranest_results.json", "w") as fp:
            json.dump(results, fp, sort_keys=True, indent=4)

    def save_bumps_pars(self, problem):
        pardata = "".join("%s %.15g\n" % (name, value) for name, value in zip(problem.labels(), problem.getp()))
        # TODO: implement below once --session is done
        # open(problem.output_path + ".par", 'wt').write(pardata)
        open(self.save_name + ".par", "wt").write(pardata)


def main():
    from bumps.cli import load_model, load_best

    parser = argparse.ArgumentParser(
        description="run nested sampling on bumps model with ultranest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-p", "--pars", type=str, default="", help="retrieve starting point from .par file")
    parser.add_argument("-x", "--export", type=str, default="", help="store folder for UltraNest results")
    parser.add_argument("modelfile", type=str, nargs=1, help="bumps model file")
    parser.add_argument("modelopts", type=str, nargs="*", help="options passed to the model")
    opts = parser.parse_args()

    try:
        import ultranest, dynesty
    except ImportError:
        print("Nested samplers not available. Use 'pip install ultranest dynesty' first.")
        return

    print(opts)
    print(opts.modelfile)
    problem = load_model(opts.modelfile[0], model_options=opts.modelopts)
    if opts.pars:
        load_best(problem, opts.pars)
    sampler = Sampler(problem, opts.export)
    sampler.sample()


if __name__ == "__main__":
    main()
