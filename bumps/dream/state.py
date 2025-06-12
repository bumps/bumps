"""
Sampling history for MCMC.

MCMC keeps track of a number of things during sampling.

The results may be queried as follows::

    draws, generation, thinning, total_generations
    sample(condition) returns draws, points, logp
    logp()            returns draws, logp
    acceptance_rate() returns draws, AR
    chains()          returns draws, chains, logp
    CR_weight()       returns draws, CR_weight
    best()            returns best_x, best_logp
    outliers()        returns outliers
    show()/save(file)/load(file)

Data is stored in circular arrays, which keeps the last N generations and
throws the rest away.

draws is the total number of draws from the sampler.

generation is the total number of generations. Due to tests for partially
full circular buffers throughout the code (state.generation < state.Ngen)
we are resetting generation to the size of the stored history on resume,
and setting the generation_offset to the start of the history. If you need the
number of generations across resume then use total_generations.

thinning is the number of generations per stored sample.

draws[i] is the number of draws including those required to produce the
information in the corresponding return vector.  Note that draw numbers
need not be linearly spaced, since techniques like delayed rejection
will result in a varying number of samples per generation.

logp[i] is the set of log likelihoods, one for each member of the population.
The logp() method returns the complete set, and the sample() method returns
a thinned set, with on element of logp[i] for each vector point[i, :].

AR[i] is the acceptance rate at generation i, showing the proportion of
proposed points which are accepted into the population.

chains[i, :, :] is the set of points in the differential evolution population
at thinned generation i.  Ideally, the thinning rate of the MCMC process
is chosen so that thinned generations i and i+1 are independent samples
from the posterior distribution, though there is a chance that this may
not be the case, and indeed, some points in generation i+1 may be identical
to those in generation i.  Actual generation number is i*thinning.

points[i, :] is the ith point in a returned sample.  The i is just a place
holder; there is no inherent ordering to the sample once they have been
extracted from the chains.  Note that the sample may be from a marginal
distribution.

R[i] is the Gelman R statistic measuring convergence of the Markov chain.

CR_weight[i] is the set of weights used for selecting between the crossover
ratios available to the candidate generation process of differential
evolution.  These will be fixed early in the sampling, even when adaptive
differential evolution is selected.

outliers[i] is a vector containing the thinned generation number at which
an outlier chain was removed, the id of the chain that was removed and
the id of the chain that replaced it.  We leave it to the reader to decide
if the cloned samples, point[:generation, :, removed_id], should be included
in further analysis.

best_logp is the highest log likelihood observed during the analysis and
best_x is the corresponding point at which it was observed.

generation is the last generation number
"""

# TODO: state should be collected in files as we go

__all__ = ["MCMCDraw", "load_state", "save_state"]

import os.path
import re
import gzip
from typing import List, Dict, Tuple, Literal, Union, Optional, Callable
from pathlib import Path

import numpy as np
from numpy import empty, sum, asarray, inf, argmax, hstack, dstack
from numpy import savetxt, reshape
from numpy.typing import NDArray

from .convergence import burn_point
from .outliers import identify_outliers
from .util import draw, rng
from .gelman import gelman

# Don't load hdf5 if you don't need it
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from h5py import Group
else:

    class Group: ...


EXT = ".mc.gz"
CREATE = gzip.open
# EXT = ".mc"
# CREATE = open


SelectionType = Optional[Dict[Union[int, Literal["logp"]], Tuple[float, float]]]


UNCERTAINTY_DTYPE = "d"
MAX_LABEL_LENGTH = 1024
LABEL_DTYPE = f"|S{MAX_LABEL_LENGTH}"
H5_COMPRESSION = 5


def _h5_write_field(group: "Group", field: str, data: Union[NDArray, str]):
    import h5py

    # print(f"h5 writing {field} in {group} with {data}")
    if isinstance(data, str):
        dtype = h5py.string_dtype(encoding="utf-8", length=len(data))
        return group.create_dataset(field, data=data, dtype=dtype)
    elif isinstance(data, (int, np.integer)):
        return group.create_dataset(field, data=data, dtype=np.int64)
    elif isinstance(data, (float, np.floating)):
        return group.create_dataset(field, data=data, dtype=np.double)
    else:
        return group.create_dataset(field, data=data, dtype=data.dtype, compression=H5_COMPRESSION)


def _h5_read_field(group: "Group", field: str):
    raw_data = group[field][()]
    size = raw_data.size
    if isinstance(raw_data, np.integer):
        return int(raw_data)
    elif isinstance(raw_data, np.floating):
        return float(raw_data)
    elif size is not None and size > 0:
        return raw_data
    else:
        return None


def h5dump(group: "Group", state: "MCMCDraw"):
    class Fields:
        total_generations = state.total_generations
        thinning = state.thinning
        gen_draws, gen_logp = state.logp(full=True)
        _, AR = state.acceptance_rate()
        thin_draws, thin_point, thin_logp = state.chains()
        update_draws, update_CR_weight = state.CR_weight()
        labels = np.array(state.labels, dtype=LABEL_DTYPE)
        # These are marked outliers. They should still be marked when reloading the
        # state, but they need to be cleared when updates are given. Make sure the
        # request for the population from dream includes the outliers.
        good_chains = None if isinstance(state._good_chains, slice) else state._good_chains
        best_x, best_logp = state.best()
        best_gen = state._best_gen
        portion = state.portion

    # print(f"wrote {Fields.total_generations} generations")
    # print(f"{state._gen_offset=} {state.Ngen=}")
    # print("wrote", Fields.__dict__)
    for field, value in Fields.__dict__.items():
        if field[0] != "_" and value is not None:
            _h5_write_field(group, field, value)


def h5load(group: "Group"):
    # print(f"dream.state.h5load {group}")
    class Fields: ...

    for field in group:
        setattr(Fields, field, _h5_read_field(group, field))
    # print("read", Fields.__dict__)
    # Guess dimensions
    Ngen = Fields.gen_draws.shape[0]
    Nthin, Npop, Nvar = Fields.thin_point.shape
    Nupdate, Ncr = Fields.update_CR_weight.shape
    # Nthin -= skip
    good_chains = getattr(Fields, "good_chains", None)
    total_generations = getattr(Fields, "total_generations", Ngen)
    thinning = getattr(Fields, "thinning", 1)
    best_x = getattr(Fields, "best_x", None)
    best_logp = getattr(Fields, "best_logp", 0.0)
    best_gen = getattr(Fields, "best_gen", 0)
    portion = getattr(Fields, "portion", 1.0)

    # Create empty draw and fill it with loaded data
    state = MCMCDraw(0, 0, 0, 0, 0, 0, thinning, portion=portion)
    state.draws = Ngen * Npop
    state.labels = [label.decode() for label in Fields.labels]
    state.generation = Ngen
    # print(f"loading {total_generations=} {Ngen=}")
    state._gen_offset = total_generations - Ngen
    state._gen_index = 0
    state._gen_draws = Fields.gen_draws.astype(int)
    state._gen_acceptance_rate = Fields.AR
    state._gen_logp = Fields.gen_logp
    state._thin_count = Nthin
    state._thin_index = 0
    state._thin_draws = Fields.thin_draws.astype(int)
    state._thin_logp = Fields.thin_logp
    state._thin_point = Fields.thin_point
    state._gen_current = state._thin_point[-1].copy()
    state._update_count = Nupdate
    state._update_index = 0
    state._update_draws = Fields.update_draws.astype(int)
    state._update_CR_weight = Fields.update_CR_weight
    state._outliers = []

    if best_x is not None:
        state._best_x = best_x
        state._best_logp = best_logp
        state._best_gen = best_gen
    else:
        bestidx = np.unravel_index(np.argmax(Fields.thin_logp), Fields.thin_logp.shape)
        state._best_logp = Fields.thin_logp[bestidx]
        state._best_x = Fields.thin_point[bestidx]
        # We are not multiplying the following by thinning because thinning and best_gen
        # were added at the same time, so the reloaded thinning=1. Even if it were not
        # one this could still be wrong since thinning may have changed on resume.
        state._best_gen = bestidx[0]

    state._good_chains = slice(None, None) if good_chains is None else good_chains.astype(int)

    return state


def save_state(state: "MCMCDraw", filename: str):
    # import sys; trace = sys.stdout
    # trace = open(filename+"-trace.mc", "wt")
    # trace.write("starting trace\n")

    # Build 2-D data structures
    # trace.write("extracting draws, logp\n")
    draws, logp = state.logp(full=True)
    # trace.write("extracting acceptance rate\n")
    _, AR = state.acceptance_rate()
    # trace.write("building chain from draws, AR and logp\n")
    chain = hstack((draws[:, None], AR[:, None], logp))

    # trace.write("extracting point, logp\n")
    _, point, logp = state.chains()
    Nthin, Npop, Nvar = point.shape
    # trace.write("shape is %d,%d,%d\n" % (Nthin, Npop, Nvar))
    # trace.write("adding logp to point\n")
    point = dstack((logp[:, :, None], point))
    # trace.write("collapsing to draws x point\n")
    point = reshape(point, (point.shape[0] * point.shape[1], point.shape[2]))

    # trace.write("extracting CR_weight\n")
    draws, CR_weight = state.CR_weight()
    Nupdate, Ncr = CR_weight.shape
    # trace.write("building stats\n")
    stats = hstack((draws[:, None], CR_weight))

    # TODO: missing _outliers from save_state

    # Write convergence info
    # trace.write("writing chain\n")
    fid = CREATE(filename + "-chain" + EXT, "wb")
    fid.write(f"# draws acceptance_rate {Npop}*logp\n".encode())
    savetxt(fid, chain)
    fid.close()

    # Write point info
    # trace.write("writing point\n")
    fid = CREATE(filename + "-point" + EXT, "wb")
    fid.write(f"# logp point (Nthin x Npop x Nvar = [{Nthin},{Npop},{Nvar}])\n".encode())
    savetxt(fid, point)
    fid.close()

    # Write stats
    # trace.write("writing stats\n")
    fid = CREATE(filename + "-stats" + EXT, "wb")
    fid.write(f"# draws {Ncr}*CR_weight\n".encode())
    savetxt(fid, stats)
    fid.close()
    # trace.write("done state save\n")
    # trace.close()


IND_PAT = re.compile("-1#IND")
INF_PAT = re.compile("1#INF")


def loadtxt(file, report=0):
    """
    Like numpy loadtxt, but adapted for windows non-finite numbers.
    """
    if not hasattr(file, "readline"):
        if file.endswith(".gz"):
            # print("opening with gzip")
            fh = gzip.open(file, "rt")
        else:
            fh = open(file, "rt")
    else:
        fh = file
    res = []
    section = 0
    lineno = 0
    for line in fh:
        lineno += 1
        if report and lineno % report == 0:
            print("read", section * report)
            section += 1
        IND_PAT.sub("nan", line)
        INF_PAT.sub("inf", line)
        line = line.split("#")[0].strip()
        values = line.split()
        if len(values) > 0:
            try:
                res.append([float(v) for v in values])
            except ValueError:
                print("Parse error:", values)
    if fh != file:
        fh.close()
    return asarray(res)


def path_contains_saved_state(filename):
    chain_file = filename + "-chain" + EXT
    return os.path.exists(chain_file)


def openmc(filename):
    if filename.endswith(".gz"):
        if os.path.exists(filename):
            # print("opening with gzip")
            fh = gzip.open(filename, "rt")
        elif os.path.exists(filename[:-3]):
            fh = open(filename[:-3], "rt")
        else:
            raise RuntimeError("file %s does not exist" % filename)
    else:
        if os.path.exists(filename):
            fh = open(filename, "rt")
        elif os.path.exists(filename + ".gz"):
            # print("opening with gzip")
            fh = gzip.open(filename + ".gz", "rt")
        else:
            raise RuntimeError("file %s does not exist" % filename)
    return fh


def load_state(filename, skip=0, report=0, derived_vars=0):
    """
    *filename* is the path to the saved MCMC state up to the final -chain.mc, etc.

    *derived_vars* is the number of columns added to each point, derived
    from other columns in that point. The newer set_derived_vars interface generates
    the derived variables on demand rather than storing them in the state object and
    so it will be zero always.
    """
    # Read chain file
    with openmc(filename + "-chain" + EXT) as fid:
        chain = loadtxt(fid)

    # Read point file
    with openmc(filename + "-point" + EXT) as fid:
        line = fid.readline()
        point_dims = line[line.find("[") + 1 : line.find("]")]
        Nthin, Npop, Nvar = eval(point_dims)
        for _ in range(skip * Npop):
            fid.readline()
        point = loadtxt(fid, report=report * Npop)

    # Read stats file
    with openmc(filename + "-stats" + EXT) as fd:
        stats_header = fd.readline()
        stats = loadtxt(fd)

    # Determine number of R-stat stored in the stats file
    if "R-stat" in stats_header:
        # Old header looks like:
        #     # draws {Nvar}*R-stat {Ncr}*CR_weight
        # however, number of R-stat stored in stats file is the number of
        # variables stored each generation, not including the derived variables
        # calculated after the MCMC has completed.
        num_r = int(stats_header.split("*")[0].split()[-1]) - derived_vars
    else:
        num_r = 0

    # Guess dimensions
    Ngen = chain.shape[0]
    thinning = 1
    Nthin -= skip
    Nupdate = stats.shape[0]

    # Create empty draw and fill it with loaded data
    state = MCMCDraw(0, 0, 0, 0, 0, 0, thinning)
    # print("gen, var, pop", Ngen, Nvar, Npop)
    state.draws = Ngen * Npop
    state.generation = Ngen
    state._gen_offset = 0
    state._gen_index = 0
    state._gen_draws = chain[:, 0]
    state._gen_acceptance_rate = chain[:, 1]
    state._gen_logp = chain[:, 2:]
    state._thin_count = Ngen // thinning
    state._thin_index = 0
    state._thin_draws = state._gen_draws[(skip + 1) * thinning - 1 :: thinning]
    state._thin_logp = point[:, 0].reshape((Nthin, Npop))
    state._thin_point = reshape(point[:, 1 : Nvar + 1 - derived_vars], (Nthin, Npop, -1))
    state._gen_current = state._thin_point[-1].copy()
    state._update_count = Nupdate
    state._update_index = 0
    state._update_draws = stats[:, 0]
    state._update_CR_weight = stats[:, 1 + num_r :]
    state._outliers = []

    bestidx = np.argmax(point[:, 0])
    state._best_logp = point[bestidx, 0]
    state._best_x = point[bestidx, 1 : Nvar + 1 - derived_vars]
    state._best_gen = 0

    return state


class MCMCDraw(object):
    """ """

    _derived_fn: Callable[[NDArray], NDArray] = None
    _derived_labels: List[str] = None
    _labels = None
    _integer_vars = None  # boolean array of integer variables, or None
    title = None

    def __init__(
        self, Ngen: int, Nthin: int, Nupdate: int, Nvar: int, Npop: int, Ncr: int, thinning: int, portion: float = 1.0
    ):
        # self.generation and self.draws are used to control the number of iterations in
        # the dream loop, and must therefore be set to the current size of the state on
        # resume rather than the cumulative across all runs. To retrieve the totals use
        # generations=self.total_generations and draws=self.total_generations*self.Npop.

        # Total number of draws so far
        self.draws = 0

        # Portion: a fraction of the chain to use for statistical plots.  A value can be
        # automatically determined by calling self.trim_portion(), or supplied by the
        # user interactively.  The value should be between 0.0 and 1.0, where 1.0
        # means the entire chain is used for statistical plots.
        self.portion = portion

        # Maximum observed likelihood
        # Note: with thinning and burn the best may not be in the set of samples
        self._best_x = None
        self._best_logp = -inf
        self._best_gen = 0

        # Per generation iteration
        self.generation = 0
        self._gen_offset = 0
        self._gen_index = 0
        self._gen_draws = empty(Ngen, "i")
        self._gen_logp = empty((Ngen, Npop))
        self._gen_acceptance_rate = empty(Ngen)

        # If we are thinning, we need to keep the current generation
        # separately. [Note: don't remember why we need both the _gen_*
        # and _thin_*]  [Note: the caller x vector is assigned to
        # _gen_current; this may lead to unexpected behaviour if x is
        # changed by the caller.
        self._gen_current = None

        # Per thinned generation iteration
        self.thinning = thinning
        self._thin_index = 0
        self._thin_count = 0
        self._thin_timer = 0
        self._thin_draws = empty(Nthin, "i")
        self._thin_point = empty((Nthin, Npop, Nvar))
        self._thin_logp = empty((Nthin, Npop))

        # Per update iteration
        self._update_index = 0
        self._update_count = 0
        self._update_draws = empty(Nupdate, "i")
        self._update_CR_weight = empty((Nupdate, Ncr))

        self._outliers = []

        # Query functions will not return outlier chains; initially, all
        # chains are marked as good.  Call mark_outliers to remove
        # outlier chains from the set.
        self._good_chains = slice(None, None)

    @property
    def Ngen(self):
        return self._gen_draws.shape[0]

    @property
    def total_generations(self):
        return self._gen_offset + self.generation

    @property
    def Nsamples(self):
        return self._gen_logp.size

    @property
    def Nthin(self):
        return self._thin_draws.shape[0]

    @property
    def Nupdate(self):
        return self._update_draws.shape[0]

    @property
    def Nvar(self):
        """Number of parameters in the fit"""
        return self._thin_point.shape[2]

    @property
    def Npop(self):
        return self._gen_logp.shape[1]

    @property
    def Ncr(self):
        return self._update_CR_weight.shape[1]

    def resize(self, Ngen: int, Nthin: int, Nupdate: int, Nvar: int, Npop: int, Ncr: int, thinning: int):
        if self.Nvar != Nvar or self.Npop != Npop or self.Ncr != Ncr:
            raise ValueError("Cannot change Nvar, Npop or Ncr on resize")

        # print("resize")
        # print(self.generation, self.Ngen, Ngen)
        # print(self._update_count, self.Nupdate, Nupdate)
        # print(self._thin_count, self.Nthin, Nthin)

        # When resuming from live state, make sure we unroll the state before resizing.
        # This is probably a no-op since plotting will already have forced an unroll.
        self._unroll()

        self.thinning = thinning

        def buf_resize(v, new_size, position):
            """
            Resize a circular buffer to *new_size*.

            *position* is the number of entries that have been added to the buffer. This
            may be less than the existing size if the buffer is not yet filled with one
            complete cycle. Assume the buffer has been unrolled so that the next location
            index is zero. Don't assume that the next location index is *position%size*.

            When extending, we add space to the end of the buffer and set the next index
            after the previous end of the buffer. When contracting, we keep the last N entries
            in the buffer. If there are fewer than N entries, keep all of them and include
            enough empty entries to make the buffer size N.

            Need to set the next index to min(position, old_size, new_size)%new_size
            """
            if v.shape[0] < new_size:  # expand
                v = np.resize(v, (new_size, *v.shape[1:]))
            elif position < new_size:  # contract when not enough
                v = v[:new_size]
            else:  # contract when have enough
                v = v[-new_size:]
            return v

        self._gen_offset += max(self.generation - Ngen, 0)
        self.generation = min(self.generation, Ngen, self.Ngen)
        self.draws = self.generation * self.Npop
        self._gen_index = 0
        # Reset sample size to current size on resume.
        self.generation = min(self.generation, self.Ngen, Ngen)
        self.draws = self.generation * self.Npop
        self._gen_index = self.generation % Ngen
        self._gen_draws = buf_resize(self._gen_draws, Ngen, self.generation)
        self._gen_logp = buf_resize(self._gen_logp, Ngen, self.generation)
        self._gen_acceptance_rate = buf_resize(self._gen_acceptance_rate, Ngen, self.generation)

        self._thin_count = min(self._thin_count, self.Nthin, Nthin)
        self._thin_index = self._thin_count % Nthin
        self._thin_draws = buf_resize(self._thin_draws, Nthin, self._thin_count)
        self._thin_point = buf_resize(self._thin_point, Nthin, self._thin_count)
        self._thin_logp = buf_resize(self._thin_logp, Nthin, self._thin_count)

        self._update_count = min(self._update_count, self.Nupdate, Nupdate)
        self._update_index = self._update_count % Nupdate
        self._update_draws = buf_resize(self._update_draws, Nupdate, self._update_count)
        self._update_CR_weight = buf_resize(self._update_CR_weight, Nupdate, self._update_count)

    def save(self, filename: Union[Path, str]):
        save_state(self, filename)

    def trim_portion(self):
        index = burn_point(self)
        portion = 1 - (index / self.Ngen) if index >= 0 else 0.5
        return portion

    def show(self, portion: Optional[float] = None, figfile: Union[str, Path, None] = None):
        from .views import plot_all

        plot_all(self, portion=portion, figfile=figfile)

    def _last_gen(self):
        """
        Returns x, logp for most recent generation to dream.py.
        """
        # Note: if generation number has wrapped and _gen_index is 0
        # (the usual case when this function is called to resume an
        # existing chain), then this returns the last row in the array.
        return (self._thin_point[self._thin_index - 1], self._thin_logp[self._thin_index - 1])

    def _generation(self, new_draws: int, x: NDArray, logp: NDArray, accept: NDArray, force_keep: bool = False):
        """
        Called from dream.py after each generation is completed with
        a set of accepted points and their values.
        """
        # Keep track of the total number of draws
        # Note: this is first so that we tag the record with the number of
        # draws taken so far, including the current draw.
        self.draws += new_draws
        self.generation += 1

        # Record if this is the best so far
        maxid = argmax(logp)
        if logp[maxid] > self._best_logp:
            self._best_logp = logp[maxid]
            self._best_x = x[maxid, :] + 0  # Force a copy
            self._best_gen = self.total_generations
            # print("new best", logp[maxid], self.generation)

        # Record acceptance rate and cost
        i = self._gen_index
        # print("generation", i, self.draws, "\n x", x, "\n logp", logp, "\n accept", accept)
        self._gen_draws[i] = self.draws
        self._gen_acceptance_rate[i] = 100 * sum(accept) / new_draws
        self._gen_logp[i] = logp
        i = i + 1
        if i == len(self._gen_draws):
            i = 0
        self._gen_index = i

        # Keep every nth iteration
        self._thin_timer += 1
        if self._thin_timer == self.thinning or force_keep:
            self._thin_timer = 0
            self._thin_count += 1
            i = self._thin_index
            self._thin_draws[i] = self.draws
            self._thin_point[i] = x
            self._thin_logp[i] = logp
            i = i + 1
            if i == len(self._thin_draws):
                i = 0
            self._thin_index = i
            self._gen_current = x + 0  # force a copy
        else:
            self._gen_current = x + 0  # force a copy

    def _update(self, CR_weight: NDArray):
        """
        Called from dream.py when a series of DE steps is completed and
        summary statistics/adaptations are ready to be stored.
        """
        self._update_count += 1
        i = self._update_index
        # print("update", i, self.draws, "\n CR weight", CR_weight)
        self._update_draws[i] = self.draws
        self._update_CR_weight[i] = CR_weight
        i = i + 1
        if i == len(self._update_draws):
            i = 0
        self._update_index = i

    @property
    def labels(self):
        if self._labels is None:
            result = ["P%d" % i for i in range(self._thin_point.shape[2])]
        else:
            result = self._labels
        if self._derived_labels:
            result = [*result, *self._derived_labels]
        return result

    @labels.setter
    def labels(self, v: List[str]):
        self._labels = v

    def _draw_pop(self):
        """
        Return the current population.
        """
        return self._gen_current

    def _draw_large_pop(self, Npop: int):
        _, chains, _ = self.chains()
        Ngen, Nchain, Nvar = chains.shape
        points = reshape(chains, (Ngen * Nchain, Nvar))

        # There are two complications with the history buffer:
        # (1) due to thinning, not every generation is stored
        # (2) because it is circular, the cursor may be in the middle
        # If the current generation isn't in the buffer (but is instead
        # stored separately as _gen_current), then the entire buffer
        # becomes the history pool.
        # otherwise we need to exclude the current generation from
        # the pool.  If (2) happens, we need to increment everything
        # above the cursor by the number of chains.
        if self._gen_current is not None:
            pool_size = Ngen * Nchain
            cursor = pool_size  # infinite
        else:
            pool_size = (Ngen - 1) * Nchain
            k = len(self._thin_draws)
            cursor = Nchain * ((k + self._thin_index - 1) % k)

        # Make a return population and fill it with the current generation
        pop = empty((Npop, Nvar), "d")
        if self._gen_current is not None:
            pop[:Nchain] = self._gen_current
        else:
            # print(pop.shape, points.shape, chains.shape)
            pop[:Nchain] = points[cursor : cursor + Nchain]

        if Npop > Nchain:
            # Find the remainder with unique ancestors.
            # Again, because this is a circular buffer, their may be random
            # numbers generated at or above the cursor.  All of these must
            # be shifted by Nchains to avoid the cursor.
            perm = draw(Npop - Nchain, pool_size)
            perm[perm >= cursor] += Nchain
            # print("perm", perm; raw_input('wait'))
            pop[Nchain:] = points[perm]

        return pop

    def _unroll(self):
        """
        Unroll the circular queue so that data access can be done inplace.

        Call this when done stepping, and before plotting.  Calls to
        logp, sample, etc. assume the data is already unrolled.
        """
        if self.generation > self._gen_index > 0:
            self._gen_draws[:] = np.roll(self._gen_draws, -self._gen_index, axis=0)
            self._gen_logp[:] = np.roll(self._gen_logp, -self._gen_index, axis=0)
            self._gen_acceptance_rate[:] = np.roll(self._gen_acceptance_rate, -self._gen_index, axis=0)
            self._gen_index = 0

        if self._thin_count > self._thin_index > 0:
            self._thin_draws[:] = np.roll(self._thin_draws, -self._thin_index, axis=0)
            self._thin_point[:] = np.roll(self._thin_point, -self._thin_index, axis=0)
            self._thin_logp[:] = np.roll(self._thin_logp, -self._thin_index, axis=0)
            self._thin_index = 0

        if self._update_count > self._update_index > 0:
            self._update_draws[:] = np.roll(self._update_draws, -self._update_index, axis=0)
            self._update_CR_weight[:] = np.roll(self._update_CR_weight, -self._update_index, axis=0)
            self._update_index = 0

    def remove_outliers(self, x: NDArray, logp: NDArray, test: str = "IQR"):
        """
        Replace outlier chains with clones of good ones.  This should happen
        early in the sampling processes so the clones have an opportunity
        to evolve their own identity.  Only the head of the chain is modified.

        *state* contains the chains, with log likelihood for each point.

        *x*, *logp* are the current population and the corresponding
        log likelihoods; these are updated with cloned chain values.

        *test* is the name of the test to use (one of IQR, Grubbs, Mahal
        or none). See :func:`.outliers.identify_outliers` for details.

        Updates *state*, *x* and *logp* to reflect the changes.

        Returns a list of the outliers that were removed.
        """
        # Grab the last part of the chain histories
        _, chains = self.logp()
        chain_len, Nchains = chains.shape
        outliers = identify_outliers(test, chains, x)
        # if len(outliers): print("old llf", logp[outliers])

        # Loop over each outlier chain, replacing each with another
        for old in outliers:
            # Draw another chain at random, with replacement
            # TODO: consider using relative likelihood as a weight factor
            while True:
                new = rng.randint(Nchains)
                if new not in outliers:
                    break
            # Update the saved state and current population
            self._replace_outlier(old=old, new=new)
            x[old, :] = x[new, :]
            logp[old] = logp[new]

        # if len(outliers): print("new llf", logp[outliers])
        return outliers

    def _replace_outlier(self, old: int, new: int):
        """
        Called from outliers.py when a chain is replaced by the
        clone of another.
        """
        self._outliers.append((self._thin_index, old, new))

        # 2017-10-06 [PAK] only replace the head, not the full chain
        index = self._gen_index
        self._gen_current[old] = self._gen_current[new]
        self._gen_logp[index, old] = self._gen_logp[index, new]
        self._thin_logp[index, old] = self._thin_logp[index, new]
        self._thin_point[index, old, :] = self._thin_point[index, new, :]

    def mark_outliers(self, test: str = "IQR", portion: Optional[float] = None):
        """
        Mark some chains as outliers but don't remove them.  This can happen
        after drawing is complete, so that chains that did not converge are
        not included in the statistics.

        *test* is 'IQR', 'Mahol' or 'none'.

        *portion* indicates what portion of the samples should be included
        in the outlier test.  If None, then the stored portion is used.
        """
        _, chains, logp = self.chains()

        if test == "none":
            self._good_chains = slice(None, None)
        else:
            Ngen = chains.shape[0]
            portion = self.portion if portion is None else portion
            start = int(Ngen * (1 - portion))
            outliers = identify_outliers(test, logp[start:], chains[-1])
            # print("outliers", outliers)
            # print(logp.shape, chains.shape)
            if len(outliers) > 0:
                self._good_chains = np.array([i for i in range(logp.shape[1]) if i not in outliers])
            else:
                self._good_chains = slice(None, None)
            # print(self._good_chains)

    def logp(self, full: bool = False):
        """
        Return the iteration number and the log likelihood for each point in
        the individual sequences in that iteration.

        For example, to plot the convergence of each sequence::

            draw, logp = state.logp()
            plot(draw, logp)

        Note that draw[i] represents the total number of samples taken,
        including those for the samples in logp[i].

        If full is True, then return all chains, not just good chains.
        """
        # self._unroll()
        # draws, logp = self._gen_draws, self._gen_logp
        # if self.generation == self._gen_index:
        #    draws, logp = [v[:self.generation] for v in (draws, logp)]

        # Don't do a full unroll here
        if self.generation == self._gen_index:
            draws = self._gen_draws[: self._gen_index]
            logp = self._gen_logp[: self._gen_index]
        elif self._gen_index > 0:
            draws = np.roll(self._gen_draws, -self._gen_index, axis=0)
            logp = np.roll(self._gen_logp, -self._gen_index, axis=0)
        else:
            draws = self._gen_draws
            logp = self._gen_logp

        # TODO: just return logp, not logp and draws
        return draws, (logp if full else logp[:, self._good_chains])

    def logp_slice(self, n: int):
        """
        Return a slice of the logp chains, either the first n if n > 0
        or the last n if n < 0.  Avoids unrolling the circular buffer if
        possible.
        """
        if n < 0:  # tail
            if self._gen_index >= -n:
                return self._gen_logp[self._gen_index + n : self._gen_index]
            elif self._gen_index == 0:
                return self._gen_logp[n:]
            else:  # unroll across boundary
                return np.vstack((self._gen_logp[n + self._gen_index :], self._gen_logp[: self._gen_index]))
        else:  # head
            if self.generation < self.Ngen:
                return self._gen_logp[:n]
            elif self._gen_index + n <= self.Ngen:
                return self._gen_logp[self._gen_index : self._gen_index + n]
            else:
                return np.vstack((self._gen_logp[self._gen_index :], self._gen_logp[-n + self._gen_index :]))

    def min_slice(self, n: int):
        """
        Return the minimum logp for n slices, from the head if positive
        or the tail if negative.

        This is a specialized function so it can be fast.  Convergence
        can be quickly rejected if the min in a short head is smaller
        than the min in a long tail.  Unfortunately, if the data is
        wrapped, then the max function will cost extra.
        """
        # Copy the logic of slice
        if n < 0:  # tail
            if self._gen_index >= -n:
                return np.min(self._gen_logp[self._gen_index + n : self._gen_index])
            elif self._gen_index == 0:
                return np.min(self._gen_logp[n:])
            else:  # max across boundary
                return min(np.min(self._gen_logp[n + self._gen_index :]), np.min(self._gen_logp[: self._gen_index]))
        else:  # head
            if self.generation < self.Ngen:
                return np.min(self._gen_logp[:n])
            elif self._gen_index + n <= self.Ngen:
                return np.min(self._gen_logp[self._gen_index : self._gen_index + n])
            else:
                return min(np.min(self._gen_logp[self._gen_index :]), np.min(self._gen_logp[-n + self._gen_index :]))

    def acceptance_rate(self):
        """
        Return the iteration number and the acceptance rate for that iteration.

        For example, to plot the acceptance rate over time::

            draw, AR = state.acceptance_rate()
            plot(draw, AR)

        """
        retval = self._gen_draws, self._gen_acceptance_rate
        if self.generation == self._gen_index:
            retval = [v[: self.generation] for v in retval]
        elif self._gen_index > 0:
            retval = [np.roll(v, -self._gen_index, axis=0) for v in retval]
        return retval

    def chains(self):
        """
        Returns the observed Markov chains and the corresponding likelihoods.

        The return value is a tuple (*draws*, *chains*, *logp*).

        *draws* is the number of samples taken up to and including the samples
        for the current generation.

        *chains* is a three dimensional array of generations X chains X vars
        giving the set of points observed for each chain in every generation.
        Only the thinned samples are returned.

        *logp* is a two dimensional array of generation X population giving
        the log likelihood of observing the set of variable values given in
        chains.
        """
        self._unroll()
        retval = self._thin_draws, self._thin_point, self._thin_logp
        if self._thin_count == self._thin_index:
            retval = [v[: self._thin_count] for v in retval]
        return retval

    def gelman(self, portion: Optional[float] = None):
        """
        Compute the Gelman and Rubin R-statistic for the Markov chains.
        """
        portion = self.portion if portion is None else portion
        if self.generation < self.Ngen:
            return gelman(self._thin_point[: self.generation], portion=portion)
        else:
            return gelman(self._thin_point, portion=portion)

    def CR_weight(self):
        """
        Return the crossover ratio weights to be used in the next generation.

        For example, to see if the adaptive CR is stable use::

            draw, weight = state.CR_weight()
            plot(draw, weight)

        See :mod:`.crossover` for details.
        """
        self._unroll()
        retval = self._update_draws, self._update_CR_weight
        if self._update_count == self._update_index:
            retval = [v[: self._update_count] for v in retval]
        return retval

    def outliers(self):
        """
        Return a list of outlier removal operations.

        Each outlier operation is a tuple giving the thinned generation
        in which it occurred, the old chain id and the new chain id.

        The chains themselves have already been updated to reflect the
        removal.

        Curiously, it is possible for the maximum likelihood seen so far
        to be removed by this operation.
        """
        return asarray(self._outliers, "i")

    def best(self):
        """
        Return the best point seen and its log likelihood.
        """
        return self._best_x, self._best_logp

    def stable_best(self):
        """
        Return True if the at least one full cycle of the circular
        buffer has passed since the best logp was first observed.
        """
        # print(f"stable_best {self._best_gen} + {self.Ngen} <= {self.total_generations}")
        return self._best_gen + self.Ngen <= self.total_generations

    def keep_best(self):
        """
        Place the best point at the end of the last good chain.

        Good chains are defined by mark_outliers.

        Because the Markov chain is designed to wander the parameter
        space, the best individual seen during the random walk may have
        been observed during the burn-in period, and may no longer be
        present in the chain.  If this is the case, replace the final
        point with the best, otherwise swap the positions of the final
        and the best.
        """

        # Get state as a 1D array
        _, chains, logp = self.chains()
        Ngen, Npop, Nvar = chains.shape
        points = reshape(chains, (Ngen * Npop, Nvar))
        logp = reshape(logp, Ngen * Npop)

        # Set the final position to the end of the last good chain.  If
        # mark_outliers has not been called, then _good_chains will
        # just be slice(None, None)
        if isinstance(self._good_chains, slice):
            final = -1
        else:
            final = self._good_chains[-1] - Npop

        # Find the location of the best point if it exists and swap with
        # the final position
        idx = np.where(logp == self._best_logp)[0]
        if len(idx) == 0:
            logp[final] = self._best_logp
            points[final, :] = self._best_x
        else:
            idx = idx[0]
            logp[final], logp[idx] = logp[idx], logp[final]
            points[final, :], points[idx, :] = points[idx, :], points[final, :]
        # For multiple minima, arbitrarily choose one of them
        # TODO: this will lead to possible confusion when the best value
        # spontaneously changes when the fit is complete.
        self._best_p = points[final]
        self._best_logp = logp[final]

    def sample(self, **kw):
        """
        Return a sample from the posterior distribution.

        **Deprecated** use :meth:`draw` instead.
        """
        drawn = self.draw(**kw)
        return drawn.points, drawn.logp

    def entropy(
        self,
        vars: Optional[List[int]] = None,
        portion: Optional[float] = None,
        selection: SelectionType = None,
        n_est: int = 10000,
        thin: Optional[int] = None,
        method: Optional[str] = None,
    ):
        r"""
        Return entropy estimate and uncertainty from an MCMC draw.

        *portion* is the portion of each chain to use (uses self.portion if None).

        *vars* is the set of variables to marginalize over.  It is None for
        the visible variables, or a list of variables.

        *vars* is the list of variables to use for marginalization.

        *selection* sets the range each parameter in the returned distribution,
        using {variable: (low, high)}. Missing variables use the full range.

        *n_est* is the number of points to use from the draw when estimating
        the entropy (default=10000).

        *thin* is the amount of thinning to use when selecting points from the
        draw.

        *method* determines which entropy calculation to use:

        * gmm: fit sample to a gaussian mixture model (GMM) with $5 \sqrt{d}$
          components where $d$ is the number fitted parameters and estimate
          entropy by sampling from the GMM.

        * llf: estimates likelihood scale factor from ratio of density
          estimate to model likelihood, then computes Monte Carlo entropy
          from sample; this does not work for marginal likelihood estimates.
          DOI:10.1109/CCA.2010.5611198

        * mvn: fit sample to a multi-variate Gaussian and return the entropy
          of the best fit gaussian; uses bootstrap to estimate uncertainty.

        * wnn: estimate entropy from nearest-neighbor distances in sample.
          DOI:10.1214/18-AOS1688
        """
        from . import entropy

        # Get the sample from the state.
        # set default thinning to max((steps * samples/step) // n_est, 1)
        if thin is None:
            Nsteps = min(self.Nthin, self._thin_count)
            thin = max(Nsteps * self.Npop // n_est, 1)
            # print("thin", thin, Nsteps, self.Npop, self.Nthin, self._thin_count)
        drawn = self.draw(portion=portion, vars=vars, selection=selection, thin=thin)

        # TODO: don't print within a library function!
        M = entropy.MVNEntropy(drawn.points)
        print("Entropy from MVN: %s" % str(M))

        if method is None:
            # TODO: change default to gmm
            method = "llf"

        if method == "llf":
            S, Serr = entropy.entropy(drawn.points, drawn.logp, N_entropy=n_est)
            # print("Entropy from llf (Kramer): %s"%str(S))
        elif method == "gmm":
            # Try pure gmm ... pretty good
            S, Serr = entropy.gmm_entropy(drawn.points, n_est=n_est)
            # print("Entropy from gmm: %g +/- %g"% (S, Serr))
        elif method == "wnn":
            # Try pure wnn ... no good
            S, Serr = entropy.wnn_entropy(drawn.points, n_est=n_est)
            # print("Entropy from wnn: %s"%str(S))
        elif method == "mvn":
            S, Serr = entropy.mvn_entropy_bootstrap(drawn.points)
            # print("Entropy from mvn: %s"%str(S))
        else:
            raise ValueError("unknown method %r" % method)

        # Always return entropy estimate from draw, even if it is normal
        return S, Serr

    def draw(
        self,
        portion: Optional[float] = None,
        vars: Optional[List[int]] = None,
        selection: SelectionType = None,
        thin: int = 1,
    ):
        """
        Return a sample from the posterior distribution.

        *portion* is the portion of each chain to use

        *vars* is a list of variables to return for each point

        *selection* sets the range each parameter in the returned distribution,
        using {variable: (low, high)}. Missing variables use the full range.

        *thin* takes every nth item.

        To plot the distribution for parameter p1::

            draw = state.draw()
            hist(draw.points[:, 0])

        To plot the interdependence of p1 and p2::

            draw = state.sample()
            plot(draw.points[:, 0], draw.points[:, 1], '.')
        """
        vars = vars if vars is not None else getattr(self, "_shown", None)
        portion = self.portion if portion is None else portion
        return Draw(self, portion=portion, vars=vars, selection=selection, thin=thin)

    # TODO: Move processing of visible/integer/derived out of state
    def set_visible_vars(self, labels: List[str]):
        self._shown = [self.labels.index(v) for v in labels]
        # print("\n".join(str(pair) for pair in enumerate(self.labels)))
        # print(labels)
        # print(self._shown)

    def set_integer_vars(self, labels: List[str]):
        """
        Indicate tha variables should be considered integer variables when
        computing statistics.
        """
        self._integer_vars = np.array([var in labels for var in self.labels])

    def set_derived_vars(self, fn: Callable[[NDArray], NDArray], labels: List[str]):
        """
        Define derived variables from the sample. When calling draw() it will add
        columns for the derived variables to each sample.

        *fn* is a function taking points p[:, k] for k in 0 ... samples and
        returning a set of derived variables pj[k] for each sample k.  The
        variables can be returned as any kind of sequence including an
        array or a tuple with one entry per variable.  The caller uses
        asarray to convert the returned variables into a vars X samples array.
        For convenience, a single variable can be returned by itself.

        *labels* are the labels to use for the derived variables.

        The following example adds the new variable x+y = P[0] + P[1]::

            state.derive_vars(lambda p: p[0]+p[1], labels=["x+y"])
        """
        self._derived_fn = fn
        self._derived_labels = labels

    def derive_vars(self, fn: Callable[[NDArray], NDArray], labels: Optional[List[str]] = None):
        """
        *** DEPRECATED ***

        Like set_derived_vars but operating in place, modifying the points in the history.
        """
        # Grab all samples as a set of points
        _, chains, _ = self.chains()
        Ngen, Npop, Nvar = chains.shape
        points = reshape(chains, (Ngen * Npop, Nvar))

        # Compute new variables from the points
        newvars = asarray(fn(points.T)).T
        Nnew = newvars.shape[1] if len(newvars.shape) == 2 else 1
        newvars.reshape((Ngen, Npop, Nnew))

        # Extend new variables to be the same length as the stored selection
        Nthin = self._thin_point.shape[0]
        newvars = np.resize(newvars, (Nthin, Npop, Nnew))

        # Add new variables to the points
        self._thin_point = dstack((self._thin_point, newvars))

        # Add labels for the new variables, if available.
        if labels is not None:
            self.labels = self.labels + labels
        elif self._labels is not None:
            labels = ["P%d" % i for i in range(Nvar, Nvar + Nnew)]
            self.labels = self.labels + labels
        else:  # no labels specified, old or new
            pass


class Draw:
    state: MCMCDraw
    vars: Optional[List[int]]
    portion: float
    selection: SelectionType
    thin: int

    def __init__(
        self,
        state: MCMCDraw,
        vars: Optional[List[int]] = None,
        portion: float = 1.0,
        selection: SelectionType = None,
        thin: int = 1,
    ):
        self.state = state
        self.vars = vars
        self.portion = portion
        self.selection = selection
        self.points, self.logp = _sample(state, portion=portion, selection=selection, thin=thin)
        # Derived variables are created during the draw so that existing data isn't altered.
        # This allows resume to work without extra effort. The code is also much simpler.
        if state._derived_fn is not None:
            newvars = asarray(state._derived_fn(self.points.T)).T
            self.points = np.vstack((self.points, newvars))
        if vars is not None:
            self.points = self.points[:, vars]
        self.labels = state.labels if vars is None else [state.labels[v] for v in vars]

        self.weights = None
        self.num_vars = len(self.labels)
        if state._integer_vars is not None:
            self.integers = state._integer_vars[vars] if vars else None
        else:
            self.integers = None
        self._argsort_indices = {}

    # cache the argsort indices for each variable
    def get_argsort_indices(self, var: int):
        if var not in self._argsort_indices:
            self._argsort_indices[var] = np.argsort(self.points[:, var].flatten())
        return self._argsort_indices[var]


def _sample(state, portion: float, selection: SelectionType, thin):
    """
    Return a sample from a set of chains.
    """
    draw, chains, logp = state.chains()
    start = int((1 - portion) * len(draw))

    # Collect the subset we are interested in
    chains = chains[start::thin, state._good_chains, :]
    logp = logp[start::thin, state._good_chains]

    Ngen, Npop, Nvar = chains.shape
    points = reshape(chains, (-1, Nvar))
    logp = reshape(logp, (-1))
    if selection not in [None, {}]:
        idx = True
        for v, r in selection.items():
            if v == "logp":
                idx = idx & (logp >= r[0]) & (logp <= r[1])
            else:
                idx = idx & (points[:, v] >= r[0]) & (points[:, v] <= r[1])
        points = points[idx, :]
        logp = logp[idx]
    return points, logp


def test():
    from numpy.linalg import norm
    from numpy.random import rand
    from numpy import arange

    # Make some fake data
    Nupdate, Nstep = 3, 5
    Ngen = Nupdate * Nstep
    Nvar, Npop, Ncr = 3, 6, 2
    xin = rand(Ngen, Npop, Nvar)
    pin = rand(Ngen, Npop)
    accept = rand(Ngen, Npop) < 0.8
    CRin = rand(Nupdate, Ncr)
    # thinning = 2
    # Nthin = int(Ngen/thinning)

    # Put it into a state
    thinning = 2
    Nthin = int(Ngen / thinning)
    state = MCMCDraw(Ngen=Ngen, Nthin=Nthin, Nupdate=Nupdate, Nvar=Nvar, Npop=Npop, Ncr=Ncr, thinning=thinning)
    for i in range(Nupdate):
        state._update(CR_weight=CRin[i])
        for j in range(Nstep):
            gen = i * Nstep + j
            state._generation(new_draws=Npop, x=xin[gen], logp=pin[gen], accept=accept[gen])

    # Check that it got there
    draws, logp = state.logp()
    assert norm(draws - Npop * arange(1, Ngen + 1)) == 0
    assert norm(logp - pin) == 0
    draws, AR = state.acceptance_rate()
    assert norm(draws - Npop * arange(1, Ngen + 1)) == 0
    assert norm(AR - 100 * sum(accept, axis=1) / Npop) == 0
    draws, logp = state.sample()
    # assert norm(draws - thinning*Npop*arange(1, Nthin+1)) == 0
    # assert norm(sample - xin[thinning-1::thinning]) == 0
    # assert norm(logp - pin[thinning-1::thinning]) == 0
    draws, CR = state.CR_weight()
    assert norm(draws - Npop * Nstep * arange(Nupdate)) == 0
    assert norm(CR - CRin) == 0
    x, p = state.best()
    bestid = argmax(pin)
    i, j = bestid // Npop, bestid % Npop
    assert pin[i, j] == p
    assert norm(xin[i, j, :] - x) == 0

    # Check that outlier updates properly
    state._replace_outlier(1, 2)
    outliers = state.outliers()
    draws, logp = state.sample()
    assert norm(outliers - asarray([[state._thin_index, 1, 2]])) == 0
    # assert norm(sample[:, 1, :] - xin[thinning-1::thinning, 2, :]) == 0
    # assert norm(sample[:, 2, :] - xin[thinning-1::thinning, 2, :]) == 0
    # assert norm(logp[:, 1] - pin[thinning-1::thinning, 2]) == 0
    # assert norm(logp[:, 2] - pin[thinning-1::thinning, 2]) == 0

    from .stats import var_stats, format_vars

    vstats = var_stats(state.draw())
    print(format_vars(vstats))


if __name__ == "__main__":
    test()
