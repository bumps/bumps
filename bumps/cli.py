"""
Bumps command line interface.

The functions in this module are used by the bumps command to implement
the command line interface.  Bumps plugin models can use them to create
stand alone applications with a similar interface.  For example, the
Refl1D application uses the following::

    from . import fitplugin
    from bumps.plugin import install_plugin
    from bumps.plotutil import set_mplconfig
    from bumps.cli import main as bumps_main
    set_mplconfig(appdatadir='Refl1D')
    install_plugin(fitplugin)
    bumps_main()

After completing a set of fits on related systems, a post-analysis script
can use :func:`load_model` to load the problem definition and
:func:`load_pars` to load the best value  found in the fit.  This can
be used for example in experiment design, where you look at the expected
parameter uncertainty when fitting simulated data from a range of experimental
systems.
"""

__all__ = [
    "main",
    "install_plugin",
    "set_mplconfig",
    "config_matplotlib",
    "load_model",
    "preview",
    "load_pars",
    "save_best",
    "resynth",
]

import sys
import os
import re
import warnings
import shutil
import traceback
from pathlib import Path

import numpy as np
# np.seterr(all="raise")

from .plotutil import config_matplotlib, set_mplconfig
from .plugin import install_plugin
from .fitproblem import load_pars
from .fitters import save_best
from .fitters import FitDriver, StepMonitor, ConsoleMonitor, CheckpointMonitor
from .mapper import MPMapper, MPIMapper, SerialMapper
from . import initpop
from . import __version__


def load_model(path: Path | str, model_options: list[str] | None = None):
    """
    *** DEPRECATED***. Use fitproblem.load_problem(path, [args=...]) instead.
    """
    from .fitproblem import load_problem

    problem = load_problem(path, args=model_options)
    # CRUFT: support old 'problem.options' attribute
    problem.options = problem.script_args
    return problem


def preview(problem, view=None):
    """
    Show the problem plots and parameters.
    """
    import matplotlib.pyplot as plt

    problem.show()
    problem.plot(view=view)
    plt.show()


# CRUFT
recall_best = load_best = load_pars


def store_overwrite_query_gui(path):
    """
    Ask if store path should be overwritten.

    Use this in a call to :func:`make_store` from a graphical user interface.
    """
    import wx

    msg = path + " already exists. Press 'yes' to overwrite, or 'No' to abort and restart with newpath"
    msg_dlg = wx.MessageDialog(None, msg, "Overwrite Directory", wx.YES_NO | wx.ICON_QUESTION)
    retCode = msg_dlg.ShowModal()
    msg_dlg.Destroy()
    if retCode != wx.ID_YES:
        raise RuntimeError("Could not create path")


def store_overwrite_query(path):
    """
    Ask if store path should be overwritten.

    Use this in a call to :func:`make_store` from a command line interface.
    """
    print(path, "already exists.")
    print("Press 'y' to overwrite, or 'n' to abort and restart with --overwrite, --resume, or --store=newpath")
    ans = input("Overwrite [y/n]? ")
    if ans not in ("y", "Y", "yes"):
        sys.exit(1)


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def make_store(problem, opts, exists_handler):
    """
    Create the store directory and populate it with the model definition file.
    """
    # Determine if command line override
    if opts.store:
        problem.store = opts.store
    if getattr(problem, "store", None) is None:
        raise RuntimeError("Need to specify '--store=path' on command line or problem.store='path' in definition file.")
    stem = (
        Path(problem.path).stem
        if hasattr(problem, "path")
        else sanitize_filename(problem.name)
        if problem.name
        else "problem"
    )
    problem.output_path = os.path.join(problem.store, stem)

    # Check if already exists
    store_exists = os.path.exists(problem.output_path + ".par")
    if not opts.overwrite and opts.resume is None and store_exists:
        if opts.batch:
            print(
                problem.store + " already exists.  Restart with --overwrite, --resume, or --store=newpath",
                file=sys.stderr,
            )
            sys.exit(1)
        exists_handler(problem.output_path)

    # Create it and copy model
    if not os.path.exists(problem.store):
        os.mkdir(problem.store)
    shutil.copy2(problem.path, problem.store)


def run_profiler(problem, steps):
    """
    Model execution profiler.

    Run the program with "--profiler --steps=N" to generate a function
    profile chart breaking down the cost of evaluating N models.
    """
    # Here is the findings from one profiling session::
    #   23 ms total
    #    6 ms rendering model
    #    8 ms abeles
    #    4 ms convolution
    #    1 ms setting parameters and computing nllf
    from .util import profile

    p = initpop.random_init(int(steps), initial=None, bounds=None, use_point=False, problem=problem)
    # Note: map is an iterator in python 3
    profile(lambda *args: list(map(*args)), problem.nllf, p)


def run_timer(mapper, problem, steps):
    """
    Model execution timer.

    Run the program with "--timer --steps=N" to determine the average
    run time of the model.  If --parallel is included, then the model
    will be run in parallel on separate cores.
    """
    import time

    T0 = time.time()
    steps = int(steps)
    p = initpop.generate(problem, init="random", pop=-steps, use_point=False)
    if p.shape == (0,):
        # No fitting parameters --- generate an empty population
        p = np.empty((steps, 0))
    mapper(p)
    print("time per model eval: %g ms" % (1000 * (time.time() - T0) / steps,))


def start_remote_fit(problem, options, queue, notify):
    """
    Queue remote fit.
    """
    from jobqueue.client import connect
    from cloudpickle import dumps

    data = dict(package="bumps", version=__version__, problem=dumps(problem), options=dumps(options))
    request = dict(
        service="fitter",
        version=__version__,  # fitter service version
        notify=notify,
        name=problem.title,
        data=data,
    )

    server = connect(queue)
    job = server.submit(request)
    return job


# ==== Main ====


def initial_model(opts):
    """
    Load and initialize the model.

    *opts* are the processed command line options.

    If --pars is in opts, then load the parameters from a .par file.

    If --simulate is in opts, then generate random data from the model.

    If --simrandom is in opts, then generate random data from a random model.

    If --shake is in opts, then use a random initial state for the fit.
    """
    if opts.seed is not None:
        np.random.seed(opts.seed)

    if opts.args:
        problem = load_model(opts.args[0], opts.args[1:])
        if opts.pars is not None:
            load_pars(problem, opts.pars)
        if opts.simrandom:
            problem.randomize()
        if opts.simulate or opts.simrandom:
            noise = None if opts.noise == "data" else float(opts.noise)
            problem.simulate_data(noise=noise)
            print("simulation parameters")
            print(problem.summarize())
            print("chisq at simulation", problem.chisq_str())
        if opts.shake:
            problem.randomize()
    else:
        problem = None
    return problem


def resynth(fitdriver, problem, mapper, opts):
    """
    Generate maximum likelihood fits to resynthesized data sets.

    *fitdriver* is a :class:`bumps.fitters.FitDriver` object with a fitter
    already chosen.

    *problem* is a :func:`bumps.fitproblem.FitProblem` object.  It should
    be initialized with optimal values for the parameters.

    *mapper* is one of the available :mod:`bumps.mapper` classes.

    *opts* is a :class:`bumps.options.BumpsOpts` object representing the command
    line parameters.
    """
    make_store(problem, opts, exists_handler=store_overwrite_query)
    fid = open(problem.output_path + ".rsy", "at")
    fitdriver.mapper = mapper.start_mapper(problem, opts.args)
    for i in range(opts.resynth):
        problem.resynth_data()
        best, fbest = fitdriver.fit()
        chisq = problem.chisq(nllf=fbest)
        print(f"step {i} chisq={chisq:.2f}")
        fid.write("%.15g " % chisq)
        fid.write(" ".join("%.15g" % v for v in best))
        fid.write("\n")
    problem.restore_data()
    fid.close()


def beep():
    """
    Audio signal that fit is complete.
    """
    if sys.platform == "win32":
        try:
            import winsound

            winsound.MessageBeep(winsound.MB_OK)
        except Exception:
            pass
    else:
        print("\a", file=sys.__stdout__)


def run_command(c):
    """
    Run an arbitrary python command.
    """
    exec(c, globals())


def setup_logging():
    """Start logger"""
    import logging

    logging.basicConfig(level=logging.INFO)


# From http://stackoverflow.com/questions/22373927/get-traceback-of-warnings
# answered by mgab (2014-03-13)
# edited by Gareth Rees (2015-11-28)
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    """
    Alternate warning printer which shows a traceback with the warning.

    To use, set *warnings.showwarning = warn_with_traceback*.
    """
    traceback.print_stack()
    log = file if hasattr(file, "write") else sys.stderr
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


def main():
    """
    Run the bumps program with the command line interface.

    Input parameters are taken from sys.argv.
    """
    from . import options

    # add full traceback to warnings
    # warnings.showwarning = warn_with_traceback

    if len(sys.argv) == 1:
        sys.argv.append("-?")

    # run command with bumps in the environment
    if sys.argv[1] == "-m":
        import runpy

        sys.argv = sys.argv[2:]
        runpy.run_module(sys.argv[0], run_name="__main__")
        sys.exit(0)
    elif sys.argv[1] == "-p":
        import runpy

        sys.argv = sys.argv[2:]
        runpy.run_path(sys.argv[0], run_name="__main__")
        sys.exit()
    elif sys.argv[1] == "-c":
        run_command(sys.argv[2])
        sys.exit()
    elif sys.argv[1] == "-i":
        sys.argv = ["ipython", "--pylab"]
        from IPython import start_ipython

        sys.exit(start_ipython())

    opts = options.getopts()
    setup_logging()

    if opts.edit:
        from .gui.gui_app import main as gui

        gui()
        return

    # Set up the matplotlib backend to minimize the wx/gui dependency.
    # If no GUI specified and not editing, then use the default mpl
    # backend for the python version.
    if opts.batch or opts.remote or opts.noshow:  # no interactivity
        config_matplotlib(backend="agg")
    else:  # let preview use default graphs
        config_matplotlib()

    problem = initial_model(opts)
    if problem is None:
        print("\n!!! Model file missing from command line --- abort !!!.", file=sys.stderr)
        sys.exit(1)

    if opts.mpi:
        MPIMapper.start_worker(problem)
        mapper = MPIMapper
    elif opts.parallel != "" or opts.worker:
        if opts.transport == "mp":
            mapper = MPMapper
        else:
            raise ValueError("unknown mapper")
    else:
        mapper = SerialMapper
    if opts.worker:
        mapper.start_worker(problem)
        return

    if np.isfinite(float(opts.time)):
        import time

        start_time = time.time()
        stop_time = start_time + float(opts.time) * 3600
        abort_test = lambda: time.time() >= stop_time
    else:
        abort_test = lambda: False

    fitdriver = FitDriver(
        opts.fit_config.selected_fitter, problem=problem, abort_test=abort_test, **opts.fit_config.selected_values
    )

    # Start fitter within the domain so that constraints are valid
    clipped = fitdriver.clip()
    if clipped:
        print("Start value clipped to range for parameter", ", ".join(clipped))

    if opts.time_model:
        run_timer(mapper.start_mapper(problem, opts.args), problem, steps=int(opts.steps))
    elif opts.profile:
        run_profiler(problem, steps=int(opts.steps))
    elif opts.chisq:
        if opts.cov:
            fitdriver.show_cov()
        print("chisq", problem.chisq_str())
        # import pprint; pprint.pprint(problem.to_dict(), indent=2, width=272)
    elif opts.preview:
        if opts.cov:
            fitdriver.show_cov()
        preview(problem, view=opts.view)
    elif opts.resynth > 0:
        resynth(fitdriver, problem, mapper, opts)

    elif opts.remote:
        # Check that problem runs before submitting it remotely
        # TODO: this may fail if problem requires remote resources such as GPU
        print("initial chisq:", problem.chisq_str())
        job = start_remote_fit(problem, opts, queue=opts.queue, notify=opts.notify)
        print("remote job:", job["id"])

    else:
        # Show command line arguments and initial model
        print("#", " ".join(sys.argv), "--seed=%d" % opts.seed)
        problem.show()

        # Check that there are parameters to be fitted.
        if not len(problem.getp()):
            print("\n!!! No parameters selected for fitting---abort !!!\n", file=sys.stderr)
            sys.exit(1)

        # Run the fit
        if opts.resume == "-":
            opts.resume = opts.store if os.path.exists(opts.store) else None
        if opts.resume:
            resume_path = os.path.join(opts.resume, problem.name)
        else:
            resume_path = None

        make_store(problem, opts, exists_handler=store_overwrite_query)

        # Redirect sys.stdout to capture progress
        if opts.batch:
            sys.stdout = open(problem.output_path + ".mon", "w")

        # TODO: fix techical debt with checkpoint monitor implementation
        # * The current checkpoint implementation is self-referential:
        #     checkpoint = lambda: save_best(fitdriver, ...)
        #     fitdriver.monitors = [..., CheckpointMonitor(checkpoint), ...]
        #   It is done this way because the checkpoint monitor needs the fitter
        #   so it can ask it to save state, but the fitter needs the list of
        #   monitors, including the checkpoint monitor, before it is run.
        # * Figures are cumulative, with each checkpoint adding a new set
        # * Figures are slow! Can they go into a separate thread?  Can we
        #   have the problem cache the best value?
        checkpoint_time = float(opts.checkpoint) * 3600

        def checkpoint(history):
            problem = fitdriver.problem
            ## Use the following to save only the fitter state
            fitdriver.fitter.save(problem.output_path)
            ## Use the following to save the fitter state plus all other
            ## plots and other output files.  This won't work yet since
            ## plots are generated sequentially, with each checkpoint producing
            ## a completely new set of plots.
            # best = history.point[0]
            # save_best(fitdriver, problem, best, view=opts.view)

        monitors = [ConsoleMonitor(problem)]
        if checkpoint_time > 0 and np.isfinite(checkpoint_time):
            mon = CheckpointMonitor(checkpoint, progress=checkpoint_time)
            monitors.append(mon)
        if opts.stepmon:
            fid = open(problem.output_path + ".log", "w")
            mon = StepMonitor(problem, fid, fields=["step", "value"])
            monitors.append(mon)
        fitdriver.monitors = monitors

        # import time; t0=time.clock()
        cpus = int(opts.parallel) if opts.parallel != "" else 0
        fitdriver.mapper = mapper.start_mapper(problem, opts.args, cpus=cpus)
        best, fbest = fitdriver.fit(resume=resume_path)
        # print("time=%g"%(time.clock()-t0),file=sys.__stdout__)
        # Note: keep this in sync with the checkpoint function above
        save_best(fitdriver, view=opts.view)
        fitdriver.show()
        if opts.err or opts.cov:
            fitdriver.show_err()
        if opts.cov:
            fitdriver.show_cov()
        if opts.entropy:
            fitdriver.show_entropy(opts.entropy)
        mapper.stop_mapper()

        # If in batch mode then explicitly close the monitor file on completion
        if opts.batch:
            sys.stdout.close()
            sys.stdout = sys.__stdout__

        # Display the plots
        if not opts.batch and not opts.mpi and not opts.noshow:
            beep()
            import matplotlib.pyplot as plt

            plt.show()


# Allow  "$python -m bumps.cli args" calling pattern
if __name__ == "__main__":
    main()
