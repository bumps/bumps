"""
Warm-start integration test for load_fit_state / set_problem_keep_state.

Usage (private server)::

    # Run using a private server
    python check_warm_start.py

Usage (shared server)::

    # Start a server in a separate terminal
    python -m bumps --port=8502

    # Run using a shared server
    python check_warm_start.py http://localhost:8502

The script runs N_ITERATIONS rounds.  Each round adds one new data point to a
growing Gaussian dataset, submits the updated FitProblem via
update_serialized_problem (preserving chain state from the previous round), and
resumes the DREAM fit.  Chain heads from the prior round seed the new fit,
so burn-in should shrink with each iteration.

On the first iteration a cold start is done via set_serialized_problem so
there is a valid fit_state in the session before warm-starting begins.
"""

import sys
import asyncio
import json
from typing import cast

import numpy as np

import bumps.names as bp

# ── configuration ─────────────────────────────────────────────────────────────
HOST = "127.0.0.1"
PORT = 8502
N_ITERATIONS = 15
FIT_SECONDS = 30  # wall time to let each fit run before stopping
POLL_INTERVAL = 2.0  # seconds between active_fit polls

# ── ground truth ──────────────────────────────────────────────────────────────
TRUE_A = 3.0
TRUE_X0 = 0.0
TRUE_SIGMA = 1.0
NOISE = 0.2

rng = np.random.default_rng(42)


def gaussian(x, A, x0, sigma):
    """Fixed x0=0, sigma=1 — only A is free."""
    return A * np.exp(-0.5 * ((x - x0) / sigma) ** 2)


def make_problem(xdata, ydata, dydata):
    """Build a serialized FitProblem JSON string for the current dataset."""
    plot_x = np.linspace(xdata.min() - 1, xdata.max() + 1, 400)
    m = bp.Curve(gaussian, xdata, ydata, dydata, A=1.0, x0=5.0, sigma=0.1, plot_x=plot_x)
    m.A.range(0.0, 10.0)
    m.x0.range(-5.0, 5.0)
    m.sigma.range(0.0, 5.0)

    problem = bp.FitProblem(m)
    return json.dumps(bp.serialize(problem))


# ── main loop ─────────────────────────────────────────────────────────────────


async def run(client: bp.BumpsClient):
    # Accumulate data points one per iteration
    all_x = np.array([], dtype=float)
    all_y = np.array([], dtype=float)
    all_dy = np.array([], dtype=float)

    dream_options = {"samples": 5000, "burn": 1000, "alpha": 0.05, "pop": 8, "thin": 1, "init": "lhs"}

    has_prior = False

    for i in range(N_ITERATIONS):
        print(f"\n{'='*60}")
        print(f"Iteration {i+1}/{N_ITERATIONS}")

        # Add one new data point drawn from the true Gaussian + noise
        x_new = rng.uniform(-3.0, 3.0)
        y_new = gaussian(x_new, TRUE_A, TRUE_X0, TRUE_SIGMA) + rng.normal(0, NOISE)
        all_x = np.append(all_x, x_new)
        all_y = np.append(all_y, y_new)
        all_dy = np.append(all_dy, NOISE)
        print(f"  dataset now has {len(all_x)} point(s); new x={x_new:.3f} y={y_new:.3f}")
        if len(all_x) < 4:
            continue

        serialized = make_problem(all_x, all_y, all_dy)

        if not has_prior:
            # ── cold start ────────────────────────────────────────────────────
            print("  cold start: set_serialized_problem")
            await client.set_serialized_problem(
                serialized,  # serialized: str
                True,  # new_model: bool
                f"iter_{i+1}",  # name: str
                "dataclass",  # method: str
            )
            await client.start_fit_thread("dream", dream_options, False)

        else:
            # ── warm start ────────────────────────────────────────────────────
            print("  warm start: prepare_warm_start -> start_fit_thread(resume=True)")
            print("calling update_serialized")
            await client.update_serialized_problem(serialized, "dataclass", f"iter_{i+1}")
            await client.start_fit_thread("dream", dream_options, True)

        await client.wait_for_fit(timeout=FIT_SECONDS)
        has_prior = True

        # Print current best estimate of A
        params = await client.get_parameters(True)  # only_fittable=True
        if isinstance(params, list):
            for p in params:
                if isinstance(p, dict):
                    print(f"  {p['name']} = {p['value_str']}  [{p.get('min_str','?')}, {p.get('max_str','?')}]")


async def main():
    # Use an existing server if provided, otherwise start our own
    # url = f"http://{HOST}:{PORT}"
    url = sys.argv[1] if sys.argv[1:] else None

    async with bp.remote_bumps(url=url) as client:
        client = cast(bp.BumpsClient, client)  # Shouldn't be necessary...
        await run(client)


if __name__ == "__main__":
    asyncio.run(main())
