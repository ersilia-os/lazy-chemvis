"""
The time and memory it took to produce a cached output.

A fit resumed after a crash reuses the featurizations and the TMAP layout already
on disk, so its own clock and memory readings would miss most of the work: an
Enamine run resumed after the TMAP step would report a TMAP time of two minutes
instead of the hours the layout took. Every cached output therefore stores what
it cost, in a ``cost.json`` next to its ``cache.json``, and a run that reuses it
adds that cost back into the manifest.

A cost is ``{"seconds": float, "peak_memory_gb": {"main": ..., "subprocesses":
...}}``. The peak is the producing process's running peak when the output was
finished, the same quantity the manifest records after each step, so the two
combine with a plain maximum.
"""

import json
import os

COST_FILENAME = "cost.json"


def read_cost(dir_path):
    """
    Return the cost stored in ``dir_path``, or None if there is none.

    Parameters
    ----------
    dir_path : str
        Directory holding a cached output.

    Returns
    -------
    dict or None
        The stored cost, or None when it is missing or unreadable (outputs
        written before costs were recorded have none).
    """
    path = os.path.join(dir_path, COST_FILENAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def write_cost(dir_path, cost):
    """
    Store ``cost`` in ``dir_path``.

    Parameters
    ----------
    dir_path : str
        Directory holding a cached output.
    cost : dict
        ``{"seconds": ..., "peak_memory_gb": ...}``.
    """
    os.makedirs(dir_path, exist_ok=True)
    partial = os.path.join(dir_path, COST_FILENAME + ".part")
    with open(partial, "w") as f:
        json.dump(cost, f, indent=2)
    os.replace(partial, os.path.join(dir_path, COST_FILENAME))


def max_peak(*peaks):
    """
    Element-wise maximum of peak-memory dicts, ignoring None entries and values.

    Parameters
    ----------
    *peaks : dict or None
        ``{"main": float or None, "subprocesses": float or None}``.

    Returns
    -------
    dict or None
        The combined peak, or None if no argument carried one.
    """
    peaks = [p for p in peaks if p]
    if not peaks:
        return None
    combined = {}
    for field in ("main", "subprocesses"):
        values = [p.get(field) for p in peaks if p.get(field) is not None]
        combined[field] = max(values) if values else None
    return combined


def combine(*costs):
    """
    Total cost of work split across runs: seconds add up, peaks take the maximum.

    Parameters
    ----------
    *costs : dict or None
        Costs to combine; None entries are skipped.

    Returns
    -------
    dict or None
        The combined cost, or None if every argument was None.
    """
    costs = [c for c in costs if c]
    if not costs:
        return None
    return {
        "seconds": sum(c.get("seconds", 0.0) for c in costs),
        "peak_memory_gb": max_peak(*(c.get("peak_memory_gb") for c in costs)),
    }


def add_cost(dir_path, cost):
    """
    Add ``cost`` to the cost already stored in ``dir_path``.

    Used per batch by the Ersilia featurizers, so the batches computed before a
    crash are still counted when the run is resumed.

    Parameters
    ----------
    dir_path : str
        Directory holding the partial output.
    cost : dict
        The cost of the work just finished.
    """
    write_cost(dir_path, combine(read_cost(dir_path), cost))


def merge_measures(timings, after_step, run_peak, carried):
    """
    Fold the cost of reused outputs into one run's own measurements.

    Parameters
    ----------
    timings : dict
        ``{step: seconds}`` measured in this run.
    after_step : dict
        ``{step: peak}`` measured in this run, in execution order; each value is
        the running peak of this process up to the end of that step.
    run_peak : dict
        This process's peak at the end of the run.
    carried : dict
        ``{step: [cost, ...]}``: what the outputs reused in each step cost when
        they were produced.

    Returns
    -------
    tuple of (dict, dict, dict, list)
        Timings, peaks after each step, the run peak, and the steps whose peak is
        unknown, as if every reused output had been computed in this run.

    Notes
    -----
    A reused output may carry a time but no peak (a cost reconstructed from a log).
    Its step's peak is then reported as None, and so is the run peak, rather than
    the maximum of the parts that were measured: that maximum would understate
    the run, and for the TMAP layout, which is the largest subprocess, by a lot.
    """
    unknown = {"main": None, "subprocesses": None}
    merged_timings = {
        step: seconds + sum(c.get("seconds", 0.0) for c in carried.get(step, []))
        for step, seconds in timings.items()
    }
    merged_after = {}
    unknown_steps = []
    running = None
    for step, peak in after_step.items():
        reused_peaks = [c.get("peak_memory_gb") for c in carried.get(step, [])]
        running = max_peak(running, peak, *reused_peaks)
        if any(not p for p in reused_peaks):
            unknown_steps.append(step)
            merged_after[step] = dict(unknown)
        else:
            merged_after[step] = running
    merged_run = dict(unknown) if unknown_steps else max_peak(run_peak, running)
    return merged_timings, merged_after, merged_run, unknown_steps
