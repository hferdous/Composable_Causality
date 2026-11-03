"""Counterfactual twins must agree exactly until the intervention starts."""
import numpy as np

from dtcd.generators import LinearTimeSeriesGenerator
from dtcd.generators.universal import UniversalTimeSeriesGenerator
from dtcd.interventions.base import CounterfactualSuite, InterventionManager
from dtcd.structure.graph import generate_random_dag
from dtcd.structure.scheduler import RegimeScheduler, SmoothDriftScheduler

START = 150


def _do_x1():
    im = InterventionManager()
    im.add_intervention(target="X1", intervention_type="do", value=2.0, start=START)
    return im


def _check(gen):
    obs, twin = CounterfactualSuite(gen).generate_twins(n_points=300, intervention_manager_for_twin=_do_x1())
    cols = [c for c in obs.columns if c.startswith("X")]
    diff = (twin[cols] - obs[cols]).to_numpy()
    assert np.all(diff[:START] == 0.0), "twins differ before the intervention"
    # The intervention takes effect on its target (observed values may carry
    # measurement noise, which is identical in both twins).
    assert np.all(diff[START:, cols.index("X1")] != 0.0)
    return diff


def test_linear_generator_twins_are_exact():
    _check(LinearTimeSeriesGenerator(n_vars=4, max_lag=2, random_state=0))


def test_universal_generator_twins_are_exact():
    links = generate_random_dag(n_vars=4, n_edges=6, max_lag=2, random_state=1)
    sched = RegimeScheduler(n_vars=4, max_lag=2, regimes=[{"start": 0, "links": links}])
    _check(UniversalTimeSeriesGenerator(n_vars=4, max_lag=2, scheduler=sched, random_state=0))


def test_drifting_graph_twins_are_exact():
    links = generate_random_dag(n_vars=4, n_edges=6, max_lag=2, random_state=1)
    sched = SmoothDriftScheduler(n_vars=4, max_lag=2, initial_links=links, drift_scale=0.01, random_state=0)
    _check(UniversalTimeSeriesGenerator(n_vars=4, max_lag=2, scheduler=sched, random_state=0))
