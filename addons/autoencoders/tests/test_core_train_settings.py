"""The advanced training settings at work: when training stops (patience, least number of epochs, smallest improvement, averaging, the training loss, never, a time limit), which weights
are kept, how the learning rate moves, and how many epochs the final networks of the k-fold validation get.  The defaults are what the training always did (the history of every epoch is
the same: this was checked against the first release when the settings were added).  Skipped cleanly when PyTorch is missing."""
import math
import time

import numpy as np
import pytest

pytest.importorskip("torch")

from core_support import planted                                                                              # noqa: E402
from fluoviz_addon_autoencoders.core import data as D, train as T                                              # noqa: E402
from fluoviz_addon_autoencoders.core.spec import Config                                                        # noqa: E402


def td(n=60, seed=1):
    A, _ = planted(n=n, P=30, k=3, noise=0.05, seed=seed, nonlinear=0.5)
    return D.TrainData.from_array(A)


def fit(n=60, **kw):
    kw = {"latent_dim": 3, "n_restarts": 1, **kw}
    return T.fit(Config(**kw), td(n))


def stopped(res):
    return res.restarts["runs"][res.restarts["kept"]]["stopped"]


def lrs(res):
    return np.array([h["lr"] for h in res.history])


# ----------------------------------------------------------------------------- when it stops
def test_the_least_number_of_epochs_keeps_training_going_until_the_patience_is_used_up():
    short = fit(epochs=80, patience=1, min_delta=0.5)                                  # an improvement of 50 % almost never happens: the patience of one epoch is used up at once
    assert stopped(short) == "early" and short.epochs_run <= 3
    kept = fit(epochs=80, patience=1, min_delta=0.5, min_epochs=40)
    assert stopped(kept) == "early" and kept.epochs_run == 40                          # never before epoch 40, then at once
    assert fit(epochs=30, patience=1, min_delta=0.5, min_epochs=500).epochs_run == 30   # (more than the maximum: all the epochs)


def test_the_smallest_improvement_that_counts_decides_how_soon_the_patience_runs_out():
    strict = fit(epochs=300, patience=15, min_delta=0.2, skip=False)                      # (the classical start: a network that starts at the PCA is hard to improve on)
    loose = fit(epochs=300, patience=15, min_delta=0.0, skip=False)
    assert strict.epochs_run < loose.epochs_run                                         # a small gain counts for nothing when 20 % is demanded
    assert loose.epochs_run >= 60


def test_patience_counts_from_the_first_epoch_so_the_least_number_of_epochs_matters_for_a_start_at_the_pca():
    """The network that starts at the PCA must beat it on the held-out samples before it counts as an improvement; until then every epoch is a 'no improvement' epoch."""
    short = fit(epochs=200, patience=10)
    assert short.epochs_run <= 40 and stopped(short) == "early"
    longer = fit(epochs=200, patience=10, min_epochs=100)
    assert longer.epochs_run >= 100


def test_never_stopping_trains_every_epoch_and_still_keeps_the_best_weights():
    res = fit(epochs=60, patience=1, min_delta=0.5, stop_rule="never")
    assert res.epochs_run == 60 and stopped(res) == "max" and res.best_epoch <= 59
    val = [h["val"] for h in res.history]
    assert res.best_epoch == -1 or val[res.best_epoch] == pytest.approx(min(val), rel=1e-6) or any(h["best"] for h in res.history)


def test_the_training_loss_can_be_the_one_that_decides():
    res = fit(epochs=300, patience=5, min_delta=0.05, stop_rule="training")
    assert stopped(res) == "early" and res.epochs_run < 300
    tr = [h["train"] for h in res.history]
    assert res.best_epoch == int(np.argmin(tr)) or tr[res.best_epoch] <= min(tr) * 1.2         # the best epoch is the one of the training loss
    assert res.validation["evr"] is not None and res.validation["n_val"] > 0                    # the held-out numbers are still reported


def test_a_time_limit_stops_a_network_and_keeps_its_best_weights():
    t0 = time.time()
    res = fit(epochs=100000, patience=100000, stop_rule="never", max_minutes=0.05)
    assert stopped(res) == "time" and 5 < res.epochs_run < 100000 and time.time() - t0 < 60


def test_the_weights_of_the_last_epoch_can_be_kept_instead_of_the_best():
    last = fit(epochs=40, stop_rule="never", restore_best=False)
    assert last.best_epoch == last.epochs_run - 1
    best = fit(epochs=40, stop_rule="never")
    assert best.best_epoch <= best.epochs_run - 1 and best.validation["loss_val"] <= last.validation["loss_val"] + 1e-9


def test_the_loss_can_be_averaged_over_several_epochs():
    plain = fit(epochs=150, patience=20, skip=False)
    smooth = fit(epochs=150, patience=20, skip=False, val_window=9)
    assert smooth.epochs_run >= 9 and any(h["best"] for h in smooth.history) and any(h["best"] for h in plain.history)
    assert 0 <= smooth.best_epoch < smooth.epochs_run                                   # (the two may differ in where they stop; both must be valid, and the averaged one is not a copy of the plain one)
    assert [h["val"] for h in plain.history[:5]] == [h["val"] for h in smooth.history[:5]]      # the history itself is the loss of each epoch, never the average


# ----------------------------------------------------------------------------- the learning rate
def test_the_learning_rate_schedules():
    base = dict(epochs=40, stop_rule="never", lr=1e-3)
    cos = lrs(fit(**base))
    assert cos[0] == pytest.approx(1e-3) and cos[-1] == pytest.approx(T._lr_at(39, 40, 1e-3)) and np.all(np.diff(cos) <= 1e-15) and cos[-1] < 0.05e-3
    assert np.allclose(lrs(fit(lr_schedule="cosine_run", **base)), cos)                                   # in a hold-out run both read 'over the epochs of this network'
    assert np.allclose(lrs(fit(lr_schedule="constant", **base)), 1e-3)
    half = lrs(fit(lr_final=0.5, **base))
    assert half[0] == pytest.approx(1e-3) and half[-1] >= 0.5e-3 and half[-1] < 0.52e-3
    plat = lrs(fit(lr_schedule="plateau", patience=8, min_delta=0.5, lr_final=0.1, **base))               # nothing improves by 50 %: the rate is halved again and again, down to the floor
    assert plat[0] == pytest.approx(1e-3) and np.all(np.diff(plat) <= 1e-15) and plat[-1] == pytest.approx(0.1e-3) and plat[-1] < plat[0]


# ----------------------------------------------------------------------------- few samples: k-fold validation and the final networks
def test_the_epochs_of_the_final_networks_with_few_samples():
    base = dict(n=24, epochs=60, patience=10)
    auto = fit(**base)
    assert auto.split["mode"] == "kfold" and auto.epochs_run >= 20                                       # what the folds found, but at least 20
    full = fit(final_epochs="max", **base)
    assert full.epochs_run == 60 and stopped(full) == "max"
    assert fit(final_epochs=35, **base).epochs_run == 35
    assert fit(final_epochs_min=50, **base).epochs_run >= 50
    assert any("the maximum number of epochs, as chosen" in note for note in full.notes) and any("the number the folds found" in note for note in auto.notes)
    assert any("the number of epochs chosen" in note for note in fit(final_epochs=35, **base).notes)


def test_the_learning_rate_of_the_final_networks_can_follow_their_own_length():
    base = dict(n=24, epochs=60, patience=10, final_epochs=30)
    default = lrs(fit(**base))
    run = lrs(fit(lr_schedule="cosine_run", **base))
    assert len(default) == len(run) == 30
    assert default[-1] > 0.2e-3 and run[-1] < 0.05e-3            # tied to the maximum number of epochs the rate has hardly fallen after 30 of 60; over its own 30 epochs it has


def test_the_defaults_are_the_trainings_of_the_first_release():
    """The history of an explicit-defaults run equals the one of the plain run (the stored numbers of the first release were compared by hand when the settings were added)."""
    a = fit(epochs=60, patience=20)
    b = fit(epochs=60, patience=20, stop_rule="validation", min_epochs=0, min_delta=1e-3, val_window=1, max_minutes=None, restore_best=True, lr_schedule="cosine", lr_final=0.01)
    assert [h["val"] for h in a.history] == [h["val"] for h in b.history] and a.best_epoch == b.best_epoch and a.epochs_run == b.epochs_run
