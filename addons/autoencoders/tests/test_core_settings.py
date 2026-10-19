"""The advanced training settings that need no PyTorch: the defaults are what the training did before they existed, every value is checked in plain English, the request carries them, the
time estimate and the words of the verdict and of the methods paragraph follow them."""
import pytest

from fluoviz_addon_autoencoders.core import advice as AD, diagnostics as G
from fluoviz_addon_autoencoders.core.spec import AutoencoderError, Config
from fluoviz_addon_autoencoders.server import methods as M
from fluoviz_addon_autoencoders.server.request import parse_request
from test_core_diagnostics import fake_result

NEW = {"stop_rule": "validation", "min_epochs": 0, "min_delta": 1e-3, "val_window": 1, "max_minutes": None, "restore_best": True, "lr_schedule": "cosine", "lr_final": 0.01,
       "final_epochs": "auto", "final_epochs_min": 20}


def test_the_defaults_are_what_the_training_always_did():
    c = Config()
    assert {k: getattr(c, k) for k in NEW} == NEW
    assert c.epochs == 600 and c.patience == 60 and c.kfold_below == 30 and c.grad_clip == 5.0 and c.uses_early_stopping()
    assert Config.from_dict({k: v for k, v in c.to_dict().items() if k not in NEW}).to_dict() == c.to_dict()          # a result made before the settings existed opens with the defaults
    assert Config.from_json(c.to_json()).to_dict() == c.to_dict()


@pytest.mark.parametrize("bad, words", [({"stop_rule": "sometimes"}, "stopping rule"), ({"min_epochs": -1}, "minimum number of epochs"), ({"min_epochs": 2.5}, "whole number"),
                                        ({"min_delta": -0.1}, "smallest improvement"), ({"min_delta": 0.9}, "smallest improvement"), ({"val_window": 0}, "averaged"),
                                        ({"val_window": 1.5}, "whole number"), ({"max_minutes": 0}, "time limit"), ({"max_minutes": "soon"}, "time limit"), ({"restore_best": "yes"}, "best epoch"),
                                        ({"lr_schedule": "linear"}, "learning-rate schedule"), ({"lr_final": 0}, "final learning rate"), ({"lr_final": 2}, "final learning rate"),
                                        ({"final_epochs": "never"}, "final networks"), ({"final_epochs": 0}, "final networks"), ({"final_epochs_min": 0}, "least number")])
def test_every_value_is_checked_in_plain_english(bad, words):
    with pytest.raises(AutoencoderError, match=words):
        Config(**bad)


def test_valid_values_are_kept():
    c = Config(stop_rule="never", min_epochs=100, min_delta=0.0, val_window=5, max_minutes=2.5, restore_best=False, lr_schedule="plateau", lr_final=0.1, final_epochs=300, final_epochs_min=40)
    assert (c.stop_rule, c.min_epochs, c.min_delta, c.val_window, c.max_minutes, c.restore_best, c.lr_schedule, c.lr_final, c.final_epochs, c.final_epochs_min) == \
        ("never", 100, 0.0, 5, 2.5, False, "plateau", 0.1, 300, 40) and not c.uses_early_stopping()
    assert Config(final_epochs="max").final_epochs == "max"
    assert Config.from_json(c.to_json()).to_dict() == c.to_dict()


def test_the_request_carries_the_settings_and_replays_them():
    r = parse_request({"latent_dim": 3, "stop_rule": "never", "min_epochs": 50, "lr_schedule": "constant", "final_epochs": "max", "input": {"variables": ["fluorescence"]}})
    assert r.cfg.stop_rule == "never" and r.cfg.min_epochs == 50 and r.cfg.lr_schedule == "constant" and r.cfg.final_epochs == "max"
    p = r.params()
    assert p["stop_rule"] == "never" and p["min_epochs"] == 50 and p["final_epochs"] == "max" and p["min_delta"] == 1e-3
    assert parse_request(p).cfg.to_dict() == r.cfg.to_dict()                                                           # the result's parameters replay to the same settings
    with pytest.raises(AutoencoderError, match="stopping rule"):
        parse_request({"stop_rule": "whenever", "input": {"variables": ["fluorescence"]}})


def test_the_time_estimate_follows_the_stopping_settings():
    n, P = 300, 20000
    base = AD.estimate_seconds(Config(latent_dim=3), n, P)
    full = AD.estimate_seconds(Config(latent_dim=3, stop_rule="never"), n, P)
    assert full == pytest.approx(base / 0.7)                                                                           # no early stopping: every network trains all its epochs
    capped = AD.estimate_seconds(Config(latent_dim=3, stop_rule="never", max_minutes=0.1), n, P)
    assert capped == pytest.approx(3 * 6.0) and capped < full                                                          # 3 networks, 6 seconds each at most
    assert AD.estimate_seconds(Config(latent_dim=3, max_minutes=1000.0), n, P) == pytest.approx(base)


def fake(config, **kw):
    r = fake_result(**kw)
    r.config = {"epochs": 600, "patience": 60, **config}
    return r


def test_the_verdict_says_why_training_ended_and_which_setting_to_change():
    early = G.training_verdict(fake({}))["stopping_words"]
    assert "stopped at epoch 160" in early and "validation loss had not improved for 60 epochs" in early and "epoch 101" in early and "raise the patience" in early and "Advanced training settings" in early
    assert "training loss had not improved" in G.training_verdict(fake({"stop_rule": "training"}))["stopping_words"]
    assert "the weights of the last epoch were kept" in G.training_verdict(fake({"restore_best": False}))["stopping_words"]
    r = fake({"max_minutes": 2.5}, best_epoch=300, epochs_run=410)
    r.restarts["runs"][0]["stopped"] = "time"
    t = G.training_verdict(r)["stopping_words"]
    assert "time limit of 2.5 minutes" in t and "epoch 301" in t
    r = fake({"stop_rule": "never"}, best_epoch=300, epochs_run=600)
    assert "Early stopping was off: all 600 epochs were trained" in G.training_verdict(r)["stopping_words"]
    assert "still near the end" in G.training_verdict(fake({}, best_epoch=590, epochs_run=600))["stopping_words"] and "all 600 epochs" in G.training_verdict(fake({}, best_epoch=300, epochs_run=600))["stopping_words"]
    pca = G.training_verdict(fake({}, best_epoch=-1, epochs_run=61))["stopping_words"]
    assert "could not improve on its PCA starting point" in pca and "epoch 0" not in pca and "least number of epochs" in pca
    off = G.training_verdict(fake({"stop_rule": "never"}, best_epoch=-1, epochs_run=600))["stopping_words"]
    assert "could not improve on its PCA starting point" in off and "Early stopping was off: all 600 epochs were trained." in off
    kf = G.training_verdict(fake({}, mode="kfold", epochs_run=20))["stopping_words"]
    assert "k-fold validation" in kf and "number of epochs ({})".format(20) in kf and "final networks" in kf
    assert "as chosen" in G.training_verdict(fake({"final_epochs": 150}, mode="kfold", epochs_run=150))["stopping_words"]


def test_the_methods_paragraph_reads_as_before_for_the_defaults_and_names_the_new_settings():
    d = Config().to_dict()
    assert M._schedule_words(d) == "with cosine decay" and M._stopping_words(d) == "up to 600 epochs with early stopping (patience 60)"
    n = {**d, "stop_rule": "training", "min_epochs": 100, "min_delta": 0.01, "val_window": 5, "max_minutes": 3.0, "lr_schedule": "plateau", "lr_final": 0.05}
    w = M._stopping_words(n)
    assert "on the training loss" in w and "never before epoch 100" in w and "at least 1 % counting" in w and "averaged over 5 epochs" in w and "at most 3 minutes per network" in w
    assert "halved whenever the loss stopped improving" in M._schedule_words(n) and "0.05" in M._schedule_words(n)
    assert M._stopping_words({**d, "stop_rule": "never"}) == "600 epochs without early stopping"
    assert M._schedule_words({**d, "lr_schedule": "constant"}) == "kept constant" and "cosine decay to 0.1 of its start" in M._schedule_words({**d, "lr_final": 0.1})
