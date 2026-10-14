"""Training, in a separate process (``core.worker``): the server only hands over the numbers and waits.  Nothing here imports PyTorch."""
from __future__ import annotations

from ..core import worker


def train(cfg, prepared, history_path, progress):
    """Train the network of ``cfg`` on ``prepared`` (``core.data.Prepared``); returns ``core.train.TrainResult``.  ``progress``: a ``Progress`` (cancelling terminates the process)."""
    return worker.run_in_worker("fit", {"config": cfg, "data": prepared.train_data(cfg.smoothify_metric)}, progress=progress, history_path=history_path)
