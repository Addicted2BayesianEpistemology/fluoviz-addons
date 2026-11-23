"""mblf: a Bayesian latent factors approach for multimodal functional data (BMLF).

    from mblf import FunctionalData, GPBasis, MvtBLF, fit_ml
"""

from .align import functional_varimax, match_columns, varimax, varimax_rsp
from .bases import GPBasis, fit_gp, kernel_matrix, trapezoid_weights
from .data import FunctionalData, mode_summaries
from .diagnostics import ess_bulk, ess_tail, log_mean_exp, loo, one_se_rule, psis, rhat
from .likelihood import marginal_loglik
from .ml import MLResult, fit_ml
from .model import MvtBLF, stan_model
from .secondary import (check_likelihood, effective_dimension, effective_ranks, heldout_scores, projection_diagnostics,
                        projection_reference, shared_spectrum)

__version__ = "1.0.0"
