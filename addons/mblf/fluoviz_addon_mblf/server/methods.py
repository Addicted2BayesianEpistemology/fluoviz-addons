"""The methods paragraph of a result (for a report) and the references."""
from __future__ import annotations

REFERENCES = [
    "B. Ursino, I. Epifani and F. Trovò. A Bayesian latent factors approach for multimodal functional data (manuscript; software: mblf-python, https://github.com/Addicted2BayesianEpistemology/mblf-python).",
    "A. Bhattacharya and D. B. Dunson (2011). Sparse Bayesian infinite factor models. Biometrika 98(2), 291–306 (the multiplicative gamma process shrinkage prior).",
    "B. Carpenter et al. (2017). Stan: a probabilistic programming language. Journal of Statistical Software 76(1).",
    "P. Papastamoulis and I. Ntzoufras (2022). On the identifiability of Bayesian factor analytic models. Statistics and Computing 32, 23 (Varimax-RSP alignment).",
    "A. Vehtari, A. Gelman and J. Gabry (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. Statistics and Computing 27, 1413–1432.",
    "A. Vehtari, A. Gelman, D. Simpson, B. Carpenter and P.-C. Bürkner (2021). Rank-normalization, folding, and localization: an improved R-hat. Bayesian Analysis 16(2), 667–718.",
]


CENTRE_WORDS = {"mean": "centred on the mean of the training samples", "median": "centred on the median of the training samples", "none": "not centred"}
SCALE_WORDS = {"global": "scaled to unit root mean square", "pointwise": "every point divided by its standard deviation", "pareto": "every point divided by the square root of its "
               "standard deviation (Pareto scaling)", "none": "not scaled"}


def _prep_words(m: dict) -> str:
    p = m.get("prep") or {}
    lg = m.get("log")
    words = f"{CENTRE_WORDS.get(p.get('centre', 'mean'))} and {SCALE_WORDS.get(p.get('scaling', 'global'))}"
    return (f"modelled as log10(value + {lg['offset']:.3g}), " if lg else "") + words


def methods_text(r) -> str:
    pl = r.payload or {}
    pr = r.params or {}
    modes = pl.get("modes", [])
    parts = []
    for m in modes:
        if m["kind"] == "vector":
            b = m.get("basis") or {}
            basis = (f"basis = {m['n_basis']} principal component(s) of their correlations, keeping {100 * (m.get('captured') or 0):.1f} % of their variance"
                     if b.get("kind") == "pca" else "identity basis")
            parts.append(f"the values {', '.join(m.get('labels', []))} (each centred and divided by its standard deviation; {basis})")
        elif (m.get("basis") or {}).get("from") == "fpca":
            b = m.get("basis", {})
            parts.append(f"the {m['label']} ({m['n_points']} points; basis = the first {m['n_basis']} eigenfunctions of a functional PCA of the training samples, smoothed by maximum "
                         f"likelihood with a {b.get('kernel', 'Matérn 5/2')} Gaussian-process basis, holding {100 * (m.get('captured') or 0):.1f} % of their smoothed variation, each "
                         f"scaled by the square root of its eigenvalue; {_prep_words(m)})")
        else:
            b = m.get("basis", {})
            parts.append(f"the {m['label']} ({m['n_points']} points; Karhunen-Loeve basis of a {b.get('kernel', 'Matérn 5/2')} Gaussian process with length-scales "
                         f"{', '.join(f'{x:.3g}' for x in b.get('lengths', []))} nm ({b.get('lengths_source', 'given')}), {m['n_basis']} functions keeping "
                         f"{100 * (m.get('captured') or 0):.1f} % of the prior variance; {_prep_words(m)})")
    cov = (pl.get("data") or {}).get("covariates")
    model = pr.get("model", {})
    s = pr.get("sampler", {})
    est = pl.get("estimation", "bayes")
    txt = (f"A Bayesian multimodal latent factor model (Ursino, Epifani and Trovò; mblf-python 1.0.0) was fitted to {len(modes)} mode(s) of {pl.get('data', {}).get('n', '?')} samples: "
           + "; ".join(parts) + ". ")
    txt += (f"Up to {pl.get('K')} shared factors were allowed; the multiplicative gamma process prior shrinks the factors the data do not need. "
            f"Samples that lack part of a mode, or a whole mode, contributed what they had (the likelihood was evaluated exactly from the observed points). ")
    if cov:
        logs = [c["label"] for c in cov["columns"] if c.get("log")]
        txt += (f"The means of the factor scores depended linearly on the covariates {', '.join(cov['labels'])} (numbers standardised"
                + (f", {', '.join(logs)} on the log scale" if logs else "") + "; categories as centred indicators). ")
    if est == "bayes":
        txt += (f"The posterior was sampled with the No-U-Turn sampler of Stan ({s.get('chains')} chains of {s.get('warmup')} warm-up and {s.get('draws')} kept iterations, target "
                f"acceptance {s.get('adapt_delta')}); convergence was checked with the rank-normalised split R-hat and the bulk and tail effective sample sizes of rotation-invariant "
                f"quantities (largest R-hat {pl.get('diagnostics', {}).get('max_rhat', float('nan')):.3f}, {pl.get('diagnostics', {}).get('n_divergent', 0)} divergent transitions). ")
    else:
        txt += f"The parameters were estimated by maximum likelihood (L-BFGS in Stan from {pr.get('ml', {}).get('starts', 8)} starting points); their uncertainty is not represented. "
    txt += ("The factors were aligned across draws by Varimax-RSP in the L2 geometry of the loading functions and ordered by strength. Predictions of one mode from the others use the "
            "Gaussian conditional distribution at every posterior draw, averaged over the draws (the modular rule); their predictive variance was split into the uncertainty of the "
            "population parameters, of the sample's factor scores, the sample-specific residual and the measurement noise. Leave-one-out predictive performance was estimated by "
            "Pareto-smoothed importance sampling. Model settings: " + ", ".join(f"{k} = {v}" for k, v in (model.get("priors") or {}).items()) + ".")
    return txt
