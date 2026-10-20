"""The methods paragraph (for a paper) and its references (SPEC section 11)."""
from __future__ import annotations

from ..core.spec import METRIC_WORDS

REFERENCES = [
    "Kohonen, T. (2013) Essentials of the self-organizing map. Neural Networks 37, 52–65.",
    "Kohonen, T. (2001) Self-Organizing Maps, 3rd ed., Springer.",
    "Vesanto, J., Himberg, J., Alhoniemi, E., Parhankangas, J. (2000) SOM Toolbox for Matlab 5. Report A57, Helsinki University of Technology.",
    "Vesanto, J., Alhoniemi, E. (2000) Clustering of the self-organizing map. IEEE Transactions on Neural Networks 11(3), 586–600.",
    "Ultsch, A., Siemon, H.P. (1990) Kohonen's self organizing feature maps for exploratory data analysis. Proc. INNC'90.",
    "Kiviluoto, K. (1996) Topology preservation in self-organizing maps. Proc. ICNN, 294–299.",
    "Bauer, H.-U., Pawelzik, K. (1992) Quantifying the neighborhood preservation of self-organizing feature maps. IEEE Transactions on Neural Networks 3(4), 570–579.",
    "Venna, J., Kaski, S. (2001) Neighborhood preservation in nonlinear projection methods: an experimental study. ICANN 2001, LNCS 2130, 485–491.",
    "Rossi, F., Conan-Guez, B., El Golli, A. (2004) Clustering functional data with the SOM algorithm. ESANN 2004, 305–312.",
    "Sammon, J.W. (1969) A nonlinear mapping for data structure analysis. IEEE Transactions on Computers C-18, 401–409.",
    "Bieroza, M., Baker, A., Bridgeman, J. (2009) Exploratory analysis of excitation-emission matrix fluorescence spectra with self-organizing maps. J. Geophys. Res. Biogeosci. 114, G00F07.",
    "Ejarque-Gonzalez, E., Butturini, A. (2014) Self-organising maps and correlation analysis as a tool to explore patterns in excitation-emission matrix data sets and to discriminate dissolved organic matter fluorescence components. PLoS ONE 9(6), e99618.",
    "Cuss, C.W., Guéguen, C. (2016) Analysis of dissolved organic matter fluorescence using self-organizing maps: mini-review and tutorial. Analytical Methods 8, 716–725.",
]


def _size(r) -> str:
    m = r.payload["map"]
    return f"{m['rows']} × {m['cols']} {m['lattice']} units{' on a toroid' if m['shape'] == 'toroid' else ''}"


def methods_text(r) -> str:
    pl, par = r.payload, r.params or {}
    q, tr = pl["quality"], par.get("train", {})
    inp = par.get("input", {})
    n = pl["n"]
    if inp.get("kind") == "variables":
        what = f"{len(inp.get('keys', []))} per-sample variables" + (" (standardised)" if inp.get("standardize", True) else "")
        reading = ""
    else:
        names = " and ".join({"fluorescence": "fluorescence excitation-emission matrices", "absorbance": "absorbance spectra"}[v] for v in pl["variables"])
        what = names
        bits = []
        for b in pl["input"]["blocks"]:
            how = "the observed pixels themselves" if b["basis"] == "pixels" else f"the coefficients of a {b['basis']} basis"
            tr_ = b.get("transform")
            bits.append(f"the {b['name']} was read as {how}" + (f" after the pre-treatment '{tr_}'" if tr_ and tr_ != "none" else ""))
        reading = "; ".join(bits) + ". "
    met = pl["metric"]
    mtext = METRIC_WORDS[met["kind"]]
    if met["kind"] == "mahalanobis":
        mtext += f" on {met['components']} leading directions (exponent {met['alpha']:g})"
    s = (f"A self-organising map (Kohonen, 2013) of {n} samples of {what} was computed with the add-on 'Self-organising map' of FluoViz. {reading}Samples were compared with prototypes with the "
         f"{mtext}. The map had {_size(r)} ({pl['map']['units']} units), was initialised {'on the plane of the first two principal directions' if tr.get('init', 'linear') == 'linear' else 'at random'} and "
         f"trained with the {tr.get('mode', 'batch')} algorithm, a {tr.get('neighbourhood', 'gaussian')} neighbourhood and {tr.get('epochs_rough', 50)} rough and {tr.get('epochs_fine', 150)} fine epochs "
         f"(seed {par.get('seed', 0)}). ")
    s += (f"The map captured {100 * q['captured']:.1f} % of the variation between the samples (quantisation error {q['qe_mean']:.3g}); the topographic error was {100 * q['topographic_error']:.1f} %, the "
          f"trustworthiness of the map at {q['trust_map']['k']} neighbours {q['trust_map']['trustworthiness']:.3f} (first two principal components: {q['trust_pca']['trustworthiness']:.3f}). ")
    if "heldout_captured" in q:
        s += f"In {pl['kfold']['folds']}-fold validation, with the distance refitted in every fold, the map captured {100 * q['heldout_captured']:.1f} % of the variation of held-out samples. "
    cl = pl["clusters"]
    s += f"The prototypes were grouped into {cl['k']} groups ({cl['method']} clustering, Vesanto and Alhoniemi, 2000). {cl['reason']}"
    return s.strip()
