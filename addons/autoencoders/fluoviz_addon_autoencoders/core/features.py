"""Features of a function: the single numbers the response curves and the heat map of two codes are drawn for (pure numpy).

A *feature* turns a function of the network (the decoding of one code, or a sample) into one number, in the units of the data after the pre-treatment::

    moved         the intensity at the pixel (or the absorbance at the wavelength) that a code moves most: the pixel with the largest range of the decoded response along
                  the traversal.  It is a different pixel for every code, so it is resolved by the tool that knows the traversal (``moved_index``)
    point         the intensity at the grid pixel nearest to an excitation / emission wavelength (that pixel must be part of the model)
    region_mean   the mean over the pixels of the model inside a box  (excitation range, emission range)
    region_max    the highest value among those pixels
    total         the overall intensity: the mean over the whole domain with the quadrature weights ``domain.w`` (they add up to 1), the L2 mean of the function
    absorbance    the absorbance at the wavelength nearest to a given one (only when the network has an absorbance block)
    coordinate    the k-th network coordinate of a block (a functional network: the standardised coefficient of the k-th term of its basis; no unit)

``compile_feature`` checks a request (``{"kind": "point", "ex": 350, "em": 450}`` ...) against the network and says in plain English what is wrong; ``Feature.evaluate``
evaluates it on functions ``{block: (k, m)}`` (``m`` = the observed pixels of the block) and, for a coordinate, on the network coordinates ``(k, P)``.
Values that are NaN (a pixel that was not measured in a sample) are left out of the means, the maxima and the total.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .spec import AutoencoderError

KINDS = ("moved", "point", "region_mean", "region_max", "total", "absorbance", "coordinate")


def _num(v, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v):
        raise AutoencoderError(f"{what} must be a number (in nm).")
    return float(v)


def _pair(v, what: str) -> tuple[float, float]:
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        raise AutoencoderError(f"{what} must be two numbers: the lowest and the highest wavelength of the box (in nm).")
    a, b = _num(v[0], what), _num(v[1], what)
    return (a, b) if a <= b else (b, a)


def _nm(x: float) -> str:
    return f"{x:g}"


@dataclass
class Feature:
    """One compiled feature (see the module text).  ``where`` says what was resolved (the pixel used, the box ...) for the page."""
    id: str
    kind: str
    block: str
    label: str
    short: str
    units: str
    per_code: bool = False
    index: int | None = None                       # the pixel (point, absorbance)
    mask: np.ndarray | None = field(default=None, repr=False)        # the pixels of a box
    k: int | None = None                           # the coordinate (0-based, within its block)
    offset: int = 0                                # where the block starts among the network coordinates
    w: np.ndarray | None = field(default=None, repr=False)           # quadrature weights (total)
    where: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ the number
    def evaluate(self, F: dict, X: np.ndarray | None = None, index: int | None = None) -> np.ndarray:
        """The feature of ``k`` functions ``F[block]`` ``(k, m)`` (and of the network coordinates ``X`` ``(k, P)`` for a coordinate): ``(k,)``.  ``index``: the pixel for ``moved``."""
        if self.kind == "coordinate":
            if X is None:
                raise AutoencoderError("This feature needs the network coordinates.")
            return np.asarray(X, float)[:, self.offset + self.k]
        A = np.asarray(F[self.block], float)
        if self.kind in ("point", "absorbance"):
            return A[:, self.index]
        if self.kind == "moved":
            if index is None:
                raise AutoencoderError("The pixel this code moves most has to be found first.")
            return A[:, index]
        if self.kind in ("region_mean", "region_max"):
            S = A[:, self.mask]
            with warnings.catch_warnings():                                   # (a row with no measured pixel in the box is NaN, not a warning)
                warnings.simplefilter("ignore", RuntimeWarning)
                return np.nanmean(S, axis=1) if self.kind == "region_mean" else np.nanmax(S, axis=1)
        ok = np.isfinite(A)                                                   # total
        if ok.all():
            return A @ self.w
        num = np.where(ok, A, 0.0) @ self.w
        den = ok @ self.w
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)

    def info(self) -> dict:
        """What the page needs to know about the feature."""
        return {"id": self.id, "kind": self.kind, "block": self.block, "label": self.label, "short": self.short, "units": self.units, "per_code": self.per_code, "where": dict(self.where)}


def moved_index(F: np.ndarray) -> int:
    """The pixel with the largest range along the traversal: ``F`` ``(k, m)`` (the decoded response, one row per position)."""
    F = np.asarray(F, float)
    return int(np.argmax(np.nanmax(F, axis=0) - np.nanmin(F, axis=0)))


def pixel_where(block, index: int) -> dict:
    """The wavelengths of observed pixel ``index`` of a block: ``{"ex", "em"}`` for a surface, ``{"wavelength"}`` for a curve."""
    dom = block.domain
    gi = dom.grid_index()
    if dom.kind == "surface":
        return {"em": float(dom.axes[0][gi[0][index]]), "ex": float(dom.axes[1][gi[1][index]])}
    return {"wavelength": float(dom.axes[0][gi[0][index]])}


def pixel_text(where: dict) -> str:
    if "wavelength" in where:
        return f"{_nm(where['wavelength'])} nm"
    return f"excitation {_nm(where['ex'])} nm, emission {_nm(where['em'])} nm"


def _surface_block(frame, name: str | None):
    for b in frame.blocks:
        if b.domain.kind == "surface" and (name is None or b.name == name):
            return b
    raise AutoencoderError("This network does not use the fluorescence, so there is no pixel of an EEM to look at." if name in (None, "fluorescence") else f"The network has no block '{name}'.")


def _find_block(frame, name: str | None, kinds=("surface", "curve")):
    if name is None:
        for b in frame.blocks:
            if b.domain.kind in kinds:
                return b
    else:
        for b in frame.blocks:
            if b.name == name:
                return b
        raise AutoencoderError(f"The network has no '{name}' (it models: {', '.join(frame.names())}).")
    raise AutoencoderError("The network has nothing to take this feature of.")


def _nearest(axis: np.ndarray, x: float, what: str) -> int:
    step = float(np.median(np.diff(axis))) if axis.size > 1 else 1.0
    if x < axis[0] - step or x > axis[-1] + step:
        raise AutoencoderError(f"The {what} {_nm(x)} nm is outside the range of the data the network uses ({_nm(float(axis[0]))}–{_nm(float(axis[-1]))} nm).")
    return int(np.argmin(np.abs(axis - x)))


def compile_feature(spec: dict, frame, fid: str = "f1", pair: bool = False) -> Feature:
    """A feature request (a dictionary, see the module text) checked against the network (``frame``: ``core.data.AEFrame``).  ``AutoencoderError`` in plain English.
    ``pair``: the feature is looked at over two codes (the words of ``moved`` then say "the two codes")."""
    if not isinstance(spec, dict):
        raise AutoencoderError("A feature must be a group of settings, for example {'kind': 'total'}.")
    kind = spec.get("kind")
    if kind not in KINDS:
        raise AutoencoderError(f"Choose what to measure: the kind of a feature must be one of: {', '.join(KINDS)}.")
    known = {"kind", "block", "ex", "em", "wavelength", "k", "id"}
    bad = set(spec) - known
    if bad:
        raise AutoencoderError(f"Unknown feature setting '{sorted(bad)[0]}'.")
    name = spec.get("block")
    if name is not None and not isinstance(name, str):
        raise AutoencoderError("The 'block' of a feature must be the name of what the network models (fluorescence or absorbance).")
    if kind == "moved":
        blk = _find_block(frame, name)
        what = "pixel" if blk.domain.kind == "surface" else "wavelength"
        who = "the two codes move" if pair else "this code moves"
        return Feature(fid, kind, blk.name, f"Intensity at the pixel {who} most" if blk.domain.kind == "surface" else f"Absorbance at the wavelength {who} most",
                       f"Most moved {what}", str(blk.element.units or ""), per_code=True)
    if kind == "point":
        blk = _surface_block(frame, name)
        dom = blk.domain
        ex, em = _num(spec.get("ex"), "The excitation wavelength"), _num(spec.get("em"), "The emission wavelength")
        iem, iex = _nearest(dom.axes[0], em, "emission wavelength"), _nearest(dom.axes[1], ex, "excitation wavelength")
        flat = iem * dom.shape[1] + iex
        hit = np.flatnonzero(dom.idx == flat)
        if not hit.size:
            raise AutoencoderError(f"The pixel at excitation {_nm(ex)} nm and emission {_nm(em)} nm is not part of the model (it is scatter, or below the Rayleigh line, or was cut out). "
                                   "Choose a pixel in the coloured part of the picture.")
        where = {"ex": float(dom.axes[1][iex]), "em": float(dom.axes[0][iem])}
        lab = f"Intensity at excitation {_nm(where['ex'])} nm, emission {_nm(where['em'])} nm"
        return Feature(fid, kind, blk.name, lab, f"Pixel at {_nm(where['ex'])} / {_nm(where['em'])} nm", str(blk.element.units or ""), index=int(hit[0]), where=where)
    if kind in ("region_mean", "region_max"):
        blk = _surface_block(frame, name)
        dom = blk.domain
        ex, em = _pair(spec.get("ex"), "The excitation range"), _pair(spec.get("em"), "The emission range")
        co = dom.coords()                                                      # (m, 2): emission, excitation
        mask = (co[:, 0] >= em[0]) & (co[:, 0] <= em[1]) & (co[:, 1] >= ex[0]) & (co[:, 1] <= ex[1])
        if not mask.any():
            raise AutoencoderError(f"No pixel of the model lies inside the box (excitation {_nm(ex[0])}–{_nm(ex[1])} nm, emission {_nm(em[0])}–{_nm(em[1])} nm). Draw it over the coloured part of the picture.")
        box = f"excitation {_nm(ex[0])}–{_nm(ex[1])} nm, emission {_nm(em[0])}–{_nm(em[1])} nm"
        what = "Mean intensity" if kind == "region_mean" else "Highest intensity"
        return Feature(fid, kind, blk.name, f"{what} in the box {box}", f"{'Mean' if kind == 'region_mean' else 'Highest'} in a box", str(blk.element.units or ""), mask=mask,
                       where={"ex": [ex[0], ex[1]], "em": [em[0], em[1]], "n_pixels": int(mask.sum())})
    if kind == "total":
        blk = _find_block(frame, name)
        if blk.domain.kind == "surface":
            lab, short = "Overall intensity (the average over the whole EEM)", "Overall intensity"
        else:
            lab, short = "Overall absorbance (the average over all wavelengths)", "Overall absorbance"
        return Feature(fid, kind, blk.name, lab, short, str(blk.element.units or ""), w=np.asarray(blk.domain.w, float))
    if kind == "absorbance":
        blk = _find_block(frame, name or "absorbance") if any(b.name == "absorbance" for b in frame.blocks) else None
        if blk is None or blk.domain.kind != "curve":
            raise AutoencoderError("This network does not use the absorbance, so there is no absorbance to look at.")
        dom = blk.domain
        wl = _num(spec.get("wavelength"), "The wavelength")
        i = _nearest(dom.axes[0], wl, "wavelength")
        hit = np.flatnonzero(dom.idx == i)
        if not hit.size:
            raise AutoencoderError(f"The absorbance at {_nm(wl)} nm is not part of the model. Choose a wavelength inside the range the network uses.")
        where = {"wavelength": float(dom.axes[0][i])}
        return Feature(fid, kind, blk.name, f"Absorbance at {_nm(where['wavelength'])} nm", f"Absorbance at {_nm(where['wavelength'])} nm", str(blk.element.units or "absorbance"), index=int(hit[0]), where=where)
    # coordinate
    blk = _find_block(frame, name)
    if blk.kind == "pixels":
        raise AutoencoderError("The coefficients of a basis exist only for a functional network; this network works on the pixels. Choose another feature.")
    k = spec.get("k")
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= blk.K:
        raise AutoencoderError(f"The number of the coefficient must be a whole number from 1 to {blk.K}.")
    offset = 0
    for b in frame.blocks:
        if b.name == blk.name:
            break
        offset += b.K
    return Feature(fid, kind, blk.name, f"Coefficient {k} of the {blk.name} (a number the network works with)", f"Coefficient {k}", "", k=k - 1, offset=offset, where={"k": k, "n": int(blk.K)})


def compile_features(specs, frame) -> list[Feature]:
    """A list of feature requests -> compiled features with the ids ``f1, f2, ...`` (at least 1 and at most 6)."""
    if not isinstance(specs, (list, tuple)) or not specs:
        raise AutoencoderError("Choose at least one feature to draw.")
    if len(specs) > 6:
        raise AutoencoderError("At most 6 features can be drawn at once.")
    return [compile_feature(s, frame, f"f{i + 1}") for i, s in enumerate(specs)]


def default_features(frame) -> list[dict]:
    """The features drawn without any choice by the person: the pixel (or wavelength) each code moves most and the overall intensity, of the first block with an EEM (else the first block)."""
    blk = next((b for b in frame.blocks if b.domain.kind == "surface"), frame.blocks[0])
    return [{"kind": "moved", "block": blk.name}, {"kind": "total", "block": blk.name}]
