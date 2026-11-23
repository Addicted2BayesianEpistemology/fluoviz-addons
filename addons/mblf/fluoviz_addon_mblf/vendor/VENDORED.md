# Vendored: mblf-python 1.0.0

* Source: https://github.com/Addicted2BayesianEpistemology/mblf-python, tag `v1.0.0`, commit `c83ddb629cdd8971a41eddb037311c64b0ab37e3`.
* Licence: MIT (`mblf/LICENSE`).
* Copied unchanged (`mblf/*.py`, `mblf/stan/*.stan`).  Do not edit these files: change the add-on instead, or update the copy from a new release of mblf-python and record the commit here.

Why a copy: FluoViz add-ons can only ask for conda packages (`pip` is not supported, see FluoViz `docs/modules/addons.md`), and mblf is not on conda-forge.  The Stan engine itself
(CmdStan, cmdstanpy, xarray and a C++ compiler) is NOT part of the add-on: it lives in a private environment that the page prepares once (`core/engine.py`).

To update: copy `mblf/` of the new release over this folder, update the commit above, run the tests of the add-on (`tests/test_vendor.py` checks that the copy imports without
cmdstanpy and that the Stan programs are present).
