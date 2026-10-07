# legacy_mfego_initial

Frozen copy of the `mfego` framework **before** the theoretical/numerical review carried out on the
branch `fix/analyse-theorique-mfego` (source commit `adc0b7f` of `main`).

Content:

* `mfego/main.py` and `mfego/src/*.py`: initial source code of the framework.
* `example/hartmann_6d/*.py`, `example/hydrofoil_optim/*.py`: initial example scripts.
* `README_initial.md`: initial top-level README.

This copy is kept for traceability only (it is used by `analysis/` scripts to measure the
"before/after" effect of each fix). It is **not** maintained: see `analysis/RAPPORT_ANALYSE.md`
for the list of issues found in it and the corresponding fixes (`# [FIX-<ID>]` tags) in `mfego/`.

Note: this initial version only imports on Python >= 3.14 with `traitlets` installed (see report,
fix R6), because of `from traitlets import List, Tuple` in `src/data_management.py`.
