#!/usr/bin/env bash
# Installs all dependencies for the epicycle system into the `manim` conda env.
#
# This machine is aarch64, where manimpango ships no wheel for python 3.13. A pure-pip
# install therefore tries to compile it and fails on the missing pangocairo C library,
# so manim and everything else with native code comes from conda-forge instead. Only
# vtracer (optional --via-svg vectorizer) is pip; it has an ARM wheel.
set -euo pipefail

ENV=manim
PY=/home/ashwin/miniconda3/envs/$ENV/bin/python
PIP=/home/ashwin/miniconda3/envs/$ENV/bin/pip

conda install -n "$ENV" -c conda-forge -y manim ffmpeg py-opencv numpy pyyaml svgelements

"$PIP" install vtracer

"$PY" - <<'EOF'
import manim, cv2, numpy, yaml, svgelements, vtracer, manimpango
print("All dependencies OK")
print("  manim       :", manim.__version__)
print("  cv2         :", cv2.__version__)
print("  numpy       :", numpy.__version__)
print("  svgelements :", svgelements.SVGELEMENTS_VERSION)
EOF
