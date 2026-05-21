#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "== Environment =="
python --version
python - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10+ is required.")
if sys.version_info >= (3, 12):
    print("Warning: Python 3.12+ may need newer scvi-tools than requirements.txt pins.")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found; continuing on CPU."
fi

echo "== Install dependencies =="
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if [ -f "vendor/cobolt-0.0.1.zip" ]; then
  python -m pip install "vendor/cobolt-0.0.1.zip"
else
  python -m pip install "git+https://github.com/epurdom/cobolt.git"
fi

echo "== Run benchmark =="
export CMML3_N_CELLS="${CMML3_N_CELLS:-5000}"
export CMML3_N_GENES="${CMML3_N_GENES:-3000}"
export CMML3_N_PEAKS="${CMML3_N_PEAKS:-5000}"
export CMML3_MULTIVI_EPOCHS="${CMML3_MULTIVI_EPOCHS:-100}"
python notebooks/01_multivi_benchmark.py

echo "== Outputs =="
ls -lh figures tables results
