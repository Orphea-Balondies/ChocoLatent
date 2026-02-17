#!/bin/bash
set -euo pipefail

# Usage:
#   bash scripts/setup_experiment_env.sh
# Optional:
#   PYTHON_BIN=python3.10 TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 bash scripts/setup_experiment_env.sh

PYTHON_BIN=${PYTHON_BIN:-python}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}

echo "[Info] Python: $PYTHON_BIN"
"$PYTHON_BIN" - <<'PY'
import sys
major, minor = sys.version_info[:2]
print(f"[Info] Detected Python {major}.{minor}")
if major != 3 or minor < 10 or minor > 13:
    raise SystemExit(
        "[Error] Recommend Python 3.10-3.13 for this project. "
        "Please create/activate a compatible environment first."
    )
PY

echo "[Step] upgrade pip/setuptools/wheel"
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

echo "[Step] install torch/torchvision from ${TORCH_INDEX_URL}"
"$PYTHON_BIN" -m pip install --upgrade torch torchvision --index-url "${TORCH_INDEX_URL}"

echo "[Step] install project requirements"
"$PYTHON_BIN" -m pip install -r requirements.txt

echo "[Step] install extra runtime deps for full pipeline"
"$PYTHON_BIN" -m pip install --upgrade lpips scikit-image accelerate transformers datasets peft safetensors

echo "[Step] optional FID backend (for --compute_fid)"
"$PYTHON_BIN" -m pip install --upgrade torchmetrics

echo "[Step] style transfer backend for glaze (onnx mosaic)"
"$PYTHON_BIN" -m pip install --upgrade onnxruntime

echo "[Done] environment setup finished."
