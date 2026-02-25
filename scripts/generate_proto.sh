#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTO_DIR="$ROOT_DIR/macaw/proto"
PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" -m grpc_tools.protoc \
  -I "$PROTO_DIR" \
  --python_out="$PROTO_DIR" \
  --grpc_python_out="$PROTO_DIR" \
  --pyi_out="$PROTO_DIR" \
  "$PROTO_DIR/stt_worker.proto" \
  "$PROTO_DIR/tts_worker.proto" \
  "$PROTO_DIR/vc_worker.proto"

PROTO_DIR="$PROTO_DIR" "$PYTHON_BIN" - <<'PY'
from __future__ import annotations

from pathlib import Path
import os

proto_dir = Path(os.environ["PROTO_DIR"]).resolve()

# Fix absolute imports → relative imports in all *_grpc.py files
for grpc_file in proto_dir.glob("*_pb2_grpc.py"):
    content = grpc_file.read_text()
    content = content.replace(
        "import stt_worker_pb2 as stt__worker__pb2",
        "from . import stt_worker_pb2 as stt__worker__pb2",
    )
    content = content.replace(
        "import tts_worker_pb2 as tts__worker__pb2",
        "from . import tts_worker_pb2 as tts__worker__pb2",
    )
    content = content.replace(
        "import vc_worker_pb2 as vc__worker__pb2",
        "from . import vc_worker_pb2 as vc__worker__pb2",
    )
    grpc_file.write_text(content)
PY
