"""Hatch build hook — generates protobuf stubs before packaging.

This ensures the wheel on PyPI contains the compiled *_pb2.py files
even though they are gitignored. Users install via pip and everything
works without needing grpcio-tools at runtime.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class ProtobufBuildHook(BuildHookInterface):
    PLUGIN_NAME = "protobuf"

    def initialize(self, version: str, build_data: dict) -> None:  # type: ignore[type-arg]
        root = Path(self.root)
        proto_dir = root / "macaw" / "proto"

        protos = list(proto_dir.glob("*.proto"))
        if not protos:
            return

        # Generate Python stubs from .proto files
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "grpc_tools.protoc",
                f"-I{proto_dir}",
                f"--python_out={proto_dir}",
                f"--grpc_python_out={proto_dir}",
                f"--pyi_out={proto_dir}",
                *[str(p) for p in protos],
            ],
        )

        # Fix absolute imports → relative imports in *_grpc.py files
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
            grpc_file.write_text(content)

        # Force-include generated files in the wheel
        for pattern in ("*_pb2.py", "*_pb2_grpc.py", "*_pb2.pyi"):
            for f in proto_dir.glob(pattern):
                rel = f.relative_to(root)
                build_data["force_include"][str(f)] = str(rel)
