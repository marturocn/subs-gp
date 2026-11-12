"""Wrapper entry-point to run the export_models_to_onnx_new.py script with default flags.

This module is used as a Poetry console script entrypoint so the user can run:

    poetry run toOnnx

and it will execute the equivalent of:

    poetry run python scripts/export_models_to_onnx_new.py --model all --fp16

Behavior:
 - If `scripts.export_models_to_onnx_new` can be imported and exposes a `main()` callable,
   it will be invoked with `sys.argv` set to include the default flags.
 - Otherwise it will fallback to invoking the script file with the current Python
   executable using subprocess.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path


def main() -> None:
    # prefer a newer script if present, fall back to the original
    module_names = [
        "scripts.export_models_to_onnx_new",
        "scripts.export_models_to_onnx",
    ]
    default_args = ["--model", "all", "--fp32", "--tensorrt"]

    # If the user passed arguments to the console entry, use them; otherwise use defaults
    passed_args = sys.argv[1:]
    args_to_use = passed_args if passed_args else default_args

    # Try to import the module and call its main() if available
    # Try import from available module names in order
    for name in module_names:
        try:
            mod = importlib.import_module(name)
            if hasattr(mod, "main"):
                old_argv = sys.argv[:]
                sys.argv = [name] + args_to_use
                try:
                    result = mod.main()
                    if isinstance(result, int):
                        raise SystemExit(result)
                    return
                finally:
                    sys.argv = old_argv
        except Exception:
            # try next module name
            continue

    # If import didn't work for any module, try to find an existing script file and run it
    script_dir = Path(__file__).parent
    candidate_files = ["export_models_to_onnx_new.py", "export_models_to_onnx.py"]
    for fname in candidate_files:
        script_path = script_dir / fname
        if script_path.exists():
            cmd = [sys.executable, str(script_path)] + args_to_use
            proc = subprocess.run(cmd)
            if proc.returncode:
                raise SystemExit(proc.returncode)
            return

    # If we get here, nothing was found/importable
    raise FileNotFoundError(
        "Ningún script disponible: se buscaron export_models_to_onnx_new.py y export_models_to_onnx.py"
    )


if __name__ == "__main__":
    main()
