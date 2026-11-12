from __future__ import annotations

import os
import subprocess
import sys


def main() -> None:
    # Read env vars
    from dotenv import load_dotenv
    load_dotenv()
    
    sonar_host = os.getenv("SONAR_HOST_URL")
    sonar_token = os.getenv("SONAR_TOKEN")

    if not sonar_host or not sonar_token:
        raise EnvironmentError(
            "Error: SONAR_HOST_URL y SONAR_TOKEN deben estar exportados "
            "antes de ejecutar este comando."
        )

    # If the user passed args, use them; otherwise use default
    default_args = ["--read-project-config"]
    passed_args = sys.argv[1:]
    args_to_use = passed_args if passed_args else default_args

    # Execute the scanner
    cmd = [sys.executable, "-m", "pysonar_scanner"] + args_to_use

    proc = subprocess.run(cmd)
    if proc.returncode:
        raise SystemExit(proc.returncode)
