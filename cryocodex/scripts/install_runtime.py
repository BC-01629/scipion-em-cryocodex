"""Install the backend into a private Conda prefix; invoked by Scipion installb.

This file intentionally uses only the Python standard library. It can run in
Scipion's Python 3.8 while installing the inference environment with Python 3.11.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request


def isolated_environment():
    env = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"):
        env.pop(name, None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def download_model(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as output:
                if "text/html" in response.headers.get("Content-Type", "").lower():
                    raise RuntimeError("Model URL returned an HTML page instead of a checkpoint.")
                shutil.copyfileobj(response, output)
                expected = response.headers.get("Content-Length")
            size = partial.stat().st_size
            if size == 0 or (expected is not None and size != int(expected)):
                raise RuntimeError("Incomplete model download; retry the installation.")
            partial.replace(destination)
            return
        except Exception:
            if partial.exists():
                partial.unlink()
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def install(home, conda, model_url, source_commit, model_file=None, source_tag=None):
    home = Path(home).resolve()
    for name in ("predict.py", "requirements.txt", "VQUnet.py"):
        if not (home / name).is_file():
            raise RuntimeError("Incomplete CryoCodex source: missing " + name)
    env = isolated_environment()
    executable = shutil.which(conda)
    if executable is None:
        raise RuntimeError("Conda executable not found: %s. Set CRYOCODEX_CONDA_EXE." % conda)
    prefix = home / "conda-env"
    python = prefix / "bin" / "python"
    if not python.is_file():
        subprocess.run([executable, "create", "--yes", "--prefix", str(prefix),
                        "python=3.11", "pip"], check=True, env=env)
    subprocess.run([str(python), "-c", "import sys; assert sys.version_info[:2] == (3, 11), "
                    "'CryoCodex requires Python 3.11'"], check=True, env=env)
    subprocess.run([str(python), "-m", "pip", "install", "-r", str(home / "requirements.txt")],
                   check=True, env=env)
    model = home / "model_state_dicts" / "CryoCodex.pth"
    if model_file:
        source = Path(model_file).expanduser().resolve()
        if not source.is_file() or source.stat().st_size == 0:
            raise RuntimeError("Model file is missing or empty: " + str(source))
        model.parent.mkdir(parents=True, exist_ok=True)
        if source != model.resolve():
            partial = model.with_suffix(".pth.part")
            shutil.copyfile(source, partial)
            partial.replace(model)
    elif not model.is_file() or model.stat().st_size == 0:
        print("Downloading CryoCodex model from " + model_url, flush=True)
        download_model(model_url, model)
    # Verify dependencies and checkpoint before recording a successful install.
    subprocess.run([str(python), str(home / "predict.py"), "--version"], check=True, env=env)
    subprocess.run([str(python), "-c",
                    "import sys, torch; c=torch.load(sys.argv[1],map_location='cpu',weights_only=True); "
                    "assert isinstance(c,dict) and 'model_state_dict' in c, 'Invalid CryoCodex checkpoint'",
                    str(model)], check=True, env=env)
    digest = hashlib.sha256()
    with model.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    record = {"source_commit": source_commit, "source_tag": source_tag, "python": str(python),
              "model_sha256": digest.hexdigest()}
    temporary = home / "cryocodex-installed.json.part"
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(home / "cryocodex-installed.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", required=True)
    parser.add_argument("--conda", required=True)
    parser.add_argument("--model-url", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tag")
    parser.add_argument("--model-file")
    args = parser.parse_args()
    install(**vars(args))


if __name__ == "__main__":
    main()
