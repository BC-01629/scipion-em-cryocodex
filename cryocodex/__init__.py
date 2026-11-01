"""Scipion registration and installation of the isolated CryoCodex runtime."""

import os
import shlex
import shutil
import sys

import pwem
from pyworkflow import VarTypes
from pyworkflow.utils import Environ

from .constants import (
    BACKEND_VERSION, SOURCE_COMMIT, SOURCE_URL, SOURCE_BUILD_DIR, SOURCE_TAG, MODEL_URL, INSTALL_MARKER,
    CRYOCODEX_HOME, CRYOCODEX_PYTHON, CRYOCODEX_MODEL, CRYOCODEX_CONDA_EXE,
)

__version__ = "0.3.2"
_references = ["Cheng2026CryoCodex"]


class Plugin(pwem.Plugin):
    _homeVar = CRYOCODEX_HOME
    _pathVars = [CRYOCODEX_HOME]
    _supportedVersions = [BACKEND_VERSION]
    _url = "https://github.com/YangLab-SDU/CryoCodex"

    @classmethod
    def _defineVariables(cls):
        cls._defineEmVar(CRYOCODEX_HOME, "cryocodex-" + BACKEND_VERSION,
                         description="Directory containing CryoCodex predict.py.",
                         var_type=VarTypes.PATH)
        cls._defineVar(CRYOCODEX_PYTHON, "",
                       description="Optional Python 3.11 executable for an existing CryoCodex environment.",
                       var_type=VarTypes.PATH)
        cls._defineVar(CRYOCODEX_MODEL, "",
                       description="Optional existing CryoCodex.pth checkpoint.",
                       var_type=VarTypes.PATH)
        cls._defineVar(CRYOCODEX_CONDA_EXE,
                       os.environ.get("CONDA_EXE") or shutil.which("conda") or "conda",
                       description="Conda executable used to install the separate runtime.",
                       var_type=VarTypes.PATH)

    @classmethod
    def getPython(cls):
        return os.path.abspath(os.path.expanduser(
            cls.getVar(CRYOCODEX_PYTHON) or cls.getHome("conda-env", "bin", "python")))

    @classmethod
    def getModel(cls):
        return os.path.abspath(os.path.expanduser(
            cls.getVar(CRYOCODEX_MODEL) or cls.getHome("model_state_dicts", "CryoCodex.pth")))

    @classmethod
    def getProgram(cls):
        return "%s -u %s" % (shlex.quote(cls.getPython()),
                               shlex.quote(os.path.abspath(cls.getHome("predict.py"))))

    @classmethod
    def getEnviron(cls):
        env = Environ(os.environ)
        for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"):
            env.pop(name, None)
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env.update({"PATH": os.path.dirname(cls.getPython())}, position=Environ.BEGIN)
        return env

    @classmethod
    def defineBinaries(cls, env):
        installer = os.path.join(os.path.dirname(__file__), "scripts", "install_runtime.py")
        command = " ".join(shlex.quote(x) for x in (
            sys.executable, installer, "--home", ".", "--conda",
            cls.getVar(CRYOCODEX_CONDA_EXE), "--model-url", MODEL_URL,
            "--source-commit", SOURCE_COMMIT,
            "--source-tag", SOURCE_TAG,
        ))
        if cls.getVar(CRYOCODEX_MODEL):
            command += " --model-file " + shlex.quote(cls.getModel())
        env.addPackage(
            "cryocodex", version=BACKEND_VERSION,
            tar="cryocodex-%s-release.tar.gz" % BACKEND_VERSION,
            url=SOURCE_URL, buildDir=SOURCE_BUILD_DIR,
            commands=[(command, INSTALL_MARKER)],
            neededProgs=["wget"], default=True,
        )

    @classmethod
    def validateInstallation(cls):
        errors = []
        for path, description in (
            (cls.getHome("predict.py"), "CryoCodex source"),
            (cls.getPython(), "CryoCodex Python 3.11 runtime"),
            (cls.getModel(), "CryoCodex checkpoint"),
        ):
            if not os.path.isfile(path) or os.path.getsize(path) == 0:
                errors.append("Missing %s: %s" % (description, path))
        if os.path.isfile(cls.getPython()) and not os.access(cls.getPython(), os.X_OK):
            errors.append("CryoCodex Python is not executable: " + cls.getPython())
        if errors:
            errors.append("Run 'scipion3 installb cryocodex-%s', or configure "
                          "CRYOCODEX_HOME, CRYOCODEX_PYTHON and CRYOCODEX_MODEL "
                          "for an existing installation." % BACKEND_VERSION)
        return errors
