"""Who ran what, when, with which software: what a statistics file carries so that it can be identified and reproduced."""
import datetime
import getpass
import os
import platform
import socket
import subprocess
import sys
import uuid
from importlib import metadata


def new_uuid() -> str:
    """A fresh run identifier (32 hexadecimal characters)."""
    return uuid.uuid4().hex


def _version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return ""


def _git_commit():
    """Commit of the triggerkit checkout this code runs from, ``""`` if it is not one (an installed copy) or git is missing."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        out = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "-C", here, "status", "--porcelain", "--untracked-files=no", here], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return ""
    if out.returncode != 0:
        return ""
    return out.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")


def run_info() -> dict:
    """The facts of a run, all strings: date and time (UTC and local), machine, user, command line, software versions."""
    now = datetime.datetime.now(datetime.timezone.utc)
    return {
        "created_utc": now.isoformat(timespec="seconds"),
        "created_local": now.astimezone().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "user": getpass.getuser() if os.environ.get("USER") or os.environ.get("LOGNAME") else "",
        "argv": " ".join(sys.argv),
        "python": platform.python_version(),
        "triggerkit": _version("ll-ml-triggerkit"),
        "git_commit": _git_commit(),
        "tensorflow": _version("tensorflow"),
        "keras": _version("keras"),
        "numpy": _version("numpy"),
    }
