"""Files given as a list in a text file (``--to-process-list``, ``--simtel-list``, ``--real-list``).

One path per line; empty lines and lines starting with ``#`` are ignored; a relative path is relative to the directory the command is run from (as the paths given on the command
line). A path that does not exist is an error naming the line, so a typo does not become a short production.
"""
import os


def read_file_list(list_file):
    """The paths of ``list_file``, in order. Raises ``ValueError`` for an empty list and ``FileNotFoundError`` for a listed file that does not exist."""
    paths = []
    with open(list_file) as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if not os.path.exists(line):
                raise FileNotFoundError(f"{list_file}, line {number}: {line}: no such file")
            paths.append(line)
    if not paths:
        raise ValueError(f"{list_file} lists no file (one path per line, # for comments)")
    return paths


def join_files(files, list_file):
    """``files`` (given on the command line, or None) followed by the files of ``list_file`` (or None): the one list a command works on."""
    out = list(files or [])
    if list_file:
        out += read_file_list(list_file)
    return out
