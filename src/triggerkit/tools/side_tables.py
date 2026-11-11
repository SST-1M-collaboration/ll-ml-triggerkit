"""Tables streamed to a temporary file next to the output, copied into it at the end: for the simulation converter, whose ctapipe writer keeps the output file open all along.

The big per-event tables (the raw ``r0`` of 1296 pixels, the true signal per pixel) never sit in memory as a whole: ``append`` buffers a few rows and writes them, ``merge_into`` copies
the tables into the output and removes the temporary file. :mod:`triggerkit.tools.stages_table` does the same for the stages (it predates this module).
"""
import os

_FLUSH_ROWS = 16


class SideTables:
    """Rows of several tables, written to ``<output>.<tag>.tmp``; ``merge_into(output)`` copies them into ``output`` (their groups are created) and deletes the temporary file.

    ``append(group, name, description, row, title=...)`` adds one row to the table ``group/name`` (created from ``description``, a pytables description dict, at the first row)."""

    def __init__(self, output_file, tag):
        self.path = f"{output_file}.{tag}.tmp"
        self._h5 = None
        self._tables, self._rows = {}, {}

    def append(self, group, name, description, row, title=""):
        import tables

        if self._h5 is None:
            self._h5 = tables.open_file(self.path, "w")
        key = (group, name)
        if key not in self._tables:
            self._tables[key] = self._h5.create_table(group, name, description, createparents=True, title=title,
                                                      filters=tables.Filters(complevel=1, complib="zlib"))
            self._rows[key] = []
        self._rows[key].append(row)
        if len(self._rows[key]) >= _FLUSH_ROWS:
            self._flush(key)

    def _flush(self, key):
        if self._rows[key]:
            self._tables[key].append(self._rows[key])
            self._rows[key] = []

    def merge_into(self, output_file, skip=()):
        """Copy the tables into ``output_file`` (a pytables file that exists), except the ``(group, name)`` pairs of ``skip``, and remove the temporary file."""
        import tables

        if self._h5 is not None:
            for key in self._tables:
                self._flush(key)
            with tables.open_file(output_file, "a") as dest:
                for (group, name), table in self._tables.items():
                    if (group, name) in skip:
                        continue
                    parent = dest.get_node(group) if group in dest else dest.create_group(*_split(group), createparents=True)
                    table.copy(newparent=parent, newname=name)
        self.discard()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:                                  # the conversion failed: do not leave the temporary file behind
            self.discard()
        return False

    def discard(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
        if os.path.exists(self.path):
            os.remove(self.path)


def _split(group):
    parent, _, name = group.rstrip("/").rpartition("/")
    return parent or "/", name
