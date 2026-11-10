"""Sphinx configuration of the ll-ml-triggerkit documentation.

Build:  pip install -r docs/requirements.txt && sphinx-build -b html docs docs/_build/html
"""
project = "ll-ml-triggerkit"
author = "Tanguy Dietrich"
copyright = "2026, Tanguy Dietrich"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "sphinx_togglebutton",
]

html_theme = "sphinx_rtd_theme"
exclude_patterns = ["_build", "refonte_82_bins.md"]    # the French working plan is not part of the site

# The API pages must build without a GPU stack: the heavy packages are replaced by mocks.
autodoc_mock_imports = ["tensorflow", "keras", "keras_hexagdly", "hgq", "hls4ml", "ctapipe", "dl1_data_handler",
                        "PyQt5", "pyqtgraph", "eventio", "protozfits", "setproctitle", "sklearn"]
autodoc_member_order = "bysource"
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

intersphinx_mapping = {"python": ("https://docs.python.org/3", None), "numpy": ("https://numpy.org/doc/stable/", None)}
