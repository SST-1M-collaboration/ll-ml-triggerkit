Reports
=======

A report compares several statistics files: efficiency against the number of photoelectrons and against the energy, rate against threshold, ROC curve,
effective area, the cross-validation folds, ... It is made by :class:`triggerkit.Statistics.StatPlotter.StatPlotter` (matplotlib), as a PDF, a Markdown
file (always written, next to the plots) and, if asked, an HTML page. Every figure is also saved as a PNG and an SVG in the output folder.

.. code-block:: python

   from triggerkit.Statistics.StatPlotter import StatPlotter

   plotter = StatPlotter(base_reference_config=CONFIG_PATCH7, stat_folder=STAT_FOLDERS, fold="rot0_original_medium")
   path = plotter.generateReport(
       configs=[(CONFIG_PATCH7, "rot0_original_medium"), (CONFIG_TDSCAN, "rot0_original_medium")],
       output_dir="reports/my_report", target_rate_hz=50_000.0,
       legend_overrides=["PATCH7", "TDSCAN"], generate_pdf=True, generate_html=False,
       title="SST-1M trigger statistics", cross_validation_condition="medium", effective_area_expected_N=None)

What the pieces are
-------------------

``stat_folder``
   a folder, or a list of folders, with the ``*.h5`` statistics files.
``configs``
   the chains to plot, as ``(config, fold)``. A *config* is the list of ``(stage, parameters)`` that describes the chain, exactly as it is stored in the file
   (``trigger_chain_json``). The report finds the file whose chain matches. For patch7::

      [("digital_sum", {"mode": "patch7", "threshold_flower": None}),
       ("threshold", {"threshold": 222.0, "binary": True, "comparison": "gt"})]

   The CNNs all end with the same threshold stage, so their config is **pinned to the run by its UUID** (a prefix of at least 8 characters is enough)::

      [("threshold", {"threshold": 0.0, "binary": True, "comparison": "gt", "uuid": StatsFile(path).uuid})]

``target_rate_hz``
   the NSB rate every chain is moved to before it is compared: the threshold of each one is read from its own scores on the reference fold. The thresholds written
   in the files are not used.
``legend_overrides``
   one label per config (``None`` keeps the automatic one).
``common_folds_only``
   (default ``True``) drops every fold that is missing from one file or ran a different sample there, and lists what was dropped. Use ``False`` when the files
   have different NSB, as for a cascade (``tdscan50k`` against the triggerless one).
``cross_validation_condition``
   the condition whose folds are drawn in the cross-validation sections (here ``"medium"``).
``generate_pdf``, ``generate_html``, ``save_svg``, ``vector_plots``
   the output formats.

Sections of the report
----------------------

Configurations; NSB trigger rate against threshold (and normalised); gamma efficiency against the NSB rate; effective area; absolute efficiency and ratio against
``n_pe`` and against the energy, for gammas, protons and muons; ``n_pe`` and energy distributions; ROC curve (gamma against NSB); the cross-validation of each
chain and of all of them (the deviation from the reference fold, in sigma); ``n_pe`` against energy; gamma efficiency against the impact distance; the efficiency
maps over the camera; one independent rate-against-threshold scan per configuration.

Two typical reports
-------------------

* **patch7, TDSCAN and CNNs at 50 kHz**: one config per chain, ``target_rate_hz=50_000``, the reference fold of every file, the CNN configs pinned by UUID
  as above.
* **A cascade, at its OUTPUT rate**: patch7 and the TDSCAN alone are brought to the target by their own threshold, the CNN of the cascade by its own (the
  TDSCAN in front stays at its 50 kHz threshold), e.g. ``target_rate_hz=7_000``; the CNN with no TDSCAN in front is added for reference. The NSB differs
  (the one the TDSCAN fired on for the cascade, the triggerless one for the others), so give ``common_folds_only=False``.

.. code-block:: python

   import glob, os
   from triggerkit.Statistics.stats_file import StatsFile

   CONFIG_PATCH7 = [("digital_sum", {"mode": "patch7", "threshold_flower": None}),
                    ("threshold", {"threshold": 222.0, "binary": True, "comparison": "gt"})]
   CONFIG_TDSCAN = [("score_quantizer", {"edges": [16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0, 72.0, 80.0, 88.0, 96.0, 104.0, 112.0, 120.0, 128.0]}),
                    ("tdscan", {"eps_xy": 1, "eps_t": 2, "filters": 1, "share_neighbors": True,
                                "ring_weights": [[0.5, -0.0078], [-0.5, -0.0312], [-0.5, -0.125], [0.5, 0.0625], [0.5, 0.25]]}),
                    ("threshold", {"threshold": 8.109374046325684, "binary": True, "comparison": "gt"})]

   def cnn_config(folder):
       """The threshold config of the run in ``folder``, pinned to it by its UUID."""
       uuid = StatsFile(sorted(glob.glob(os.path.join(folder, "*.h5")))[0]).uuid
       return [("threshold", {"threshold": 0.0, "binary": True, "comparison": "gt", "uuid": uuid})]

   plotter = StatPlotter(base_reference_config=CONFIG_PATCH7, stat_folder=["stats/patch7", "stats/tdscan", "stats/hexcnn", "stats/hexcnn_cascade"],
                         fold="rot0_original_medium")
   plotter.generateReport(
       configs=[(config, "rot0_original_medium") for config in (CONFIG_PATCH7, CONFIG_TDSCAN, cnn_config("stats/hexcnn"), cnn_config("stats/hexcnn_cascade"))],
       output_dir="reports/cascade_7khz", target_rate_hz=7_000.0,
       legend_overrides=["PATCH7", "TDSCAN", "HexCNN, no TDSCAN in front", "TDSCAN + HexCNN (cascade)"],
       generate_pdf=True, generate_html=False, title="Cascade TDSCAN + HexCNN, 7 kHz of NSB at the output",
       cross_validation_condition="medium", effective_area_expected_N=None, common_folds_only=False)

A config is the chain as the file stores it: print ``StatsFile(path).trigger_chain`` to copy it. The report is written to ``output_dir`` (``report.pdf``, ``report.md`` and the figures).

Compare files made with the same reading order and the same gamma files, as the report compares them event by event.
