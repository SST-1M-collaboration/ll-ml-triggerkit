Welcome to ll-ml-triggerkit's documentation!
============================================

.. note::
   This documentation is under construction: the pages are written section by section from the README and the code.

``ll-ml-triggerkit`` (import name ``triggerkit``) trains, evaluates and exports **low-level trigger chains** for the SST-1M telescope
(DigiCam camera): the FADC, the patch7 sum, the TDSCAN, and trainable models (hexagonal CNNs, quantized for the FPGA) that replace or follow them.
It also converts sim_telarray simulations and real SST-1M data to HDF5, merges simulated gammas into real noise, and shows the events in a viewer.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   concepts
   conversion
   merge
   dataset
   statistics
   statistics_examples
   report
   statistics_checks
   viewer
   api
