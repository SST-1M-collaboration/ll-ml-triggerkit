"""Automatic analysis of a real SST-1M acquisition where patch7 and the TDSCAN were on together: trigger rates, NSB, noise + physics fits, ROC curves, patch7 against TDSCAN.

    triggerkit-night-report FILES_OR_FOLDER --aux SLOW_CONTROL_FOLDER --output OUT_FOLDER

One module per stage: :mod:`.reader` (the only one that reads the files: protozfits, astropy, and sst1mpipe for the NSB, which stays optional), :mod:`.period` (event times, thresholds, which
algorithms were on), :mod:`.rates` (the fits), :mod:`.figures`, :mod:`.report` (the command and the PDF). Nothing here imports TensorFlow or ctapipe.
"""
