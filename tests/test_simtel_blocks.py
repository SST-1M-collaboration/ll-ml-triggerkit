"""``triggerkit.camera.simtel_blocks``: the eventio blocks the converter, the simtel opener and the viewer share."""
import numpy as np

from triggerkit.camera import simtel_blocks


def test_pedestal_is_pedestal_over_slices_in_the_stored_dtype():
    monitoring = {"pedestal": np.array([[8200.0, 16400.0, 100.0]], dtype=np.float32), "n_ped_slices": 82}
    ped = simtel_blocks.pedestal_per_sample(monitoring)
    assert ped.dtype == np.float32                       # not widened: the readers it replaces gave float32
    assert np.array_equal(ped, np.array([100.0, 200.0, 100.0 / 82], dtype=np.float32))


def test_photoelectron_block_is_indexed_by_tel_id_minus_one():
    block = {"pixel_id": np.array([3, 7], dtype=np.int32), "time": np.array([10.5, 11.0], dtype=np.float32)}
    event = {"photoelectrons": {0: block}}
    assert simtel_blocks.photoelectron_block(event, 1) is block
    assert simtel_blocks.photoelectron_block(event, 2) is None
    pixel_id, time = simtel_blocks.photoelectron_arrays(event, 1)
    assert pixel_id.tolist() == [3, 7] and time.dtype == np.float64
    pixel_id, time = simtel_blocks.photoelectron_arrays(event, 2)                   # nothing stored for this telescope
    assert pixel_id.size == 0 and time.size == 0
    assert simtel_blocks.photoelectron_arrays({}, 1)[0].size == 0                     # an event without the block at all
