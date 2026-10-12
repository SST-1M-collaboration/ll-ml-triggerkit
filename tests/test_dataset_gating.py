"""TriggerDataset(gated_by=...): train a downstream cascade stage only on what the upstream fires on."""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from synthetic import N_PIX, write_ctapipe_h5
from triggerkit.TriggerChain import TriggerChain
from triggerkit.data import TriggerDataset


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("gating")
    gamma = [write_ctapipe_h5(d / f"g{i}.h5", n_events=20, seed=i) for i in range(3)]
    nsb = [write_ctapipe_h5(d / "n.h5", n_events=60, kind="nsb", seed=10)]
    return gamma, nsb


def patch7(files, tau, num_samples=None):
    gamma, nsb = files
    chain = TriggerChain(gamma, simtel_nsb_path=nsb, num_samples=num_samples)
    chain.add_stage("digital_sum", mode="patch7")
    chain.add_stage("global_max_pooling_2d")
    chain.add_stage("threshold", init_tau=tau, temp=1.0, binary_output=True)
    chain.compile_chain()
    return chain


def events(ds):
    wf, ped, lab = [], [], []
    for features, label in ds:
        wf.append(features["waveform"].numpy().reshape(-1, N_PIX, 50))
        ped.append(features["pedestal"].numpy().reshape(-1, N_PIX))
        lab.append(label.numpy().reshape(-1))
    return np.concatenate(wf), np.concatenate(ped), np.concatenate(lab)


def dataset(files, **kw):
    gamma, nsb = files
    return TriggerDataset(gamma, nsb, batch_size=8, tel_id_only=1, percent_validation=0.34,
                          load_ram=False, **kw)


def test_only_the_events_the_gate_fires_on_are_kept(files):
    gate = patch7(files, tau=750.0)           # the synthetic pulses pass, the NSB baseline does not
    gated = dataset(files, gated_by=gate)
    train_ds, val_ds = gated.train_val_datasets()
    wf, ped, lab = events(train_ds)
    assert gated.gate_mask(wf, ped).all()                       # nothing the gate rejects
    assert {int(label.shape[0]) for _f, label in train_ds} == {8}   # full batches only

    raw_wf, raw_ped, raw_lab = events(dataset(files).train_val_datasets()[0])
    fired = gated.gate_mask(raw_wf, raw_ped)
    n_fired = (fired & (raw_lab == 1)).sum()
    assert n_fired - 8 < (lab == 1).sum() <= n_fired and n_fired > 0  # the fired gammas, minus the dropped remainder
    assert (lab == 0).sum() < (raw_lab == 0).sum()                   # NSB thinned out by the gate
    assert events(val_ds)[2].size > 0


def test_gates_must_be_compiled_full_window_chains(files):
    gamma, nsb = files
    with pytest.raises(ValueError, match="compile_chain"):
        dataset(files, gated_by=TriggerChain(gamma, simtel_nsb_path=nsb))
    with pytest.raises(ValueError, match="whole datacube"):
        dataset(files, gated_by=patch7(files, 750.0, num_samples=32))
