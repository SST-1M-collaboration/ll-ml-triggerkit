"""Protons and muons as extra negatives of the training stream, and the per-epoch rates of every type."""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
from tensorflow import keras

from synthetic import N_PIX, write_ctapipe_h5
from triggerkit.data import NsbSet, Simulation, TriggerDataset
from triggerkit.training import NpeWeighting, ParticleRates


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    d = tmp_path_factory.mktemp("neg")
    gamma = [write_ctapipe_h5(d / f"g{i}.h5", n_events=10, seed=i) for i in range(4)]
    proton = [write_ctapipe_h5(d / f"p{i}.h5", n_events=10, seed=10 + i) for i in range(4)]
    muon = [write_ctapipe_h5(d / f"m{i}.h5", n_events=10, kind="nsb", seed=20 + i) for i in range(2)]
    nsb = [write_ctapipe_h5(d / "n.h5", n_events=30, kind="nsb", seed=30)]
    return Simulation(gamma=gamma, proton=proton, muon=muon, nsb=nsb)


def dataset(sim, **kw):
    kw.setdefault("train_on", ("gamma", "nsb", "proton"))
    return TriggerDataset(sim, batch_size=8, tel_id_only=1, percent_validation=0.5, load_ram=False, **kw)


def read(ds, *keys):
    out = {k: [] for k in ("label", *keys)}
    for features, label in ds:
        out["label"].append(label.numpy().reshape(-1))
        for k in keys:
            out[k].append(features[k].numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def test_protons_are_negatives_with_their_own_code_and_weight(tree):
    train, val = dataset(tree, negative_weights={"proton": 0.25}).train_val_datasets()
    got = read(train, "particle", "particle_weight")
    assert set(got["label"]) == {0, 1} and set(got["particle"]) == {0, 1, 2}
    np.testing.assert_array_equal(got["label"], (got["particle"] == 1).astype(int))             # gamma against everything else
    np.testing.assert_allclose(got["particle_weight"][got["particle"] == 2], 0.25)
    np.testing.assert_allclose(got["particle_weight"][got["particle"] != 2], 1.0)
    assert 2 in set(read(val, "particle")["particle"])                                          # the protons have a validation split too


def test_the_extra_files_are_split_like_the_gamma_files(tree):
    ds = dataset(tree, train_on=("gamma", "nsb", "proton", "muon"))
    split = ds.split_extra_files()
    assert split["proton"] == (tree.files("proton")[:2], tree.files("proton")[2:])
    assert split["muon"] == (tree.files("muon")[:1], tree.files("muon")[1:])
    assert set(split) == {"proton", "muon"} and ds.split_files()[1] == tree.files("gamma")[2:]


def test_the_negative_weight_reaches_the_npe_channels(tree):
    ds = dataset(tree, negative_weights={"proton": 0.25}, npe_weighting=NpeWeighting(30, 50, below="negative"))
    train, _ = ds.train_val_datasets()
    got = read(train, "particle", "npe_channels")
    channels, particle = got["npe_channels"], got["particle"]
    np.testing.assert_allclose(channels[particle == 2][:, 2], 0.25)       # a proton: NSB-like negative, weight 0.25
    np.testing.assert_allclose(channels[particle == 0][:, 2], 1.0)        # the NSB keeps 1
    assert (channels[particle == 1][:, 2] < 1.0).all()                    # the gammas are untouched (1 - w of the ramp)


def test_nothing_changes_without_extra_types(tree):
    train, _ = dataset(tree, train_on=("gamma", "nsb")).train_val_datasets()
    got = read(train, "particle")
    assert set(got["particle"]) == {0, 1} and "particle_weight" not in next(iter(train))[0]


def test_extra_training_types_are_checked(tree):
    with pytest.raises(ValueError, match="'nsb'"):
        dataset(tree, train_on=("gamma", "proton"))
    with pytest.raises(ValueError, match="no such source"):
        dataset(Simulation(gamma=tree.files("gamma"), nsb=tree.files("nsb")), train_on=("gamma", "nsb", "muon"))
    with pytest.raises(ValueError, match="negative_weights"):
        dataset(tree, negative_weights={"muon": 0.5})
    with pytest.raises(ValueError, match=">= 0"):
        dataset(tree, negative_weights={"proton": -1.0})
    capped = dataset(tree, max_extra_samples_train=6)
    assert (read(capped.train_val_datasets()[0], "particle")["particle"] == 2).sum() == 6


# --------------------------------------------------------------------- ParticleRates
class Linear(keras.Model):
    """score = the single input feature: the cut is then the NSB quantile of the feature itself."""
    def call(self, x, training=False):
        return {"trigger": x[:, :1]}


def stream(values, code):
    x = np.asarray(values, dtype="float32")[:, None]
    return tf.data.Dataset.from_tensor_slices((x, {"particle": np.full(len(x), code)})).batch(50)


def test_rates_are_taken_at_the_cut_that_keeps_a_fraction_of_the_nsb(capsys):
    rng = np.random.default_rng(0)
    nsb, gamma = rng.normal(0, 1, 1000), rng.normal(3, 1, 1000)
    val = stream(np.r_[nsb, gamma], 0).unbatch().take(1000).concatenate(stream(gamma, 1).unbatch()).batch(100)
    watch = {"proton": stream(rng.normal(1, 1, 1000), 2), "muon": stream(rng.normal(5, 1, 100), 3)}
    model = Linear()
    model(np.zeros((1, 1), "float32"))
    cb = ParticleRates(val, watch, nsb_fraction=0.05, batches=100)
    cb.set_model(model)
    logs = {}
    cb.on_epoch_end(0, logs)
    cut = np.quantile(nsb, 0.95)
    assert logs["val_cut"] == pytest.approx(cut)
    assert logs["val_nsb_at_nsb"] == pytest.approx(0.05, abs=0.002)
    assert logs["val_gamma_at_nsb"] == pytest.approx(float((gamma > cut).mean()), abs=1e-9)
    assert 0.2 < logs["val_proton_at_nsb"] < 0.35 and logs["val_muon_at_nsb"] > 0.95
    assert "PARTICLES epoch=1" in capsys.readouterr().out


def test_rates_without_nsb_do_nothing_and_check_the_fraction(capsys):
    cb = ParticleRates(stream([1.0, 2.0], 1))
    cb.set_model(Linear())
    cb.on_epoch_end(0, {})
    assert "no NSB event" in capsys.readouterr().out
    with pytest.raises(ValueError, match="nsb_fraction"):
        ParticleRates(stream([1.0], 0), nsb_fraction=1.5)


def test_rates_skip_a_batch_every_event_of_which_was_dropped(capsys):
    full = stream(np.r_[np.random.default_rng(1).normal(0, 1, 100)], 0)
    empty = tf.data.Dataset.from_tensor_slices((np.zeros((0, 1), "float32"), {"particle": np.zeros(0, dtype=int)})).batch(10)
    model = Linear()
    model(np.zeros((1, 1), "float32"))
    cb = ParticleRates(full, {"proton": empty})                              # a watched source the upstream trigger dropped entirely
    cb.set_model(model)
    logs = {}
    cb.on_epoch_end(0, logs)
    assert np.isnan(logs["val_proton_at_nsb"]) and "PARTICLES epoch=1" in capsys.readouterr().out
