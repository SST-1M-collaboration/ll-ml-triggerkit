"""``Simulation`` (the files of each event type and their totals) and ``TriggerDataset`` reading several types."""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from synthetic import N_PIX, write_ctapipe_h5
from triggerkit.data import PARTICLES, NsbSet, Simulation, Source, TriggerDataset, Watch, read_n_simulated


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    d = tmp_path_factory.mktemp("sim")
    for name, n_files, run in (("gammas", 3, (300, 20)), ("protons", 2, (200, 20)), ("muons", 2, (100, 1))):
        (d / name).mkdir()
        for i in range(n_files):
            write_ctapipe_h5(d / name / f"{name[:-1]}{i}.h5", n_events=12, seed=i + len(name), run=run)
    (d / "nsb").mkdir()
    write_ctapipe_h5(d / "nsb" / "n.h5", n_events=40, kind="nsb", seed=10)
    return d


def simulation(tree, **kw):
    return Simulation(gamma=str(tree / "gammas" / "*.h5"), proton=str(tree / "protons" / "*.h5"),
                      muon=str(tree / "muons" / "*.h5"), nsb=NsbSet(str(tree / "nsb" / "*.h5")), **kw)


def test_particle_codes_are_fixed():
    assert PARTICLES == {"nsb": 0, "gamma": 1, "proton": 2, "muon": 3}


def test_patterns_are_expanded_and_sorted(tree):
    sim = simulation(tree)
    assert [p.split("/")[-1] for p in sim.files("gamma")] == ["gamma0.h5", "gamma1.h5", "gamma2.h5"]
    assert len(sim.files("proton")) == 2 and sim.files("nsb")[0].endswith("n.h5")
    assert Simulation(gamma=sim.files("gamma"), nsb=sim.files("nsb")).files("gamma") == sim.files("gamma")   # lists too


def test_a_pattern_matching_nothing_is_an_error(tree):
    with pytest.raises(ValueError, match="matches no file"):
        Simulation(gamma=str(tree / "nothing" / "*.h5"))


def test_unknown_or_missing_sources(tree):
    sim = simulation(tree)
    with pytest.raises(KeyError):
        sim["electron"]
    assert sim.files("muon") and "proton" in sim
    only_gamma = Simulation(gamma=sim.files("gamma"))
    assert only_gamma.files("proton") == [] and "proton" not in only_gamma
    with pytest.raises(KeyError, match="no 'proton' source"):
        only_gamma["proton"]
    with pytest.raises(ValueError):
        Simulation()
    with pytest.raises(ValueError, match="source of 'proton'"):
        Simulation(gamma=Source("proton", sim.files("proton")))


def test_totals_are_read_from_the_run_tables(tree):
    sim = simulation(tree)
    assert read_n_simulated(sim.files("gamma")) == 3 * 300 * 20
    assert sim.simulated("proton") == 2 * 200 * 20
    assert sim.simulated("muon") == 2 * 100 * 1
    assert Source("gamma", sim.files("gamma"), n_simulated=5).simulated() == 5                 # given by hand wins


def test_a_file_without_a_run_table_asks_for_the_total(tree):
    with pytest.raises(KeyError, match="give n_simulated"):
        simulation(tree).simulated("nsb")


def test_aligned_nsb_total_is_given_not_read(tree):
    nsb = NsbSet(str(tree / "nsb" / "n.h5"), align="given", at=22, n_simulated=12_500_000,
                 made_by=dict(trigger="tdscan", threshold=8.109374))
    assert nsb.simulated() == 12_500_000 and nsb.at == 22 and nsb.made_by["trigger"] == "tdscan"
    with pytest.raises(ValueError, match="give n_simulated"):
        NsbSet(str(tree / "nsb" / "n.h5"), align="given", at=22).simulated()


@pytest.mark.parametrize("kw", [dict(align="given"),                      # no sample
                                dict(at=22),                              # a sample without the mode
                                dict(align="given", at=-1),
                                dict(align="sideways"),
                                dict(n_simulated=0),
                                dict(made_by={"trigger": "tdscan"})])     # made_by only describes an aligned set
def test_incoherent_alignment_is_refused(tree, kw):
    with pytest.raises(ValueError):
        NsbSet(str(tree / "nsb" / "n.h5"), **kw)


def test_watch_validation():
    assert Watch("proton").label == "proton" and Watch("muon", name="mu").label == "mu"
    with pytest.raises(ValueError):
        Watch("nsb")
    assert Watch("proton", align="none", window=object()).align == "none"
    with pytest.raises(ValueError, match="align"):
        Watch("proton", align="sideways")


def dataset(sim, **kw):
    return TriggerDataset(sim, batch_size=8, tel_id_only=1, percent_validation=0.34, load_ram=False, **kw)


def collect(ds):
    particle, label = [], []
    for features, lab in ds:
        assert features["waveform"].numpy().reshape(-1, N_PIX, 50).shape[0] == features["particle"].shape[0]
        particle.append(features["particle"].numpy())
        label.append(lab.numpy().reshape(-1))
    return np.concatenate(particle), np.concatenate(label)


def test_training_events_carry_their_particle_code(tree):
    train, val = dataset(simulation(tree)).train_val_datasets()
    for stream in (train, val):
        particle, label = collect(stream)
        assert set(particle) == {0, 1} and (particle == label).all()        # gamma 1, nsb 0: the label of the old stream


def test_watched_sources_are_read_apart_and_carry_their_code(tree):
    ds = dataset(simulation(tree), watch=["proton", Watch("muon", name="mu")])
    watched = ds.watch_datasets()
    assert set(watched) == {"proton", "mu"}
    proton, _ = collect(watched["proton"])
    muon, _ = collect(watched["mu"])
    assert set(proton) == {PARTICLES["proton"]} and set(muon) == {PARTICLES["muon"]}
    assert len(proton) == 2 * 12 and len(muon) == 2 * 12
    # and the training stream never sees them
    for stream in ds.train_val_datasets():
        assert set(collect(stream)[0]) <= {0, 1}


def test_watch_max_caps_the_events_read(tree):
    watched = dataset(simulation(tree), watch=["proton"], watch_max=5).watch_datasets()
    assert len(collect(watched["proton"])[0]) == 5


def test_dataset_reports_the_simulated_totals(tree):
    assert dataset(simulation(tree)).n_simulated("gamma") == 3 * 300 * 20


def test_the_old_call_with_two_lists_still_works(tree):
    sim = simulation(tree)
    old = TriggerDataset(sim.files("gamma"), sim.files("nsb"), batch_size=8, tel_id_only=1, percent_validation=0.34, load_ram=False)
    assert old.gamma_files == sim.files("gamma") and old.nsb_files == sim.files("nsb")
    assert set(collect(old.train_val_datasets()[0])[0]) == {0, 1}
    assert old.simulation.files("gamma") == sim.files("gamma")


def test_dataset_argument_checks(tree):
    sim = simulation(tree)
    with pytest.raises(ValueError, match="do not pass nsb_files"):
        TriggerDataset(sim, sim.files("nsb"))
    with pytest.raises(TypeError, match="nsb_files is missing"):
        TriggerDataset(sim.files("gamma"))
    with pytest.raises(ValueError, match="no 'muon' source"):
        TriggerDataset(Simulation(gamma=sim.files("gamma"), nsb=sim.files("nsb")), watch=["muon"])
    with pytest.raises(ValueError, match="duplicate"):
        dataset(sim, watch=["proton", "proton"])
    with pytest.raises(ValueError, match="train_on"):
        dataset(sim, train_on=("gamma", "proton"))                     # the NSB is always read
    with pytest.raises(ValueError, match="needs at least"):
        Simulation(gamma=None)


def test_a_literal_path_with_pattern_characters_is_a_file_not_a_pattern(tmp_path):
    from triggerkit.data.simulation import expand_files
    folder = tmp_path / "run[1]"
    folder.mkdir()
    (folder / "g.h5").write_bytes(b"x")
    assert expand_files(str(folder / "g.h5")) == [str(folder / "g.h5")]
    with pytest.raises(ValueError, match="matches no file"):
        expand_files(str(tmp_path / "run[2]" / "*.h5"))                     # a pattern that finds nothing is still an error
