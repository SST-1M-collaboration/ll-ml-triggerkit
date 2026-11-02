import numpy as np
import pytest
from ctapipe.instrument import CameraGeometry

from triggerkit.augment import (
    Fold, identity_index, make_condition_folds, make_rotation_folds, roll_index, rotation_index, shuffle_index,
)
from triggerkit.data import Simulation


@pytest.fixture
def square_cam():
    return CameraGeometry.make_rectangular(4, 4)


def test_rotation_90_is_a_permutation_of_a_square_grid(square_cam):
    perm = rotation_index(square_cam, 90)
    assert sorted(perm) == list(range(square_cam.n_pixels))
    # four quarter turns bring every pixel home
    out = np.arange(square_cam.n_pixels)
    for _ in range(4):
        out = out[perm]
    assert np.array_equal(out, np.arange(square_cam.n_pixels))


def test_non_symmetry_angle_fails_loudly(square_cam):
    with pytest.raises(ValueError):
        rotation_index(square_cam, 45)


def test_simple_index_builders():
    assert np.array_equal(identity_index(5), np.arange(5))
    assert np.array_equal(roll_index(5, 2), [3, 4, 0, 1, 2])
    a, b = shuffle_index(50, 7), shuffle_index(50, 7)
    assert np.array_equal(a, b) and sorted(a) == list(range(50))


def test_fold_requires_matching_lengths():
    with pytest.raises(ValueError):
        Fold("x", np.arange(3), np.arange(4))


def _spec(**kw):
    base = {"events": {"gamma": 10, "nsb": 5}, "conditions": ["medium"]}
    base.update(kw)
    return base


CONDS = {"medium": Simulation(gamma=["g.h5"], proton=["p.h5"], nsb=["n.h5"]), "low": Simulation(gamma=["gl.h5"], nsb=["nl.h5"])}


def test_make_rotation_folds_names_and_budgets(square_cam):
    folds = make_rotation_folds(square_cam, [
        _spec(conditions=["low", "medium"]),
        _spec(shower_deg=90, shower_time_shift=5),
        _spec(nsb_kind="rolled", nsb_param=3, events={"gamma": 10, "nsb": 5, "proton": None}),
    ], CONDS)
    names = [f.name for f in folds]
    assert names == ["rot0_original_low", "rot0_original_medium",
                     "rot90_original_troll5_medium", "rot0_rolled3_medium"]
    assert len(set(names)) == len(names)
    assert folds[0].sources.files("gamma") == ["gl.h5"] and folds[1].sources.files("gamma") == ["g.h5"]
    assert folds[0].events == {"gamma": 10, "nsb": 5} and folds[3].events == {"gamma": 10, "nsb": 5, "proton": None}
    assert folds[2].shower_time_shift == 5 and folds[1].condition == "medium"
    assert folds[2].config["shower_deg"] == 90 and "gamma_deg" not in folds[2].config


@pytest.mark.parametrize("bad", [
    {"events": {"gamma": 1}},                                 # missing conditions
    {"conditions": ["medium"]},                               # missing events
    _spec(typo_key=1),                                        # unknown key
    _spec(gamma_deg=120),                                     # the old name of shower_deg
    _spec(conditions=[]),                                     # empty
    _spec(events={}),                                         # nothing to read
    _spec(events={"electron": 5}),                            # unknown type
    _spec(events={"gamma": 0}),                               # a budget of 0
])
def test_bad_specs_rejected(square_cam, bad):
    with pytest.raises(ValueError):
        make_rotation_folds(square_cam, [bad], CONDS)


def test_unknown_condition_and_kind(square_cam):
    with pytest.raises(ValueError):
        make_rotation_folds(square_cam, [_spec(conditions=["high"])], CONDS)
    with pytest.raises(ValueError):
        make_rotation_folds(square_cam, [_spec(nsb_kind="bogus")], CONDS)


def test_a_type_the_condition_does_not_have_is_refused(square_cam):
    with pytest.raises(ValueError, match="no such source"):
        make_rotation_folds(square_cam, [_spec(conditions=["low"], events={"gamma": 5, "muon": 5})], CONDS)
    with pytest.raises(ValueError, match="no such source"):
        make_condition_folds(square_cam, {"low": (CONDS["low"], {"gamma": 5, "proton": 5})})


def test_duplicate_rows_get_unique_names(square_cam):
    folds = make_rotation_folds(square_cam, [_spec(), _spec()], CONDS)
    assert len({f.name for f in folds}) == 2


def test_condition_folds(square_cam):
    folds = make_condition_folds(square_cam, {"medium": (CONDS["medium"], {"gamma": 7, "nsb": None})})
    fold = folds[0]
    assert fold.name == "medium" and fold.condition == "medium" and fold.events == {"gamma": 7, "nsb": None}
    assert np.array_equal(fold.shower_index, identity_index(square_cam.n_pixels)) and np.array_equal(fold.shower_index, fold.nsb_index)


def test_the_files_of_a_condition_must_be_a_simulation_not_the_old_pair(square_cam):
    with pytest.raises(TypeError, match="Simulation"):
        make_rotation_folds(square_cam, [_spec()], {"medium": (["g.h5"], ["n.h5"])})
    with pytest.raises(TypeError, match="Simulation"):
        make_condition_folds(square_cam, {"medium": ((["g.h5"], ["n.h5"]), {"gamma": 5})})
