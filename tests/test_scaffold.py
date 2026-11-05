"""The scaffold's symbolic stores (lifelong_ood/scaffold/stores.py)."""
import pytest

from lifelong_ood.scaffold import PICK_PLACE, EntityStore, ExceptionStore, Lexicon, Phase, Schema, SchemaRun, storage_report

SIT = {
    "put the bowl on the plate": ["bowl", "plate"],
    "put the bowl on the stove": ["bowl", "stove"],
    "put the cheese on the stove": ["cheese", "stove"],
    "put the cheese on the plate": ["cheese", "plate"],
}


def test_lexicon_parses_new_pairing():
    lx = Lexicon({k: v for k, v in SIT.items() if k != "put the cheese on the plate"})
    p = lx.parse("put the cheese on the plate")
    assert not p.abstained and (p.object, p.target) == ("cheese", "plate")


def test_lexicon_refuses_unknown_word():
    p = Lexicon(SIT).parse("put the bottle on the plate")
    assert p.abstained and p.reason == "unknown" and p.tokens == ("bottle",)
    assert p.object is None and p.target is None


def test_lexicon_refuses_ambiguous_word():
    # one situation: nothing separates "wine" from the two slots it co-occurs with
    p = Lexicon({"put the wine on the rack": ["wine", "rack"]}).parse("put the wine on the rack")
    assert p.abstained and p.reason == "ambiguous"


def test_entity_store_rejects_duplicates():
    e = EntityStore()
    e.add("cheese", key_params=10, size=(0.03, 0.05))
    with pytest.raises(KeyError):
        e.add("cheese")
    assert e.params() == 10


def test_exceptions_need_a_measured_gap():
    x = ExceptionStore(min_gap=0.10)
    assert not x.create("bottle", "plate", 8, gap=0.05, ci_low=0.01)
    assert not x.create("bottle", "stove", 8, gap=0.30, ci_low=-0.01)
    assert x.create("bottle", "rack", 8, gap=0.30, ci_low=0.10, payload="m")
    assert x.get("bottle", "rack") == "m" and x.get("bottle", "plate") is None
    assert x.params() == 8 and abs(x.density() - 1 / 3) < 1e-12


def test_pick_place_schema_is_p_min_gate():
    r = SchemaRun(PICK_PLACE)
    assert r.phase.role == "object"
    assert r.step(["held_off"]).role == "object"                            # an event the phase does not name changes nothing
    assert r.step(["held_on"]).role == "target"
    assert r.step().role == "target"
    assert r.step(["held_off"]).role == "object"
    assert r.history == ["grasp", "carry", "grasp"]


def test_schema_rejects_bad_structure():
    with pytest.raises(AssertionError):
        Schema("v", ("object",), (Phase("a", "target", "x"),))
    with pytest.raises(AssertionError):
        Schema("v", ("object",), (Phase("a", "object", "x", on=(("e", "b"),)),))


def test_storage_counts_shared_stores():
    r = storage_report(1000, {"cheese": 10, "bottle": 10}, lexicon_bytes=7)
    assert r["shared_bytes"] == 4000 and r["part_bytes"] == 80 and r["total_bytes"] == 4087
    assert r["bytes_per_atom"]["cheese"] == 40
