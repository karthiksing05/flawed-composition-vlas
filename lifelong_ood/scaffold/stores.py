"""The scaffold architecture's symbolic stores (docs/PLAN_SCAFFOLD.md section 5).

Lexicon        word -> referent and role, from cross-situational evidence (LADDER_RESULTS 5.2's rule), and it refuses a parse it cannot ground
EntityStore    atom -> visual key, frame and size; grows with new objects
ExceptionStore (object, target) -> small correction; created only on a measured gap, and its growth is a test of the thesis
Schema         verb -> phases, each naming the role it serves, with event transitions and actions per query

The neural stores (contact parts, shared backbones, the transit field) live in their own modules and report their sizes to storage_report.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..policy.parts_flow import role_lexicon, role_parse

PARAM_BYTES = 4                                                             # float32


@dataclass(frozen=True)
class Parse:
    """object and target are None exactly when reason is set."""
    object: Optional[str]
    target: Optional[str]
    reason: Optional[str] = None
    tokens: Tuple[str, ...] = ()

    @property
    def abstained(self) -> bool:
        return self.reason is not None


class Lexicon:
    """situations: {instruction: [grasp slot, release slot]}, observed from demos.  Reads no given role.

    parse() refuses in two cases, and a silent wrong parse is the failure this guards against (LADDER_RESULTS 5.7, K4 withheld):
      unknown    a token no situation has ever contained
      ambiguous  a token whose candidate referents are not yet narrowed to one (more than one slot left)
    A token with no candidates left is a function word: it co-occurs with no slot in every situation, and it carries no referent.
    """

    def __init__(self, situations: Dict[str, List[str]]):
        self.lex = role_lexicon(situations)
        toks = {s: s.lower().split() for s in situations}
        self.vocab = {w for v in toks.values() for w in v}
        self.candidates = {}
        ref = self.lex["ref"]
        for w in self.vocab:
            has = [s for s in situations if w in toks[s]]
            c = set.intersection(*[set(situations[s]) for s in has])
            if w not in ref:                                                # explaining away, at the fixpoint the lexicon reached
                gone = set.intersection(*[{ref[u] for u in toks[s] if u != w and u in ref} for s in has])
                c = c - gone
            self.candidates[w] = c

    def parse(self, instruction: str) -> Parse:
        toks = tuple(instruction.lower().split())
        unknown = tuple(w for w in toks if w not in self.vocab)
        if unknown:
            return Parse(None, None, "unknown", unknown)
        ambiguous = tuple(w for w in toks if w not in self.lex["ref"] and len(self.candidates[w]) > 1)
        if ambiguous:
            return Parse(None, None, "ambiguous", ambiguous)
        p = role_parse(self.lex, instruction)
        if p is None:
            return Parse(None, None, "no roles", toks)
        return Parse(p[0], p[1], None, toks)


@dataclass
class EntityStore:
    """atom -> what is known about it.  Adding a known atom is an error: a new demo of it updates nothing here."""
    atoms: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def add(self, atom: str, key_params: int = 0, **attrs) -> None:
        if atom in self.atoms:
            raise KeyError(f"{atom} already stored")
        self.atoms[atom] = dict(attrs, key_params=int(key_params))

    def params(self) -> int:
        return sum(a["key_params"] for a in self.atoms.values())


class ExceptionStore:
    """(object, target) -> a small correction.  create() refuses unless the measured gap clears its bar: the store stays sparse by construction, and
    n_exceptions against n_pairs_seen is the thesis's test (PLAN_SCAFFOLD C4)."""

    def __init__(self, min_gap: float = 0.10):
        self.min_gap = float(min_gap)
        self.entries: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.pairs_seen: set = set()

    def see(self, obj: str, tgt: str) -> None:
        self.pairs_seen.add((obj, tgt))

    def create(self, obj: str, tgt: str, params: int, gap: float, ci_low: float, payload: Any = None) -> bool:
        """gap: success without the exception's pair-level correction, as a shortfall against what the parts predict.  -> whether it was created."""
        self.see(obj, tgt)
        if not (ci_low > 0.0 and gap >= self.min_gap):
            return False
        self.entries[(obj, tgt)] = dict(params=int(params), gap=float(gap), ci_low=float(ci_low), payload=payload)
        return True

    def get(self, obj: str, tgt: str) -> Optional[Any]:
        e = self.entries.get((obj, tgt))
        return None if e is None else e["payload"]

    def params(self) -> int:
        return sum(e["params"] for e in self.entries.values())

    def density(self) -> float:
        return len(self.entries) / len(self.pairs_seen) if self.pairs_seen else 0.0


@dataclass(frozen=True)
class Phase:
    """One skill of a schema.  role: which slot's part acts.  store: which neural store serves it.  on: event -> next phase."""
    name: str
    role: str
    store: str
    on: Tuple[Tuple[str, str], ...] = ()
    actions_per_query: int = 5


@dataclass(frozen=True)
class Schema:
    verb: str
    slots: Tuple[str, ...]
    phases: Tuple[Phase, ...]

    def __post_init__(self):
        names = {p.name for p in self.phases}
        assert len(names) == len(self.phases), "phase names repeat"
        for p in self.phases:
            assert p.role in self.slots, (p.name, p.role)
            for _, nxt in p.on:
                assert nxt in names, (p.name, nxt)


class SchemaRun:
    """One execution: holds the active phase, and moves only on a named event.  An event the active phase does not name changes nothing."""

    def __init__(self, schema: Schema):
        self.schema = schema
        self.by_name = {p.name: p for p in schema.phases}
        self.phase = schema.phases[0]
        self.history = [self.phase.name]

    def step(self, events=()) -> Phase:
        for ev in events:
            nxt = dict(self.phase.on).get(ev)
            if nxt is not None:
                self.phase = self.by_name[nxt]
                self.history.append(nxt)
                break
        return self.phase


# P_min's hard gate as a schema: the object part until a hold, the target part during it, and back if the hold ends.
PICK_PLACE = Schema("put", ("object", "target"), (
    Phase("grasp", "object", "contact_parts", on=(("held_on", "carry"),)),
    Phase("carry", "target", "contact_parts", on=(("held_off", "grasp"),)),
))


def storage_report(shared_params: int, atom_params: Dict[str, int], entity: Optional[EntityStore] = None, exceptions: Optional[ExceptionStore] = None,
                   lexicon_bytes: int = 0, schema_bytes: int = 0) -> Dict[str, Any]:
    """Every byte the system holds, shared stores included (PLAN_SCAFFOLD 3.2): a claim about parts alone would hide the backbones they need."""
    ent = entity.params() if entity else 0
    exc = exceptions.params() if exceptions else 0
    part = sum(atom_params.values())
    total = shared_params + part + ent + exc
    return dict(shared_bytes=shared_params * PARAM_BYTES, part_bytes=part * PARAM_BYTES, entity_bytes=ent * PARAM_BYTES, exception_bytes=exc * PARAM_BYTES,
                lexicon_bytes=int(lexicon_bytes), schema_bytes=int(schema_bytes), total_bytes=total * PARAM_BYTES + int(lexicon_bytes) + int(schema_bytes),
                bytes_per_atom={k: v * PARAM_BYTES for k, v in atom_params.items()})
