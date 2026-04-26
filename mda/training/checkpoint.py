import math
import time
import numpy as np
import json
import os
import uuid
import tempfile
from datetime import datetime
from pathlib import Path
from mda.core.registry import EntityRegistry
from mda.inference.broca import BrocaModule
from mda.core.neuron import Synapse
from mda.core.bind import normalize


def save(registry: EntityRegistry, broca: BrocaModule, path: str,
         user_id: str = "default", model_name: str = "unknown",
         turn_count: int = 0) -> dict:
    base = path.replace(".npz", "").replace(".json", "").replace(".mda", "")

    # Preserve created_at if checkpoint already exists
    existing_json = base + ".json"
    if Path(existing_json).exists():
        try:
            with open(existing_json, encoding="utf-8") as f:
                old = json.load(f)
            created_at = old.get("session", {}).get("created_at",
                                 datetime.utcnow().isoformat())
        except Exception:
            created_at = datetime.utcnow().isoformat()
    else:
        created_at = datetime.utcnow().isoformat()

    # Apply temporal decay to all synapses before saving, then reset timestamps
    now = time.time()
    for e in registry._entities.values():
        e.last_activated = getattr(e, "last_activated", now)
        for syn in e.synapses.values():
            syn.apply_decay(now)

    meta = {
        "session": {
            "user_id":    user_id,
            "session_id": str(uuid.uuid4()),
            "model_name": model_name,
            "turn_count": turn_count,
            "mda_version": "1.0",
            "created_at": created_at,
            "updated_at": datetime.utcnow().isoformat(),
        },
        "entities": {
            eid: {
                "surface":        e.surface,
                "category":       e.category,
                "use_count":      e.use_count,
                "beta":           e.beta,
                "last_activated": e.last_activated,
                "synapses": {
                    syn.target_id: {
                        "strength":         syn.strength,
                        "activation_count": syn.activation_count,
                        "last_activated":   syn.last_activated,
                    }
                    for syn in e.synapses.values()
                },
            }
            for eid, e in registry._entities.items()
        },
        "broca": {
            "categories":       broca._categories,
            "entity_facts":     broca._entity_facts,
            "entity_positions": broca._entity_positions,
        },
    }

    # Top 50 proper noun entity by use_count — only these get W saved
    w_candidates = [
        e for e in registry._entities.values()
        if e.W is not None and e.surface and e.surface[0].isupper()
    ]
    w_candidates.sort(key=lambda e: e.use_count, reverse=True)
    top_w_ids = {e.id for e in w_candidates[:50]}

    arrays = {}
    w_saved = 0
    for eid, e in registry._entities.items():
        arrays[f"{eid}_v"] = e.v.astype(np.float32)
        arrays[f"{eid}_h"] = e.h.astype(np.float32)
        if e.W is not None and e.id in top_w_ids:
            arrays[f"{eid}_W"] = e.W.astype(np.float32)
            w_saved += 1

    Path(base).parent.mkdir(parents=True, exist_ok=True)

    # Atomic write — tmp → rename so a crash mid-save never corrupts the existing file
    # Note: np.savez_compressed appends .npz if not already present, so name accordingly
    tmp_npz = base + ".tmp.npz"
    np.savez_compressed(tmp_npz, **arrays)
    os.replace(tmp_npz, base + ".npz")

    tmp_json = base + ".json.tmp"
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    os.replace(tmp_json, base + ".json")

    npz_kb  = os.path.getsize(base + ".npz")  / 1024
    json_kb = os.path.getsize(base + ".json") / 1024
    n = len(registry._entities)
    print(f"Saved: {base}")
    print(f"  user_id:{user_id} | model:{model_name} | turns:{turn_count}")
    print(f"  {n} entities ({w_saved} W saved) | {npz_kb:.1f} KB npz | {json_kb:.1f} KB json")

    return meta["session"]


def load(registry: EntityRegistry, broca: BrocaModule,
         path: str, decoder_trainer=None) -> dict:
    base = path.replace(".npz", "").replace(".json", "").replace(".mda", "")

    with open(base + ".json", encoding="utf-8") as f:
        meta = json.load(f)

    arrays = np.load(base + ".npz", allow_pickle=False)

    # old_id (JSON) → new entity.id (assigned by registry counter)
    id_map: dict[str, str] = {}

    for old_eid, edata in meta["entities"].items():
        entity = registry.get_or_create(edata["surface"], edata["category"])
        id_map[old_eid] = entity.id
        entity.use_count      = edata["use_count"]
        entity.beta           = edata["beta"]
        entity.last_activated = edata.get("last_activated", time.time())

        v_key = f"{old_eid}_v"
        h_key = f"{old_eid}_h"
        w_key = f"{old_eid}_W"
        if v_key in arrays:
            entity.v = arrays[v_key].astype(np.float64)
        if h_key in arrays:
            entity.h = arrays[h_key].astype(np.float64)
        entity.W = arrays[w_key].astype(np.float64) if w_key in arrays else None

    # Restore synapses — remap target IDs too
    for old_eid, edata in meta["entities"].items():
        new_eid = id_map[old_eid]
        entity  = registry.get_by_id(new_eid)
        if entity is None:
            continue
        for old_tid, sdata in edata.get("synapses", {}).items():
            new_tid = id_map.get(old_tid, old_tid)
            syn = Synapse(source_id=new_eid, target_id=new_tid, vector=entity.v)
            syn.strength         = sdata["strength"]
            syn.activation_count = sdata["activation_count"]
            syn.last_activated   = sdata.get("last_activated", time.time())
            entity.synapses[new_tid] = syn

    # Restore broca — remap all keys from old IDs to new IDs
    old_facts     = meta["broca"]["entity_facts"]
    old_positions = meta["broca"]["entity_positions"]

    for old_eid, facts in old_facts.items():
        new_eid = id_map.get(old_eid, old_eid)
        existing = broca._entity_facts.get(new_eid, [])
        new_only = [f for f in facts if f not in existing]
        broca._entity_facts[new_eid] = existing + new_only

    for old_eid, positions in old_positions.items():
        new_eid = id_map.get(old_eid, old_eid)
        existing = broca._entity_positions.get(new_eid, [])
        broca._entity_positions[new_eid] = existing + positions

    broca._categories = meta["broca"]["categories"]

    # Reconstruct fact_vecs from text at load time (avoids storing large arrays)
    for new_eid, facts in broca._entity_facts.items():
        broca._fact_vecs[new_eid] = [
            normalize(broca.encoder.encode(f)) for f in facts
        ]

    session = meta.get("session", {})
    user_id    = session.get("user_id", "default")
    model_name = session.get("model_name", "unknown")
    turn_count = session.get("turn_count", 0)
    n          = len(registry._entities)
    fact_count = sum(len(v) for v in broca._entity_facts.values())

    print(f"Loaded: {base}")
    print(f"  user_id:{user_id} | model:{model_name} | turns:{turn_count}")
    print(f"  {n} entities | {fact_count} facts")

    return session
