import json
import os
from datetime import UTC, datetime, timedelta

import pytest

from platetrace import memory
from platetrace.memory import MemoryStore, prepare_lessons
from platetrace.models import RunRequest, VehicleRecord


def request(**changes):
    return RunRequest(**{"plate": "ABC-123", "jurisdiction": "US / CA", "make": "Toyota", "model": "Corolla",
                         "year": 2020, "authorized": True, **changes})


def result(name="fetch_url", **data):
    return {"type": "tool_result", "data": {"name": name, "result": data}}


def run(index=1, **changes):
    return {"id": f"{index:032x}", "provider": "ollama", "created_at": datetime.now(UTC).isoformat(),
            "status": "completed", "events": [result()], "sources": [], **changes}


def save(store, data=None, req=None):
    return store.remember(data or run(), req or request(), generation=store.generation)


def test_persistence_permissions_and_snapshot_isolation(tmp_path):
    directory = tmp_path / "memory"
    directory.mkdir(mode=0o777)
    directory.chmod(0o777)
    store = MemoryStore(directory)
    entry = save(store, run(research_lessons=["Fetch original manufacturer pages before citing recall information."]))
    assert entry["lessons"]
    assert entry["tool_outcomes"] == {"fetch_url": {"success": 1, "failure": 0}}
    assert directory.stat().st_mode & 0o777 == 0o700
    assert store.path.stat().st_mode & 0o777 == 0o600
    assert not list(directory.glob("*.tmp"))

    loaded = MemoryStore(directory)
    loaded.load()
    assert loaded.snapshot() == store.snapshot()
    loaded.snapshot()["entries"][0]["lessons"].clear()
    assert loaded.snapshot()["entries"][0]["lessons"]
    loaded.recall(request())[0]["lessons"].clear()
    assert loaded.snapshot()["entries"][0]["lessons"]


def test_secret_minimization_and_source_filtering(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brave-secret")
    req = request(api_key="request-secret", vin="1HGCM82633A004352",
                  records=[VehicleRecord(plate="ZZZ999", jurisdiction="US / CA", vin="JH4KA9650MC000001")])
    sources = [
        {"kind": "web", "url": "https://www.nhtsa.gov/recalls?api_key=request-secret#ABC123", "id": "S1",
         "title": "ABC123 raw case title", "excerpt": "secret report body"},
        {"kind": "web", "url": "https://example.com/ABC123"},
        {"kind": "web", "url": "https://example.com/ABC%252D123/details"},
        {"kind": "web", "url": "https://example.com/1HGCM82633A004352"},
        {"kind": "web", "url": "https://example.com/ZZZ999"},
        {"kind": "web", "url": "https://example.com/JH4KA9650MC000001"},
        {"kind": "web", "url": "https://user:pass@example.com/recalls"},
        {"kind": "web", "url": "https://example.com/plates/different-identifier"},
        {"kind": "provided", "url": "https://example.com/records"},
        {"kind": "demo", "url": "https://example.com/demo"},
        {"kind": "web", "url": "http://127.0.0.1/private"},
        {"kind": "web", "url": "http://localhost/private"},
    ]
    data = run(sources=sources, plate=req.plate, vin=req.vin, records=["raw records"],
               report={"summary": "ABC123 secret summary", "findings": [{"detail": "sensitive findings"}]},
               events=[result("terminal", exit_code=0, stdout="raw terminal log", command="wget private-url"),
                       result("fetch_url", error="request-secret bad credentials")],
               research_lessons=["ABC123 1HGCM82633A004352 ZZZ999 JH4KA9650MC000001 request-secret brave-secret S1",
                                 "Source https://example.com/private?token=unknown-secret needs a fresh fetch.",
                                 "The API returned token=other-secret and Bearer credential-value.",
                                 "Run `curl private-url` to fetch raw records.",
                                 "curl -H 'Authorization: Bearer raw-key' https://example.com/"])
    store = MemoryStore(tmp_path)
    entry = save(store, data, req)
    assert entry["sources"] == [{"url": "https://www.nhtsa.gov/recalls"}]
    assert entry["tool_outcomes"] == {"terminal": {"success": 1, "failure": 0},
                                      "fetch_url": {"success": 0, "failure": 1}}
    text = store.path.read_text()
    for private in ("ABC123", "ABC-123", req.vin, "ZZZ999", "JH4KA9650MC000001", "request-secret", "brave-secret",
                    "unknown-secret", "other-secret", "credential-value", "raw-key", "secret report body",
                    "raw terminal log", "secret summary", "sensitive findings", "S1", "curl", "raw records"):
        assert private not in text
    assert "[REDACTED" in text


def test_context_and_objective_do_not_retain_case_identifiers(tmp_path):
    req = request(jurisdiction="US / CA ABC-123", make="ABC123", model="request-secret", api_key="request-secret",
                  objective="Find recall info for ABC123 with token=other-secret")
    entry = save(MemoryStore(tmp_path), req=req)
    assert entry["jurisdiction"] == entry["make"] == entry["model"] == ""
    assert entry["purpose_tags"] == ["recalls"]
    assert "objective" not in entry and "purpose" not in entry


@pytest.mark.parametrize("status", ["queued", "running", "cancelling", "cancelled", "interrupted"])
def test_nonterminal_and_cancelled_runs_not_learned(tmp_path, status):
    store = MemoryStore(tmp_path)
    assert save(store, run(status=status)) is None
    assert not store.path.exists()


def test_demo_disabled_memory_and_no_observations_not_learned(tmp_path):
    store = MemoryStore(tmp_path)
    assert save(store, req=request(provider="demo")) is None
    assert save(store, run(provider="demo")) is None
    assert save(store, req=request(use_memory=False)) is None
    assert save(store, run(status="failed", events=[])) is None
    assert save(store, run(status="incomplete", events=[])) is None
    assert save(store, run(events=[result("finish_report", saved=True)],
                           research_lessons=["Unobserved model advice alone must not become experience."])) is None
    save(store)
    assert store.recall(request(use_memory=False)) == []
    assert store.recall(request(provider="demo")) == []


def test_failed_and_incomplete_runs_learn_observed_outcomes(tmp_path):
    store = MemoryStore(tmp_path)
    entry = save(store, run(status="failed", events=[result("search_web", error="offline"),
                     result("terminal", exit_code=2, stderr="raw failure"), result("terminal", exit_code=0),
                     result("terminal", exit_code=0, timed_out=True), result("unknown-secret", error="key")]))
    assert entry["tool_outcomes"] == {"search_web": {"success": 0, "failure": 1},
                                      "terminal": {"success": 1, "failure": 2}}
    entry = save(store, run(2, status="incomplete"))
    assert entry["status"] == "incomplete"


def test_relevance_ranking_and_exclusion(tmp_path):
    store = MemoryStore(tmp_path)
    old = (datetime.now(UTC) - timedelta(days=2)).isoformat()
    best = save(store, run(1, created_at=old))
    save(store, run(2), request(make="Honda", model="Civic"))
    irrelevant = request(jurisdiction="France", make="Peugeot", model="208", year=2024,
                         purpose="other", objective="Check emissions")
    save(store, run(3), irrelevant)
    assert [entry["id"] for entry in store.recall(request())] == [best["id"], f"{2:032x}"]
    assert [entry["id"] for entry in store.recall(request(), exclude_run_id=best["id"])] == [f"{2:032x}"]
    assert store.recall(irrelevant)[0]["id"] == f"{3:032x}"


def test_bounds_freshness_deduplication_and_recall_limit(tmp_path, monkeypatch):
    store = MemoryStore(tmp_path)
    # Exercise bounded selection without fsyncing 205 identical fixture writes.
    monkeypatch.setattr(store, "_save", lambda: None)
    for index in range(memory.MAX_ENTRIES + 5):
        save(store, run(index))
    assert len(store.entries) == memory.MAX_ENTRIES
    assert len(store.recall(request())) == memory.MAX_RECALLED
    original = store.snapshot()
    same = save(store, run(memory.MAX_ENTRIES + 4, research_lessons=["Duplicate lesson must not overwrite."]))
    assert same["lessons"] == []
    assert store.snapshot() == original
    stale = run(999, created_at=(datetime.now(UTC) - timedelta(days=91)).isoformat())
    assert save(store, stale) is None
    store.entries[0]["created_at"] = stale["created_at"]
    assert len(store.snapshot()["entries"]) == memory.MAX_ENTRIES - 1
    assert all(entry["id"] != store.entries[0]["id"] for entry in store.recall(request()))


def test_clear_generation_blocks_inflight_save_and_survives_restart(tmp_path):
    store = MemoryStore(tmp_path)
    save(store)
    generation = store.generation
    store.clear()
    assert store.snapshot() == {"entries": [], "count": 0, "generation": generation + 1}
    assert store.remember(run(2), request(), generation=generation) is None
    loaded = MemoryStore(tmp_path)
    loaded.load()
    assert loaded.generation == generation + 1
    assert loaded.snapshot()["count"] == 0
    assert loaded.remember(run(3), request(), generation=generation) is None
    assert save(loaded, run(4)) is not None


@pytest.mark.parametrize("raw", ["broken json", "[]", '{}', '{"version":1,"generation":false,"entries":[]}',
                                 '{"version":1,"generation":3,"entries":null}'])
def test_corrupt_store_load_is_safe(tmp_path, raw):
    store = MemoryStore(tmp_path)
    store.path.write_text(raw)
    store.load()
    assert store.snapshot()["count"] == 0
    assert store.bootstrap([run()]) == 0


def test_load_filters_invalid_schema_expired_and_duplicate_entries(tmp_path):
    store = MemoryStore(tmp_path)
    good = save(store)
    bad = {**good, "id": f"{2:032x}", "plate": "do-not-load"}
    invalid = {**good, "id": f"{3:032x}", "status": []}
    stale = {**good, "id": f"{4:032x}", "created_at": (datetime.now(UTC) - timedelta(days=100)).isoformat()}
    overflow = {**good, "id": f"{5:032x}", "created_at": "0001-01-01T00:00:00+01:00"}
    no_url = {**good, "id": f"{6:032x}", "sources": [{"url": None}]}
    store.path.write_text(json.dumps({"version": 1, "generation": 4,
                                     "entries": [good, bad, invalid, stale, overflow, no_url, good]}))
    store.path.chmod(0o644)
    loaded = MemoryStore(tmp_path)
    loaded.load()
    assert loaded.snapshot() == {"entries": [good], "count": 1, "generation": 4}
    assert loaded.path.stat().st_mode & 0o777 == 0o600


def test_write_failure_is_generic_atomic_and_rolls_back(tmp_path, monkeypatch):
    store = MemoryStore(tmp_path)
    save(store)
    before = store.path.read_bytes()

    def fail(*args):
        raise OSError("secret path and credential")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError, match="Research memory could not be saved locally") as error:
        save(store, run(2))
    assert "secret" not in str(error.value)
    assert store.path.read_bytes() == before
    assert store.snapshot()["count"] == 1
    with pytest.raises(OSError):
        store.clear()
    assert store.generation == 0 and store.snapshot()["count"] == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_prepare_lessons_bounds_and_decoded_identifiers():
    lessons = prepare_lessons(request(), ["Use original pages.", "Use original pages.", "ABC%252D123 S25",
                                          "x" * 1001, "Use official recall pages.", "Omitted sixth lesson."])
    assert len(lessons) == 3
    assert "[REDACTED VEHICLE ID]" in lessons[1]
    assert "S25" not in lessons[1]
    assert prepare_lessons(request(), "wrong shape") == []


def test_bootstrap_imports_old_outcomes_and_origins_once(tmp_path):
    store = MemoryStore(tmp_path)
    history = [{**request().model_dump(exclude={"api_key", "records"}), **run(),
                "research_lessons": ["Legacy raw secret should not be imported."],
                "sources": [{"kind": "web", "url": "https://www.nhtsa.gov/deep/raw-secret?token=private"}]},
               {**request(provider="demo").model_dump(), **run(2, provider="demo")},
               {**request(use_memory=False).model_dump(), **run(3)}]
    assert store.bootstrap(history) == 1
    entry = store.entries[0]
    assert entry["sources"] == [{"url": "https://www.nhtsa.gov"}]
    assert entry["lessons"] == [] and entry["tool_outcomes"]
    assert "secret" not in store.path.read_text()
    store.clear()
    assert store.bootstrap(history) == 0
    loaded = MemoryStore(tmp_path)
    loaded.load()
    assert loaded.bootstrap(history) == 0
    assert loaded.snapshot()["count"] == 0


def test_empty_bootstrap_creates_tombstone(tmp_path):
    store = MemoryStore(tmp_path)
    assert store.bootstrap([]) == 0
    assert store.path.exists()
    assert store.snapshot()["count"] == 0


def test_unicode_heavy_memory_stays_within_loadable_byte_budget(tmp_path):
    store = MemoryStore(tmp_path)
    entry = save(store)
    store.entries = [{**entry, "id": f"{index:032x}",
                      "lessons": ["😀" * 999 + str(lesson) for lesson in range(5)]}
                     for index in range(memory.MAX_ENTRIES)]
    store._save()
    assert 0 < len(store.entries) < memory.MAX_ENTRIES
    assert store.path.stat().st_size <= memory._MAX_FILE_BYTES
    loaded = MemoryStore(tmp_path)
    loaded.load()
    assert loaded.snapshot()["count"] == len(store.entries)


def test_bootstrap_skips_malformed_legacy_status_and_event(tmp_path):
    store = MemoryStore(tmp_path)
    base = request().model_dump()
    history = [{**base, **run(1, status=[])},
               {**base, **run(2, created_at="0001-01-01T00:00:00+01:00")},
               {**base, **run(3, events=[{"type": "tool_result", "data": {"name": []}}, result()],
                              sources=[{"kind": "web", "url": {"invalid": "shape"}}])}]
    assert store.bootstrap(history) == 1
