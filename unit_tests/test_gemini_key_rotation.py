"""GeminiClient multi-key handling: key loading, switching on 429/503, dropping bad keys.

Runs offline: ``genai.Client`` is replaced by a fake whose responses are scripted per key.
"""

import types

import pytest
from pydantic import BaseModel

gc = pytest.importorskip("diffusion_policy.veritas.gemini_client")


class Reply(BaseModel):
    ok: bool


class FakeClient:
    script = {}   # key -> list of outcomes ("ok" or an error message), consumed in order
    calls = []    # keys in call order

    def __init__(self, api_key):
        self.key = api_key
        self.models = types.SimpleNamespace(generate_content=self._generate)

    def _generate(self, **_):
        FakeClient.calls.append(self.key)
        outcome = FakeClient.script[self.key].pop(0)
        if outcome == "ok":
            return types.SimpleNamespace(text='{"ok": true}', usage_metadata=None)
        raise RuntimeError(outcome)


@pytest.fixture
def keydir(tmp_path, monkeypatch):
    monkeypatch.setattr(gc.genai, "Client", FakeClient)
    monkeypatch.setattr(gc.time, "sleep", lambda s: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "0")          # start on the first key
    monkeypatch.setenv("GEMINI_KEY_GLOB", str(tmp_path / "api_key*"))
    FakeClient.calls = []
    return tmp_path


def write_keys(d, **keys):
    for name, key in keys.items():
        (d / name).write_text(key + "\n")


def call(client):
    return client._call_with_retry(["hi"], {}, Reply, max_retries=3)


def test_load_dedups_env_and_files(keydir, monkeypatch):
    write_keys(keydir, api_key="A", api_key2="B", api_key3="A")
    monkeypatch.setenv("GEMINI_API_KEY", "B")
    assert gc.load_api_keys() == [("GEMINI_API_KEY", "B"), ("api_key", "A")]


def test_rate_limit_switches_key_without_backoff(keydir, monkeypatch):
    write_keys(keydir, api_key="A", api_key2="B")
    slept = []
    monkeypatch.setattr(gc.time, "sleep", slept.append)
    FakeClient.script = {"A": ["503 UNAVAILABLE"], "B": ["ok"]}
    assert call(gc.GeminiClient()).ok
    assert FakeClient.calls == ["A", "B"] and slept == []


def test_backs_off_only_when_all_keys_limited(keydir, monkeypatch):
    write_keys(keydir, api_key="A", api_key2="B")
    slept = []
    monkeypatch.setattr(gc.time, "sleep", slept.append)
    FakeClient.script = {"A": ["429 quota", "ok"], "B": ["503 UNAVAILABLE"]}
    assert call(gc.GeminiClient()).ok
    assert FakeClient.calls == ["A", "B", "A"] and len(slept) == 1


def test_bad_key_dropped(keydir):
    write_keys(keydir, api_key="A", api_key2="B")
    FakeClient.script = {"A": ["401 UNAUTHENTICATED"], "B": ["ok", "ok"]}
    client = gc.GeminiClient()
    assert call(client).ok and call(client).ok
    assert FakeClient.calls == ["A", "B", "B"]
    assert [label for label, _ in client._keys] == ["api_key2"]


def test_jobs_start_on_different_keys(keydir, monkeypatch):
    write_keys(keydir, api_key="A", api_key2="B")
    monkeypatch.setenv("SLURM_JOB_ID", "41094715")
    assert gc.GeminiClient().key_label == "api_key2"


def test_single_key_unchanged(keydir):
    write_keys(keydir, api_key="A")
    FakeClient.script = {"A": ["503 UNAVAILABLE", "503 UNAVAILABLE", "503 UNAVAILABLE"]}
    assert call(gc.GeminiClient()) is None
    assert FakeClient.calls == ["A", "A", "A"]


def test_backoff_jittered_and_global_rng_untouched(keydir, monkeypatch):
    import random
    write_keys(keydir, api_key="A")
    slept = []
    monkeypatch.setattr(gc.time, "sleep", slept.append)
    FakeClient.script = {"A": ["503 UNAVAILABLE"] * 2 + ["ok"]}
    random.seed(0)
    state = random.getstate()
    client = gc.GeminiClient()
    assert client._call_with_retry(["hi"], {}, Reply, max_retries=3,
                                   initial_retry_delay=5.0).ok
    assert random.getstate() == state
    # base delays 10, 20 (doubling), each scaled into [0.5, 1.5)
    assert 5 <= slept[0] < 15 and 10 <= slept[1] < 30
