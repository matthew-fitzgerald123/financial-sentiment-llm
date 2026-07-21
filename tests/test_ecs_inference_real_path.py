"""
Tests for app/main_ecs.py inference helpers in non-MOCK_MODE (real-path)
execution.

All existing ECS tests either patch _generate/_stream_into_queue entirely
(test_api_ecs.py) or set MOCK_MODE=true which short-circuits both functions
before the actual inference code.  These tests exercise the real (non-mock)
code paths by:
  - Setting MOCK_MODE=False via monkeypatch
  - Injecting a mock model + tokenizer into the module-level pipeline dict
  - Using patch.dict(sys.modules) to supply a fake transformers module so
    the real transformers package is never imported (it calls find_spec on
    mlx/peft stubs that lack __spec__, which raises ValueError)

No GPU or model weights are required.
"""
import sys
import threading
from queue import Queue
from threading import Event
from unittest.mock import MagicMock, patch

import pytest

from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_DECODED_FULL = (
    "<s>[INST] question [/INST] Sentiment: positive. "
    "This statement reflects favorable financial conditions."
)
_EXPECTED_ANSWER = (
    "Sentiment: positive. This statement reflects favorable financial conditions."
)


def _make_pipeline():
    """Return a pipeline dict whose model and tokenizer are MagicMocks.

    tokenizer(prompt, return_tensors="pt").to(device) returns a plain dict so
    that **inputs unpacking in model.generate works without magic-mock issues.
    model.generate returns a list-of-lists (batch of token-id sequences).
    tokenizer.decode returns the full decoded string including the prompt prefix.
    """
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    mock_inputs = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
    mock_tokenizer.return_value.to.return_value = mock_inputs
    mock_model.generate.return_value = [[101, 102, 103]]
    mock_tokenizer.decode.return_value = _DECODED_FULL

    return {"model": mock_model, "tokenizer": mock_tokenizer}


class _FakeStreamer:
    """Stand-in for TextIteratorStreamer that yields predefined tokens synchronously."""

    def __init__(self, *args, **kwargs):
        self._tokens = ["Sentiment:", " positive.", " Favorable."]

    def __iter__(self):
        return iter(self._tokens)


def _fake_transformers(extra_attrs=None):
    """Return a MagicMock that stands in as the transformers module."""
    m = MagicMock()
    if extra_attrs:
        for k, v in extra_attrs.items():
            setattr(m, k, v)
    return m


# ---------------------------------------------------------------------------
# _generate — real (non-MOCK_MODE) path
# ---------------------------------------------------------------------------

class TestGenerateRealPath:
    """_generate exercises the transformers code path when MOCK_MODE=False.

    The `from transformers import pipeline as hf_pipeline` line inside _generate
    is dead code (the import is never used), but it still runs.  We inject a
    fake transformers module via patch.dict so the real package is never loaded.
    """

    def _call(self, monkeypatch, question="Classify this statement.", max_tokens=64):
        import app.main_ecs as m
        p = _make_pipeline()
        monkeypatch.setattr(m, "MOCK_MODE", False)
        monkeypatch.setattr(m, "pipeline", p)

        with patch.dict(sys.modules, {"transformers": _fake_transformers()}):
            result = m._generate(question, max_tokens)

        return result, p

    def test_prompt_contains_inst_tags(self, monkeypatch):
        _, p = self._call(monkeypatch)
        prompt = p["tokenizer"].call_args[0][0]
        assert "[INST]" in prompt
        assert "[/INST]" in prompt

    def test_prompt_contains_question(self, monkeypatch):
        question = "Classify the sentiment of this filing."
        _, p = self._call(monkeypatch, question=question)
        prompt = p["tokenizer"].call_args[0][0]
        assert question in prompt

    def test_tokenizer_called_with_pt_tensors(self, monkeypatch):
        _, p = self._call(monkeypatch)
        _, kwargs = p["tokenizer"].call_args
        assert kwargs.get("return_tensors") == "pt"

    def test_max_tokens_forwarded(self, monkeypatch):
        _, p = self._call(monkeypatch, max_tokens=128)
        _, kwargs = p["model"].generate.call_args
        assert kwargs["max_new_tokens"] == 128

    def test_do_sample_is_false(self, monkeypatch):
        _, p = self._call(monkeypatch)
        _, kwargs = p["model"].generate.call_args
        assert kwargs["do_sample"] is False

    def test_decode_called_on_first_output_sequence(self, monkeypatch):
        """tokenizer.decode must receive output[0], not the full batch tensor."""
        _, p = self._call(monkeypatch)
        decode_first_arg = p["tokenizer"].decode.call_args[0][0]
        assert decode_first_arg == [101, 102, 103]

    def test_answer_extracted_after_inst_tag(self, monkeypatch):
        """_generate strips the prompt prefix by splitting on [/INST]."""
        result, _ = self._call(monkeypatch)
        assert result == _EXPECTED_ANSWER

    def test_answer_does_not_include_prompt_prefix(self, monkeypatch):
        result, _ = self._call(monkeypatch)
        assert "[INST]" not in result
        assert "[/INST]" not in result


# ---------------------------------------------------------------------------
# _stream_into_queue — real (non-MOCK_MODE) path
# ---------------------------------------------------------------------------

class TestStreamIntoQueueRealPath:
    """_stream_into_queue exercises the streaming code path when MOCK_MODE=False."""

    def _call(self, monkeypatch, question="Classify this.", max_tokens=64):
        import app.main_ecs as m
        p = _make_pipeline()
        monkeypatch.setattr(m, "MOCK_MODE", False)
        monkeypatch.setattr(m, "pipeline", p)

        q: Queue = Queue()
        done = Event()

        mock_tf = _fake_transformers({"TextIteratorStreamer": _FakeStreamer})

        with patch.dict(sys.modules, {"transformers": mock_tf}):
            t = threading.Thread(
                target=m._stream_into_queue,
                args=(question, max_tokens, True, q, done, "test-request-id"),
            )
            t.start()
            t.join(timeout=5.0)

        return q, done, p

    def test_done_event_is_set(self, monkeypatch):
        _, done, _ = self._call(monkeypatch)
        assert done.is_set()

    def test_tokens_are_queued(self, monkeypatch):
        q, _, _ = self._call(monkeypatch)
        tokens = list(q.queue)
        assert len(tokens) > 0

    def test_queued_tokens_match_streamer_output(self, monkeypatch):
        q, _, _ = self._call(monkeypatch)
        tokens = list(q.queue)
        assert tokens == ["Sentiment:", " positive.", " Favorable."]

    def test_model_generate_called(self, monkeypatch):
        _, _, p = self._call(monkeypatch)
        assert p["model"].generate.called

    def test_max_tokens_forwarded_to_generate(self, monkeypatch):
        _, _, p = self._call(monkeypatch, max_tokens=128)
        _, kwargs = p["model"].generate.call_args
        assert kwargs["max_new_tokens"] == 128

    def test_prompt_contains_question(self, monkeypatch):
        question = "What is the market outlook?"
        _, _, p = self._call(monkeypatch, question=question)
        prompt = p["tokenizer"].call_args[0][0]
        assert question in prompt

    def test_prompt_contains_inst_tags(self, monkeypatch):
        _, _, p = self._call(monkeypatch)
        prompt = p["tokenizer"].call_args[0][0]
        assert "[INST]" in prompt
        assert "[/INST]" in prompt

    def test_do_sample_is_false(self, monkeypatch):
        _, _, p = self._call(monkeypatch)
        _, kwargs = p["model"].generate.call_args
        assert kwargs["do_sample"] is False


# ---------------------------------------------------------------------------
# lifespan — adapter-loading branches when MOCK_MODE=False
# ---------------------------------------------------------------------------

class TestLifespanAdapterBranches:
    """Verify the two adapter-loading branches in the non-MOCK_MODE lifespan.

    The lifespan imports transformers, torch, and peft inside its body.  We
    inject fake modules for all three via patch.dict so no real model weights
    are loaded and the C-extension re-import problem with torch is avoided.
    """

    def _mock_tf(self, mock_causal_cls, mock_tok_cls):
        tf = MagicMock()
        tf.AutoModelForCausalLM = mock_causal_cls
        tf.AutoTokenizer = mock_tok_cls
        return tf

    def _mock_peft(self, mock_peft_cls):
        peft = MagicMock()
        peft.PeftModel = mock_peft_cls
        return peft

    def _mock_torch(self):
        """Minimal torch stub: is_available returns False, dtype attrs are stubs."""
        mt = MagicMock()
        mt.cuda.is_available.return_value = False
        mt.float32 = "float32"
        mt.float16 = "float16"
        return mt

    def test_peft_model_loaded_when_adapter_exists(self, tmp_path, monkeypatch):
        """PeftModel.from_pretrained must be called when the adapter directory exists."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()

        mock_base = MagicMock()
        mock_causal_cls = MagicMock()
        mock_causal_cls.from_pretrained.return_value = mock_base
        mock_tok_cls = MagicMock()
        mock_tok_cls.from_pretrained.return_value = MagicMock()
        mock_peft_cls = MagicMock()

        import app.main_ecs as m
        monkeypatch.setattr(m, "MOCK_MODE", False)
        monkeypatch.setattr(m, "ADAPTER_PATH", str(adapter_dir))
        monkeypatch.setattr(m, "pipeline", None)

        fake_mods = {
            "transformers": self._mock_tf(mock_causal_cls, mock_tok_cls),
            "peft": self._mock_peft(mock_peft_cls),
            "torch": self._mock_torch(),
        }
        with patch.dict(sys.modules, fake_mods):
            with TestClient(m.app) as c:
                r = c.get("/health")

        assert r.status_code == 200
        mock_peft_cls.from_pretrained.assert_called_once_with(mock_base, str(adapter_dir))

    def test_peft_model_skipped_when_adapter_missing(self, tmp_path, monkeypatch):
        """PeftModel.from_pretrained must NOT be called when adapter path doesn't exist."""
        adapter_dir = str(tmp_path / "no-adapter")

        mock_base = MagicMock()
        mock_causal_cls = MagicMock()
        mock_causal_cls.from_pretrained.return_value = mock_base
        mock_tok_cls = MagicMock()
        mock_tok_cls.from_pretrained.return_value = MagicMock()
        mock_peft_cls = MagicMock()

        import app.main_ecs as m
        monkeypatch.setattr(m, "MOCK_MODE", False)
        monkeypatch.setattr(m, "ADAPTER_PATH", adapter_dir)
        monkeypatch.setattr(m, "pipeline", None)

        fake_mods = {
            "transformers": self._mock_tf(mock_causal_cls, mock_tok_cls),
            "peft": self._mock_peft(mock_peft_cls),
            "torch": self._mock_torch(),
        }
        with patch.dict(sys.modules, fake_mods):
            with TestClient(m.app) as c:
                r = c.get("/health")

        assert r.status_code == 200
        mock_peft_cls.from_pretrained.assert_not_called()

    def test_pipeline_reports_model_loaded_after_real_lifespan(self, tmp_path, monkeypatch):
        """After the real lifespan completes, /health must report model_loaded=True."""
        mock_base = MagicMock()
        mock_causal_cls = MagicMock()
        mock_causal_cls.from_pretrained.return_value = mock_base
        mock_tok_cls = MagicMock()
        mock_tok_cls.from_pretrained.return_value = MagicMock()
        mock_peft_cls = MagicMock()

        import app.main_ecs as m
        monkeypatch.setattr(m, "MOCK_MODE", False)
        monkeypatch.setattr(m, "ADAPTER_PATH", str(tmp_path / "no-adapter"))
        monkeypatch.setattr(m, "pipeline", None)

        fake_mods = {
            "transformers": self._mock_tf(mock_causal_cls, mock_tok_cls),
            "peft": self._mock_peft(mock_peft_cls),
            "torch": self._mock_torch(),
        }
        with patch.dict(sys.modules, fake_mods):
            with TestClient(m.app) as c:
                r = c.get("/health")

        assert r.status_code == 200
        assert r.json()["model_loaded"] is True

    def test_load_failure_logs_and_reraises(self, tmp_path, monkeypatch, caplog):
        """A model-load failure at startup must be logged with context, not swallowed."""
        mock_causal_cls = MagicMock()
        mock_causal_cls.from_pretrained.side_effect = RuntimeError("weights corrupted")
        mock_tok_cls = MagicMock()
        mock_peft_cls = MagicMock()

        import app.main_ecs as m
        monkeypatch.setattr(m, "MOCK_MODE", False)
        monkeypatch.setattr(m, "ADAPTER_PATH", str(tmp_path / "no-adapter"))
        monkeypatch.setattr(m, "pipeline", None)

        fake_mods = {
            "transformers": self._mock_tf(mock_causal_cls, mock_tok_cls),
            "peft": self._mock_peft(mock_peft_cls),
            "torch": self._mock_torch(),
        }
        with patch.dict(sys.modules, fake_mods):
            with caplog.at_level("ERROR", logger="app.main_ecs"):
                with pytest.raises(RuntimeError, match="weights corrupted"):
                    with TestClient(m.app):
                        pass

        assert "failed to load model" in caplog.text.lower()
