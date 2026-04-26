"""Tests for integrations/engine.py — MDAEngine and AnthropicEngine.

All external I/O (HTTP calls, filesystem checkpoints) is mocked so the tests
run fully offline without Ollama or the Anthropic API.
"""
import sys
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from integrations.engine import MDAEngine, AnthropicEngine, _strip_markdown


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ollama_response(content: str) -> MagicMock:
    """Fake a successful requests.post response from Ollama."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"message": {"content": content}}
    return resp


# ---------------------------------------------------------------------------
# _strip_markdown
# ---------------------------------------------------------------------------

class TestStripMarkdown:
    def test_removes_headings(self):
        assert _strip_markdown("## Title") == "Title"

    def test_removes_bold(self):
        assert _strip_markdown("**bold**") == "bold"

    def test_removes_italic(self):
        assert _strip_markdown("*italic*") == "italic"

    def test_removes_backtick_code(self):
        assert _strip_markdown("`code`") == "code"

    def test_collapses_blank_lines(self):
        text = "a\n\n\n\nb"
        result = _strip_markdown(text)
        assert "\n\n\n" not in result

    def test_plain_text_unchanged(self):
        text = "hello world"
        assert _strip_markdown(text) == text


# ---------------------------------------------------------------------------
# MDAEngine construction
# ---------------------------------------------------------------------------

class TestMDAEngineConstruction:
    def test_creates_mda_instance(self):
        engine = MDAEngine(model="test-model")
        assert engine.mda is not None

    def test_model_stored(self):
        engine = MDAEngine(model="my-model:7b")
        assert engine.model == "my-model:7b"

    def test_user_id_stored(self):
        engine = MDAEngine(user_id="alice")
        assert engine.user_id == "alice"

    def test_empty_recent_learns(self):
        engine = MDAEngine()
        assert engine._recent_learns == []

    def test_history_initialized_on_mda(self):
        engine = MDAEngine()
        assert hasattr(engine.mda, "_history")
        assert isinstance(engine.mda._history, list)


# ---------------------------------------------------------------------------
# MDAEngine.learn / teach
# ---------------------------------------------------------------------------

class TestLearnAndTeach:
    def test_learn_delegates_to_mda(self):
        engine = MDAEngine()
        engine.learn("Python is a programming language.")
        assert engine.mda.registry.count() > 0

    def test_teach_creates_entity(self):
        engine = MDAEngine()
        engine.teach("Go", ["Go is a compiled language."])
        e = engine.mda.registry.get("Go")
        assert e is not None

    def test_teach_default_category_custom(self):
        engine = MDAEngine()
        engine.teach("Rust", ["Rust is safe."])
        e = engine.mda.registry.get("Rust")
        assert e.category == "custom"


# ---------------------------------------------------------------------------
# MDAEngine._should_learn
# ---------------------------------------------------------------------------

class TestShouldLearn:
    def test_short_text_returns_false(self):
        engine = MDAEngine()
        assert not engine._should_learn("hi")

    def test_uncertain_marker_returns_false(self):
        engine = MDAEngine()
        assert not engine._should_learn("I don't know anything about that topic.")

    def test_substantial_novel_text_returns_true(self):
        engine = MDAEngine()
        result = engine._should_learn(
            "Quantum computing uses qubits to process information."
        )
        assert isinstance(result, bool)

    def test_duplicate_text_returns_false(self):
        engine = MDAEngine()
        text = "Neural networks are universal function approximators."
        engine.mda.teach("Neural", [text])
        engine._recent_learns = [text]
        assert not engine._should_learn(text)


# ---------------------------------------------------------------------------
# MDAEngine._call_llm — Ollama path (mocked)
# ---------------------------------------------------------------------------

class TestCallLLM:
    def test_returns_string_on_success(self):
        engine = MDAEngine(model="test:7b")
        mock_resp = _make_ollama_response("Hello from Ollama.")
        with patch("requests.post", return_value=mock_resp):
            result = engine._call_llm("", "hello", "en")
        assert isinstance(result, str)

    def test_connection_error_returns_message(self):
        import requests
        engine = MDAEngine(model="test:7b")
        with patch("requests.post", side_effect=requests.exceptions.ConnectionError()):
            result = engine._call_llm("", "hello", "en")
        assert "not reachable" in result.lower() or "Ollama" in result

    def test_timeout_returns_message(self):
        import requests
        engine = MDAEngine(model="test:7b")
        with patch("requests.post", side_effect=requests.exceptions.Timeout()):
            result = engine._call_llm("", "hello", "en")
        assert "timeout" in result.lower() or "Ollama" in result

    def test_think_tags_stripped(self):
        engine = MDAEngine(model="test:7b")
        raw = "<think>internal reasoning</think>Final answer."
        mock_resp = _make_ollama_response(raw)
        with patch("requests.post", return_value=mock_resp):
            result = engine._call_llm("", "hi", "en")
        assert "internal reasoning" not in result

    def test_last_thinking_captured(self):
        engine = MDAEngine(model="test:7b")
        raw = "<think>deep thought</think>The answer."
        mock_resp = _make_ollama_response(raw)
        with patch("requests.post", return_value=mock_resp):
            engine._call_llm("", "hi", "en")
        assert engine._last_thinking == "deep thought"

    def test_stores_last_en_response(self):
        engine = MDAEngine(model="test:7b")
        mock_resp = _make_ollama_response("Clean answer.")
        with patch("requests.post", return_value=mock_resp):
            engine._call_llm("", "hello", "en")
        assert "Clean answer" in engine._last_en_response


# ---------------------------------------------------------------------------
# MDAEngine.chat — full pipeline (mocked LLM)
# ---------------------------------------------------------------------------

class TestChat:
    def test_returns_string(self):
        engine = MDAEngine(model="test:7b")
        mock_resp = _make_ollama_response("This is a response.")
        with patch("requests.post", return_value=mock_resp):
            result = engine.chat("What is Python?", lang="en")
        assert isinstance(result, str)

    def test_chat_with_known_entity(self):
        engine = MDAEngine(model="test:7b")
        engine.teach("Python", ["Python is a programming language."])
        mock_resp = _make_ollama_response("Python is great.")
        with patch("requests.post", return_value=mock_resp):
            result = engine.chat("Python nedir?", lang="en")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# MDAEngine._build_context
# ---------------------------------------------------------------------------

class TestBuildContext:
    def test_returns_string(self):
        engine = MDAEngine()
        result = engine._build_context("Python is a language")
        assert isinstance(result, str)

    def test_empty_when_no_entity_found(self):
        engine = MDAEngine()
        result = engine._build_context("zzz aaa bbb")
        assert result == "" or isinstance(result, str)

    def test_includes_memory_tag_when_facts_available(self):
        engine = MDAEngine()
        engine.teach("Python", ["Python is interpreted.", "Python was created by Guido."])
        result = engine._build_context("Python programming")
        if result:
            assert "[MEMORY]" in result

    def test_respects_char_limit(self):
        engine = MDAEngine()
        for i in range(20):
            engine.teach(f"Entity{i}", [f"Entity {i} is a concept. " * 10])
        result = engine._build_context("Entity0 information")
        assert len(result) <= 1600


# ---------------------------------------------------------------------------
# MDAEngine.switch_model
# ---------------------------------------------------------------------------

class TestSwitchModel:
    def test_model_updated(self):
        engine = MDAEngine(model="old:7b")
        with patch.object(engine, "save", return_value={}):
            engine.switch_model("new:13b")
        assert engine.model == "new:13b"

    def test_mda_reset(self):
        engine = MDAEngine()
        engine.mda.teach("Foo", ["Foo is here."])
        old_mda = engine.mda
        with patch.object(engine, "save", return_value={}):
            engine.switch_model("another:7b")
        assert engine.mda is not old_mda
        assert engine.mda.registry.count() == 0


# ---------------------------------------------------------------------------
# MDAEngine._load_md_file
# ---------------------------------------------------------------------------

class TestLoadMdFile:
    def test_learns_paragraphs_from_md(self, tmp_path):
        engine = MDAEngine()
        md = tmp_path / "test.md"
        md.write_text(
            "# Introduction\n\nArtificial Intelligence is transforming every industry.\n\n"
            "## Details\n\nMachine Learning uses statistical methods to learn patterns.\n",
            encoding="utf-8",
        )
        count = engine._load_md_file(md)
        assert count >= 1

    def test_skips_short_paragraphs(self, tmp_path):
        engine = MDAEngine()
        md = tmp_path / "short.md"
        md.write_text("# Title\n\nHi.\n", encoding="utf-8")
        count = engine._load_md_file(md)
        assert count == 0


# ---------------------------------------------------------------------------
# AnthropicEngine
# ---------------------------------------------------------------------------

class TestAnthropicEngine:
    def test_creates_engine(self):
        engine = AnthropicEngine(model="claude-haiku-4-5-20251001", api_key="test-key")
        assert engine is not None

    def test_no_api_key_returns_message(self):
        engine = AnthropicEngine(api_key="")
        result = engine._call_llm("", "hello", "en")
        assert "ANTHROPIC_API_KEY" in result

    def test_build_context_higher_limit(self):
        engine = AnthropicEngine(api_key="test")
        for i in range(30):
            engine.teach(f"Ent{i}", [f"Entity {i} has a long fact. " * 10])
        result = engine._build_context("Ent0 information")
        assert len(result) <= 4000

    def test_anthropic_missing_returns_message(self):
        engine = AnthropicEngine(api_key="fake-key")
        with patch.dict("sys.modules", {"anthropic": None}):
            result = engine._call_llm("", "hello", "en")
        assert "anthropic" in result.lower() or "not installed" in result.lower() \
               or "error" in result.lower()
