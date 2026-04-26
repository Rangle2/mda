"""
integrations/loader.py — file extraction pipeline for MDA.

Parses source files into structured (surface, facts, relations) triples
and feeds them into MDA via teach() / relate().

Supported formats
-----------------
.py          PythonExtractor   — AST-based class/function/import extraction
.md .markdown MarkdownExtractor — heading → entity, paragraph → fact
.txt         PlainTextExtractor — paragraph-based (no header parsing)
.rs .java .ts .go .cpp .c .h .hpp  GenericCodeExtractor — regex-based

Cross-session persistence
-------------------------
_file_hashes tracks md5 digests so only changed files are reprocessed::

    # Agent 1 — indexes the codebase
    engine = MDAEngine(model="...", user_id="project_x")
    engine.loader.load_dir("./src")
    engine.loader.save_state(".mda/project_x.json")

    # Agent 2 — picks up where Agent 1 left off
    engine = MDAEngine(model="...", user_id="project_x")
    engine.loader.load_state(".mda/project_x.json")
    # Already knows everything Agent 1 learned
    response = engine.chat("What does AuthService depend on?")
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from core.bind import bind as _bind

if TYPE_CHECKING:
    from mda import MDA


# ---------------------------------------------------------------------------
# ExtractionResult
# ---------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    """Structured representation of one extracted entity."""
    surface:   str
    facts:     list[str] = field(default_factory=list)
    relations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"surface": self.surface, "facts": self.facts, "relations": self.relations}


# ---------------------------------------------------------------------------
# BaseExtractor
# ---------------------------------------------------------------------------

class BaseExtractor(ABC):
    @abstractmethod
    def can_handle(self, path: str) -> bool: ...

    @abstractmethod
    def extract(self, path: str) -> list[dict]:
        """Returns list of {surface, facts, relations} dicts."""
        ...


# ---------------------------------------------------------------------------
# MarkdownExtractor
# ---------------------------------------------------------------------------

class MarkdownExtractor(BaseExtractor):
    """Headers become entity surfaces; paragraphs under each header become facts."""

    def can_handle(self, path: str) -> bool:
        return Path(path).suffix.lower() in {".md", ".markdown"}

    def extract(self, path: str) -> list[dict]:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")

        sections: dict[str, list[str]] = {}
        order:    list[str]            = []

        current_heading = Path(path).stem   # filename as fallback entity
        current_lines:  list[str] = []
        in_code_block = False

        def _flush() -> None:
            if current_lines:
                para = " ".join(current_lines).strip()
                if para:
                    sections.setdefault(current_heading, []).append(para)
                    if current_heading not in order:
                        order.append(current_heading)
                current_lines.clear()

        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block or stripped.startswith(("---", "|", ">")):
                continue
            if stripped.startswith("#"):
                _flush()
                heading = stripped.lstrip("#").strip()
                if heading:
                    current_heading = heading
                if current_heading not in order:
                    order.append(current_heading)
                continue
            if stripped == "":
                _flush()
                continue
            current_lines.append(stripped)
        _flush()

        results = []
        for heading in order:
            facts = [p for p in sections.get(heading, []) if len(p) >= 20]
            if not facts:
                continue
            # Capitalised words in the text become candidate relation surfaces
            relations = list({
                w.rstrip(".,;:")
                for para in facts
                for w in para.split()
                if w and w[0].isupper()
                and w.rstrip(".,;:") != heading
                and len(w) > 2
            })
            results.append({"surface": heading, "facts": facts, "relations": relations})

        return results


# ---------------------------------------------------------------------------
# PythonExtractor
# ---------------------------------------------------------------------------

def _annotation_str(node) -> str:
    """Best-effort stringification of an AST annotation node."""
    if node is None:
        return ""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _annotation_str(node.value)
    if isinstance(node, ast.Constant):
        return str(node.value)
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _format_sig(func_node) -> str:
    """Format a FunctionDef/AsyncFunctionDef signature as a readable string."""
    parts = []
    for arg in func_node.args.args:
        if arg.arg == "self":
            continue
        ann = _annotation_str(arg.annotation)
        parts.append(f"{arg.arg}: {ann}" if ann else arg.arg)
    ret = _annotation_str(func_node.returns)
    sig = f"{func_node.name}({', '.join(parts)})"
    return f"{sig} -> {ret}" if ret else sig


class PythonExtractor(BaseExtractor):
    """AST-based extraction: classes → entities, methods → facts, imports → relations."""

    def can_handle(self, path: str) -> bool:
        return Path(path).suffix.lower() == ".py"

    def extract(self, path: str) -> list[dict]:
        source = Path(path).read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError:
            return []

        results: list[dict] = []

        # Module-level imports → relation candidates shared by all entities in file
        module_imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_imports.append(alias.asname or alias.name.split(".")[-1])
            elif isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    if alias.name != "*":
                        module_imports.append(alias.asname or alias.name)

        # Top-level classes
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            facts:     list[str] = []
            relations: list[str] = list(module_imports)

            doc = ast.get_docstring(node)
            if doc:
                facts.append(doc[:300])

            for base in node.bases:
                name = _annotation_str(base)
                if name:
                    relations.append(name)

            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    facts.append(f"has method {_format_sig(item)}")
                    for arg in item.args.args:
                        ann = _annotation_str(arg.annotation)
                        if ann and ann[0].isupper():
                            relations.append(ann)
                    ret = _annotation_str(item.returns)
                    if ret and ret[0].isupper():
                        relations.append(ret)

            results.append({
                "surface":   node.name,
                "facts":     facts,
                "relations": list(dict.fromkeys(r for r in relations if r)),
            })

        # Top-level functions
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            facts:     list[str] = []
            relations: list[str] = []

            doc = ast.get_docstring(node)
            if doc:
                facts.append(doc[:300])
            facts.append(f"function {_format_sig(node)}")

            for arg in node.args.args:
                ann = _annotation_str(arg.annotation)
                if ann and ann[0].isupper():
                    relations.append(ann)
            ret = _annotation_str(node.returns)
            if ret and ret[0].isupper():
                relations.append(ret)

            results.append({
                "surface":   node.name,
                "facts":     facts,
                "relations": list(dict.fromkeys(r for r in relations if r)),
            })

        return results


# ---------------------------------------------------------------------------
# GenericCodeExtractor
# ---------------------------------------------------------------------------

_LANG_PATTERNS: dict[str, dict] = {
    ".rs": {
        "entities": [
            r"(?:pub\s+)?struct\s+(\w+)",
            r"(?:pub\s+)?enum\s+(\w+)",
            r"(?:pub\s+)?trait\s+(\w+)",
            r"impl(?:\s+\w+\s+for)?\s+(\w+)",
            r"(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*\(",
        ],
        "sig_pattern": r"(?:pub\s+)?(?:async\s+)?fn\s+(\w+\s*\([^)]{0,120}\)(?:\s*->\s*[\w<>, &:]+)?)",
        "imports":     [r"use\s+([\w:]+)"],
    },
    ".java": {
        "entities": [
            r"(?:public\s+|private\s+|protected\s+)?(?:abstract\s+|final\s+)?class\s+(\w+)",
            r"(?:public\s+)?interface\s+(\w+)",
        ],
        "sig_pattern": r"(?:public|private|protected)\s+(?:static\s+)?[\w<>\[\]]+\s+(\w+\s*\([^)]{0,120}\))",
        "imports":     [r"import\s+([\w.]+);"],
    },
    ".ts": {
        "entities": [
            r"(?:export\s+)?(?:abstract\s+)?class\s+(\w+)",
            r"(?:export\s+)?interface\s+(\w+)",
            r"(?:export\s+)?(?:async\s+)?function\s+(\w+)",
        ],
        "sig_pattern": r"(?:async\s+)?function\s+(\w+\s*\([^)]{0,120}\)(?:\s*:\s*[\w<>|, ]+)?)",
        "imports":     [r"from\s+['\"]([^'\"]+)['\"]"],
    },
    ".go": {
        "entities": [
            r"type\s+(\w+)\s+struct",
            r"type\s+(\w+)\s+interface",
            r"func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(",
        ],
        "sig_pattern": r"func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+\s*\([^)]{0,120}\)(?:\s*[\w*()\s,]+)?)",
        "imports":     [r'"([\w./]+)"'],
    },
    ".cpp": {
        "entities": [
            r"(?:class|struct)\s+(\w+)",
            r"(?:[\w:*&<>]+\s+)+(\w+)\s*\([^)]*\)\s*(?:const\s*)?(?:\{|;)",
        ],
        "sig_pattern": r"(?:[\w:*&<>]+\s+)+(\w+\s*\([^)]{0,120}\)(?:\s*const)?)",
        "imports":     [r'#include\s+[<"]([^>"]+)[>"]'],
    },
    ".c": {
        "entities": [
            r"(?:struct|enum|union)\s+(\w+)",
            r"(?:[\w*]+\s+)+(\w+)\s*\([^)]*\)\s*\{",
        ],
        "sig_pattern": r"(?:[\w*]+\s+)+(\w+\s*\([^)]{0,80}\))",
        "imports":     [r'#include\s+[<"]([^>"]+)[>"]'],
    },
}
_LANG_PATTERNS[".h"]   = _LANG_PATTERNS[".cpp"]
_LANG_PATTERNS[".hpp"] = _LANG_PATTERNS[".cpp"]


class GenericCodeExtractor(BaseExtractor):
    """Regex-based extractor for Rust, Java, TypeScript, Go, C, C++."""

    _SUPPORTED = frozenset(_LANG_PATTERNS)

    def can_handle(self, path: str) -> bool:
        return Path(path).suffix.lower() in self._SUPPORTED

    def extract(self, path: str) -> list[dict]:
        suffix = Path(path).suffix.lower()
        patterns = _LANG_PATTERNS.get(suffix)
        if not patterns:
            return []

        try:
            source = Path(path).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return []

        # Collect entity names
        entity_facts: dict[str, list[str]] = {}
        for pat in patterns["entities"]:
            for m in re.finditer(pat, source, re.MULTILINE):
                name = m.group(1).strip()
                if name and len(name) > 1:
                    entity_facts.setdefault(name, [])

        # Attribute signatures to matching entity names
        sig_pat = patterns.get("sig_pattern")
        if sig_pat:
            for m in re.finditer(sig_pat, source, re.MULTILINE):
                sig = m.group(1).strip()
                for name in entity_facts:
                    if sig.startswith(name):
                        entity_facts[name].append(f"signature: {sig}")
                        break

        # Collect imports as flat relation names
        imports: list[str] = []
        for pat in patterns["imports"]:
            for m in re.finditer(pat, source, re.MULTILINE):
                raw  = m.group(1)
                leaf = re.split(r"[./\\:{}]", raw)[-1].strip()
                if leaf and len(leaf) > 1:
                    imports.append(leaf)
        imports = list(dict.fromkeys(imports))

        filename = Path(path).name
        results = []
        for name, facts in entity_facts.items():
            if not facts:
                facts = [f"{name} is defined in {filename}"]
            results.append({"surface": name, "facts": facts, "relations": imports})

        return results


# ---------------------------------------------------------------------------
# PlainTextExtractor
# ---------------------------------------------------------------------------

class PlainTextExtractor(BaseExtractor):
    """Paragraph-based extraction for .txt files; filename becomes the entity surface."""

    def can_handle(self, path: str) -> bool:
        return Path(path).suffix.lower() == ".txt"

    def extract(self, path: str) -> list[dict]:
        text    = Path(path).read_text(encoding="utf-8", errors="ignore")
        surface = Path(path).stem.replace("_", " ").replace("-", " ").title()

        facts:   list[str] = []
        current: list[str] = []

        for line in text.splitlines():
            stripped = line.strip()
            if stripped == "":
                if current:
                    para = " ".join(current)
                    if len(para) >= 20:
                        facts.append(para)
                    current = []
            else:
                current.append(stripped)
        if current:
            para = " ".join(current)
            if len(para) >= 20:
                facts.append(para)

        if not facts:
            return []

        relations = list({
            w.rstrip(".,;:")
            for para in facts
            for w in para.split()
            if w and w[0].isupper()
            and w.rstrip(".,;:") != surface
            and len(w) > 2
        })

        return [{"surface": surface, "facts": facts, "relations": relations}]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


_DEFAULT_EXTENSIONS = frozenset({
    ".py", ".md", ".markdown", ".txt",
    ".rs", ".java", ".ts", ".go",
    ".cpp", ".c", ".h", ".hpp",
})


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class Loader:
    """
    File extraction pipeline that feeds any source file into MDA as
    structured entities, facts, and relations.

    Usage example::

        # Agent 1 — indexes the codebase
        engine = MDAEngine(model="...", user_id="project_x")
        engine.loader.load_dir("./src")
        engine.loader.save_state(".mda/project_x.json")

        # Agent 2 — picks up where Agent 1 left off
        engine = MDAEngine(model="...", user_id="project_x")
        engine.loader.load_state(".mda/project_x.json")
        # Already knows everything Agent 1 learned
        response = engine.chat("What does AuthService depend on?")
    """

    def __init__(self, mda: "MDA") -> None:
        self.mda = mda
        self._extractors: list[BaseExtractor] = [
            PythonExtractor(),
            MarkdownExtractor(),
            GenericCodeExtractor(),
            PlainTextExtractor(),
        ]
        self._file_hashes: dict[str, str] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _find_extractor(self, path: str) -> BaseExtractor | None:
        for e in self._extractors:
            if e.can_handle(path):
                return e
        return None

    def _feed(self, results: list[dict]) -> int:
        """Teach extracted results to MDA. Returns total fact count."""
        count = 0
        for item in results:
            surface   = (item.get("surface") or "").strip()
            facts     = [f for f in item.get("facts", [])     if f and len(f) >= 10]
            relations = [r for r in item.get("relations", []) if r and r != surface]
            if not surface or not facts:
                continue
            self.mda.teach(surface, facts)
            count += len(facts)
            for rel in relations:
                self.mda.relate(surface, rel)
        return count

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_file(self, path: str) -> int:
        """
        Load a single file into MDA.

        Returns the number of facts learned, or 0 if the file is unchanged
        since the last load (hash match) or has no registered extractor.
        """
        path = str(Path(path).resolve())
        try:
            current_hash = _md5(path)
        except OSError:
            return 0

        if self._file_hashes.get(path) == current_hash:
            return 0

        extractor = self._find_extractor(path)
        if extractor is None:
            return 0

        try:
            results = extractor.extract(path)
        except Exception:
            return 0

        with self._lock:
            count = self._feed(results)
            self._file_hashes[path] = current_hash

        return count

    def load_dir(
        self,
        directory: str,
        extensions: list[str] | None = None,
        recursive: bool = True,
    ) -> dict[str, int]:
        """
        Load all supported files in *directory*.

        Args:
            directory:  Root directory to scan.
            extensions: Explicit extension whitelist (e.g. ``[".py", ".md"]``).
                        Defaults to all supported extensions.
            recursive:  Whether to descend into subdirectories.

        Returns:
            ``{path: fact_count}`` for every file attempted.
        """
        allowed = frozenset(e.lower() for e in (extensions or _DEFAULT_EXTENSIONS))
        root    = Path(directory)
        glob    = "**/*" if recursive else "*"
        paths   = [
            str(p) for p in root.glob(glob)
            if p.is_file() and p.suffix.lower() in allowed
        ]

        outcomes: dict[str, int] = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_path = {executor.submit(self.load_file, p): p for p in paths}
            for future in as_completed(future_to_path):
                p = future_to_path[future]
                try:
                    outcomes[p] = future.result()
                except Exception:
                    outcomes[p] = 0

        self.mda.registry.build_cross_entity_synapses(
            bind_fn=_bind,
            min_use_count=2,
        )

        return outcomes

    def reload_changed(self, directory: str) -> dict[str, int]:
        """
        Re-scan *directory* and reload only files whose md5 hash has
        changed since the last :meth:`load_file` / :meth:`load_dir` call.

        Returns:
            ``{path: fact_count}`` for changed files only.
        """
        root    = Path(directory)
        changed: dict[str, int] = {}

        for p in root.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in _DEFAULT_EXTENSIONS:
                continue
            path_str = str(p.resolve())
            try:
                h = _md5(path_str)
            except OSError:
                continue
            if self._file_hashes.get(path_str) != h:
                count = self.load_file(path_str)
                if count > 0:
                    changed[path_str] = count

        return changed

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def save_state(self, path: str) -> None:
        """
        Persist ``_file_hashes`` to *path* as JSON.

        Pair with ``mda.save()`` for full cross-session persistence::

            engine.mda.save("checkpoint")
            engine.loader.save_state(".mda/hashes.json")
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8") as f:
            json.dump(self._file_hashes, f, indent=2)

    def load_state(self, path: str) -> None:
        """
        Restore ``_file_hashes`` from *path*.

        After loading, files already recorded will be skipped by
        :meth:`load_file` and :meth:`load_dir`, so the agent resumes
        without re-indexing the full codebase.
        """
        with open(path, "r", encoding="utf-8") as f:
            self._file_hashes = json.load(f)
