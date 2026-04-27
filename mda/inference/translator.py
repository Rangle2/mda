import json
import os
import warnings


class MDATranslator:
    def __init__(self):
        try:
            from deep_translator import GoogleTranslator as _GT
            _ = _GT
            self._has_translator = True
        except ImportError:
            self._has_translator = False
            warnings.warn(
                "deep_translator is not installed — translation is disabled. "
                "Non-English input will be passed to the engine untranslated, "
                "which may degrade entity lookup and answer quality.\n"
                "Install with: pip install deep-translator",
                stacklevel=2,
            )
        self._cache: dict[str, str] = {}

    def to_english(self, text: str) -> str:
        if not self._has_translator:
            return text
        if self._is_english(text):
            return text
        return self._translate(text, source="auto", target="en")

    def to_user_lang(self, text: str, target: str = "tr") -> str:
        if not self._has_translator:
            return text
        if target == "en":
            return text
        return self._translate(text, source="en", target=target)

    # ASCII-only Turkish function words that can't appear in English
    _TR_MARKERS = frozenset({
        "nedir", "neden", "nasil", "hangi", "kimdir", "nerede", "ne",
        "bir", "olan", "olan", "icin", "ile", "gibi", "daha", "ama",
        "veya", "ancak", "fakat", "yani", "sonra", "once", "bunu",
        "bunu", "bize", "sizi", "onun", "bunun", "olarak", "kadar",
        "degil", "evet", "hayir", "tamam",
    })

    def _is_english(self, text: str) -> bool:
        tr_chars = set("çğışöüÇĞİŞÖÜ")
        if any(c in tr_chars for c in text):
            return False
        words = text.lower().split()
        # Check for Turkish function words (ASCII-only Turkish)
        if any(w.strip(".,!?;:") in self._TR_MARKERS for w in words):
            return False
        ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
        if ascii_ratio < 0.90:
            return False
        # Turkish morphological suffixes on the last word signal Turkish
        _TR_SUFFIXES = ("dur", "dır", "dir", "tır", "tir", "tur", "tür",
                        "dur", "lar", "ler", "dan", "den", "tan", "ten",
                        "da", "de", "ta", "te", "nin", "nun", "nün", "nın",
                        "ya", "ye", "a", "e", "ı", "i", "u", "ü")
        last = words[-1].strip(".,!?;:").lower() if words else ""
        if len(last) > 4 and any(last.endswith(s) for s in _TR_SUFFIXES):
            # Verify with langdetect before committing — suffix alone can FP
            try:
                from langdetect import detect
                if detect(text) != "en":
                    return False
            except Exception:
                pass
        # Use langdetect only for texts long enough to be reliable (8+ words)
        if len(words) >= 8:
            try:
                from langdetect import detect
                return detect(text) == "en"
            except Exception:
                pass
        return ascii_ratio > 0.95

    # Common sentence-starter words that should NOT be treated as proper nouns
    _SENTENCE_STARTERS = frozenset([
        # Turkish
        "Bu", "Bir", "Ve", "Ya", "Ama", "Fakat", "Ancak", "Çünkü", "Eğer",
        "Şu", "Her", "Hiç", "Bazı", "Tüm", "Hem", "Ne", "Nasıl", "Neden",
        # English articles, pronouns, prepositions, conjunctions
        "The", "A", "An", "This", "That", "These", "Those",
        "It", "He", "She", "We", "They", "I", "You",
        "Its", "His", "Her", "Our", "Their", "My", "Your",
        "In", "On", "At", "For", "With", "From", "To", "Of", "By", "As",
        "But", "And", "Or", "So", "Yet", "Nor", "If", "When", "Where",
        "While", "Although", "Because", "Since", "Unless", "Until",
        # English auxiliary verbs
        "Is", "Are", "Was", "Were", "Be", "Been", "Being",
        "Has", "Have", "Had", "Do", "Does", "Did",
        "Will", "Would", "Could", "Should", "Shall", "May", "Might", "Must",
        "Can", "Need", "Dare", "Used",
    ])
    # Turkish verb/copula/possession suffixes that indicate a common word, not a name.
    # Used to filter words like "Abisidir" (is his brother), "Mimarisi" (its architecture)
    # from being treated as the start of a proper noun phrase.
    # Rule: suffix must account for >2 chars extra (len > len(suffix)+2) to avoid
    # filtering short genuine names like "Kadir" (len=5, suffix "dir" len=3 → 5>5 False).
    _TR_COMMON_SUFFIXES = (
        "dır", "dir", "dur", "dür",   # copula "is/are"
        "tır", "tir", "tur", "tür",   # copula variant
        "ydı", "ydi", "ydu", "ydü",   # past copula "was"
        "ydık", "ydiniz",             # past copula plural
        "nın", "nin", "nun", "nün",   # genitive suffix
        "sının", "sinin",             # genitive + possessive
        "ları", "leri",               # plural + possessive
        "ıdır", "idir",               # extended copula
    )

    def _is_likely_proper_name(self, word: str) -> bool:
        """Return False if the word clearly ends with a Turkish grammatical suffix,
        indicating it is a common word rather than a proper noun."""
        w = word.lower()
        for sfx in self._TR_COMMON_SUFFIXES:
            if w.endswith(sfx) and len(word) > len(sfx) + 2:
                return False
        return True
    def _extract_proper_nouns(self, text: str) -> list[str]:
        import re
        seen: set[str] = set()
        ordered: list[str] = []

        # Pass 1: multi-word proper noun phrases
        # Match 2-3 consecutive capitalized words on the same line
        for m in re.finditer(
            r'[A-ZÇĞİÖŞÜ][a-zçğışöüA-ZÇĞİÖŞÜ]+'
            r'(?:[ \t]+[A-ZÇĞİÖŞÜ][a-zçğışöüA-ZÇĞİÖŞÜ]+){1,2}',
            text,
        ):
            phrase = m.group()
            words = phrase.split()
            # Keep phrase if:
            # - at least one word is not a sentence starter
            # - the FIRST word looks like a proper name (not a Turkish common word ending)
            if (any(w not in self._SENTENCE_STARTERS for w in words)
                    and self._is_likely_proper_name(words[0])):
                if phrase not in seen:
                    seen.add(phrase)
                    ordered.append(phrase)

        # Pass 2: individual capitalized words (also catches words already in phrases
        # so they can be masked when they appear standalone)
        for m in re.finditer(r'[A-ZÇĞİÖŞÜ][a-zçğışöüA-ZÇĞİÖŞÜ]+', text):
            word = m.group()
            if word in self._SENTENCE_STARTERS:
                continue
            if word not in seen:
                seen.add(word)
                ordered.append(word)

        # Pass 3: acronyms
        for acr in re.findall(r'\b[A-Z]{2,}\b', text):
            if acr not in seen:
                seen.add(acr)
                ordered.append(acr)

        return ordered

    def _translate(self, text: str, source: str, target: str) -> str:
        key = f"{source}:{target}:{text}"
        if key in self._cache:
            return self._cache[key]
        try:
            import re
            from deep_translator import GoogleTranslator

            proper_nouns = self._extract_proper_nouns(text)

            ph_map: dict[str, str] = {}       # placeholder  → original
            word_to_ph: dict[str, str] = {}   # original     → placeholder
            for i, word in enumerate(proper_nouns):
                ph = f"__PN{i}__"
                ph_map[ph] = word
                word_to_ph[word] = ph

            masked = text
            # Substitute longest tokens first — prevents a shorter word
            for word, ph in sorted(word_to_ph.items(), key=lambda x: -len(x[0])):
                masked = re.sub(rf'\b{re.escape(word)}\b', ph, masked)

            translated = GoogleTranslator(source=source, target=target).translate(masked)
            if not translated:
                return text

            # Restore: tolerant regex handles spaces Google may insert around digits
            # e.g. "__ PN 0 __" or "__pn0__" are all matched
            def _restore(m: re.Match) -> str:
                digits = re.search(r'\d+', m.group(0))
                if digits:
                    ph = f"__PN{digits.group()}__"
                    return ph_map.get(ph, m.group(0))
                return m.group(0)

            result = re.sub(r'__\s*PN\s*\d+\s*__', _restore, translated, flags=re.IGNORECASE)

            self._cache[key] = result
            return result
        except Exception as exc:
            warnings.warn(
                f"Translation failed ({source}->{target}): {exc}. "
                "Returning original text -- output quality may be reduced.",
                stacklevel=3,
            )
            return text

    def save_cache(self, path: str) -> None:
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)

    def load_cache(self, path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path, encoding="utf-8") as f:
            self._cache = json.load(f)
