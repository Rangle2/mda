"""
title: MDA Memory Filter
author: Rangle
version: 1.0.0
license: MIT
description: MDA cognitive memory system — injects associative context into every prompt
requirements:
"""

from pydantic import BaseModel
from typing import Optional
import sys
import os


class Filter:
    class Valves(BaseModel):
        MDA_PATH: str = "C:/Users/kha12/mda-lib"
        DEPTH: int = 6
        TOP_K: int = 5
        MIN_SCORE: float = 0.15
        ENABLED: bool = True

    def __init__(self):
        self.valves = self.Valves()
        self.engine = None
        self._learn_count = 0
        self._init_engine()

    def _init_engine(self):
        try:
            os.environ.setdefault("MDA_PATH", self.valves.MDA_PATH)
            os.chdir(self.valves.MDA_PATH)
            if self.valves.MDA_PATH not in sys.path:
                sys.path.insert(0, self.valves.MDA_PATH)
            from mda.integrations.engine import MDABatchEngine
            self.engine = MDABatchEngine(
                depth=self.valves.DEPTH,
                top_k_branches=self.valves.TOP_K,
            )
        except Exception as e:
            import traceback
            print(f"[MDA] init error: {e}")
            print(traceback.format_exc())
            self.engine = None

    async def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        if not self.valves.ENABLED or self.engine is None:
            return body

        messages = body.get("messages", [])
        user_messages = [m for m in messages if m["role"] == "user"]
        if not user_messages:
            return body

        query = user_messages[-1]["content"]
        if not isinstance(query, str):
            return body

        try:
            contexts = self.engine.build_context_batch([query])
            context = contexts[0] if contexts else ""
        except Exception as e:
            print(f"[MDA] inlet error: {e}")
            return body

        if not context.strip():
            return body

        print(f"[MDA] injecting {len(context.splitlines())} lines for: {query[:50]}")

        existing = [m for m in messages if m["role"] == "system"]
        if existing:
            existing[0]["content"] = context + "\n\n" + existing[0]["content"]
        else:
            body["messages"] = [{"role": "system", "content": context}] + messages

        return body

    async def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        if not self.valves.ENABLED or self.engine is None:
            return body

        messages = body.get("messages", [])
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        if not assistant_msgs:
            return body

        last = assistant_msgs[-1].get("content", "")
        if not isinstance(last, str) or len(last.strip()) < 30:
            return body

        try:
            should = (
                self.engine._should_learn(last)
                if hasattr(self.engine, "_should_learn")
                else True
            )
            if should:
                self.engine.mda.learn(last)
                print("[MDA] learned from assistant response")

                self._learn_count += 1
                if self._learn_count % 5 == 0:
                    try:
                        self.engine.save()
                    except Exception as e:
                        print(f"[MDA] save error: {e}")
        except Exception as e:
            print(f"[MDA] outlet error: {e}")

        return body
