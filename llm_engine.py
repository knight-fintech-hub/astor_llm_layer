"""
llm_engine.py — LLM Interface for Generation + Transition Evaluation
======================================================================
Core functions:
  1. generate_response() — Generate agent response for current node
  2. evaluate_transition() — Check if a transition condition is met
  3. flex_mode_evaluate() — Flex Mode: LLM picks best node when no edge matches
  4. detect_tool_intent() — Subagent: LLM decides which tool to call

Uses local Qwen model via transformers (same as other projects).
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import re
import threading
import time as _time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LogitsProcessor, LogitsProcessorList, TextIteratorStreamer


from config import cfg
from logger import get_logger

logger = get_logger(__name__)


# ── TTFT Timer ───────────────────────────────────────────────────────────────

class _TTFTTimer(LogitsProcessor):
    """
    Hooks into model.generate() as a LogitsProcessor to record wall-clock
    time-to-first-token (TTFT) without requiring a separate forward pass.
    The __call__ fires once per generated token; we capture only the first.
    """
    def __init__(self):
        self._t0 = _time.perf_counter()
        self.ttft_ms: Optional[float] = None

    def __call__(self, input_ids, scores):
        if self.ttft_ms is None:
            self.ttft_ms = (_time.perf_counter() - self._t0) * 1000
        return scores


# ── System Prompt KV Cache ────────────────────────────────────────────────────

class SystemPromptCache:
    """
    Bounded LRU cache that stores prefill-phase KV states for static system
    prompts. Keyed by a 16-char SHA-256 hex digest of the prompt text.

    Why this helps:
      Every call to model.generate() must process all input tokens through
      every transformer layer (prefill). When the system prompt is the same
      across turns (same agent, same node type), we pay the prefill cost for
      those tokens over and over. By caching the resulting past_key_values
      tensor tuple after a single forward() pass, subsequent generate() calls
      start with the KV state already populated and only process the shorter
      user+history suffix — cutting prefill time proportionally.

    Thread safety:
      Protected by a dedicated Lock; GPU tensors are kept on-device and NOT
      copied (avoid memory doubling). Callers must NOT mutate the returned
      past_key_values in place — generate() consumes but does not modify them.
    """

    def __init__(self, max_entries: int = 4):
        self.max_entries = max_entries
        self._cache: OrderedDict[str, Tuple] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _hash(text: str) -> str:
        """16-char hex digest of the system prompt string."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def get(self, prompt_text: str) -> Optional[Tuple]:
        """Return cached past_key_values or None on miss. Bumps LRU order on hit."""
        key = self._hash(prompt_text)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)  # mark as recently used
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, prompt_text: str, past_key_values: Tuple) -> None:
        """Store past_key_values; evicts LRU entry when at capacity."""
        key = self._hash(prompt_text)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self.max_entries:
                    evicted_key, _ = self._cache.popitem(last=False)  # pop LRU
                    logger.info(f"[KVCache] Evicted LRU entry key={evicted_key}")
            self._cache[key] = past_key_values

    def invalidate(self, prompt_text: str) -> None:
        """Remove a specific entry (e.g. after agent config update)."""
        key = self._hash(prompt_text)
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        """Flush the entire cache and free GPU memory."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_entries": self.max_entries,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 3) if total else 0.0,
            }


# ── Local LLM ────────────────────────────────────────────────────────────────

class LLMEngine:
    """Loads Qwen model for both response generation and transition evaluation."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._loaded = False
        self._lock = threading.Lock()
        # Serializes actual model.generate()/model() calls. generate_response()
        # and friends are dispatched onto worker threads via asyncio.to_thread so
        # the event loop stays free for concurrent async tool I/O; this lock keeps
        # those worker threads from calling into the same GPU model at once.
        self._generate_lock = threading.Lock()
        # Prefill-phase KV cache — only materialised on CUDA
        self._kv_cache: Optional[SystemPromptCache] = None

    def load(self):
        """Load model and tokenizer."""
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return

            logger.info(f"[LLM] Loading from: {cfg.MODEL_PATH}")
            logger.info(f"[LLM] Device: {self.device}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                cfg.MODEL_PATH, trust_remote_code=True
            )

            if self.device == "cuda":
                print("")
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    # load_in_16bit = True
                    
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    cfg.MODEL_PATH,
                    quantization_config=quant_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    cfg.MODEL_PATH,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
                self.model.to(self.device)

            self.model.eval()
            self._loaded = True

            # Initialise the system-prompt KV cache (CUDA only, if enabled)
            if self.device == "cuda" and cfg.KV_CACHE_ENABLED:
                self._kv_cache = SystemPromptCache(
                    max_entries=cfg.KV_CACHE_MAX_ENTRIES
                )
                logger.info(
                    f"[KVCache] System-prompt KV cache enabled | "
                    f"max_entries={cfg.KV_CACHE_MAX_ENTRIES}"
                )
            else:
                reason = "CPU device" if self.device != "cuda" else "disabled via config"
                logger.info(f"[KVCache] System-prompt KV cache NOT active ({reason})")

            logger.info("[LLM] Model ready")

    def _generate(self, messages: List[Dict[str, str]], max_tokens: int = None, greedy: bool = False) -> str:
        """Core generation from messages (used by all utility methods).

        greedy=True → deterministic decoding (no sampling). Used for routing /
        tool-selection / extraction decisions where the same input must always
        produce the same output. Response generation for the customer keeps
        greedy=False so replies stay natural and varied.
        """
        if not self._loaded:
            self.load()

        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=10000,
        ).to(self.device)

        ttft_timer = _TTFTTimer()
        with self._generate_lock, torch.no_grad():
            if greedy:
                # Deterministic decoding — identical input always routes the same way.
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens or cfg.MAX_NEW_TOKENS,
                    do_sample=False,
                    num_beams=1,
                    repetition_penalty=1.1,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    logits_processor=LogitsProcessorList([ttft_timer]),
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens or cfg.MAX_NEW_TOKENS,
                    temperature=cfg.TEMPERATURE,
                    top_p=cfg.TOP_P,
                    top_k=cfg.TOP_K,
                    do_sample=True,
                    repetition_penalty=1.1,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    logits_processor=LogitsProcessorList([ttft_timer]),
                )
        if ttft_timer.ttft_ms is not None:
            logger.info(f"[LLM] TTFT={ttft_timer.ttft_ms:.0f}ms (no-cache path)")

        input_length = inputs["input_ids"].shape[-1]
        generated_ids = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response.strip()
    
    def _generate_stream(self, messages: List[Dict[str, str]], max_tokens: int = None):
        """
        Streaming version of _generate(). Yields decoded text chunks token by
        token as the model produces them.

        Uses TextIteratorStreamer so model.generate() runs in a background
        thread while the main thread iterates the streamer for tokens.
        """
        if not self._loaded:
            self.load()

        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=10000,
        ).to(self.device)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens or cfg.MAX_NEW_TOKENS,
            temperature=cfg.TEMPERATURE,
            top_p=cfg.TOP_P,
            top_k=cfg.TOP_K,
            do_sample=True,
            repetition_penalty=1.1,
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )

        # model.generate() blocks until done — run it in a thread so we can
        # iterate the streamer concurrently on the calling thread. Acquire
        # _generate_lock inside the thread target so this call is serialized
        # against every other model.generate()/model() call, same as the
        # non-streaming paths — without it, a concurrent locked call (e.g.
        # evaluate_transition/detect_tool_intent from another session) can
        # touch the same GPU/quantized model state at once and corrupt output.
        def _run_generate():
            with self._generate_lock, torch.no_grad():
                self.model.generate(**gen_kwargs)

        thread = threading.Thread(target=_run_generate, daemon=True)
        thread.start()

        for token_text in streamer:
            yield token_text

        thread.join()

    # ── Prefill-phase KV cache helpers ────────────────────────────────────────

    def _get_or_create_system_kv_cache(
        self, system_prompt_text: str
    ) -> Optional[Tuple]:
        """
        Return cached past_key_values for `system_prompt_text`, computing and
        storing them on a cache miss.

        Strategy
        --------
        On a MISS we:
          1. Tokenize a messages list that contains ONLY the system message,
             rendered through apply_chat_template (without the generation prompt)
             so the token sequence exactly matches what would precede the first
             user turn in a real call.
          2. Run a single model.forward() in no-grad mode to obtain
             past_key_values.  No tokens are generated — this is a pure prefill.
          3. Store the KV states in the LRU cache.

        On a HIT a deep copy of the stored tensors is returned (still on GPU).
        A copy is mandatory: model.generate() extends past_key_values in place
        (it's a mutable Cache object), so handing out the stored reference
        would let generation silently bake each turn's tokens into the cached
        "system-prompt-only" entry. Left unfixed, every later turn (and every
        other session sharing the same resolved system prompt) reuses that
        corrupted, ever-growing cache and produces garbled output.

        Returns None if the KV cache is not active (CPU, disabled, or error).
        """
        if self._kv_cache is None:
            return None

        # ── Cache hit ────────────────────────────────────────────────────────
        cached = self._kv_cache.get(system_prompt_text)
        if cached is not None:
            stats = self._kv_cache.stats
            logger.info(
                f"[KVCache] HIT | size={stats['size']} "
                f"hit_rate={stats['hit_rate']:.1%}"
            )
            return copy.deepcopy(cached)

        # ── Cache miss — run prefill forward pass ────────────────────────────
        logger.info("[KVCache] MISS — running prefill forward pass for system prompt")
        try:
            # Build the system-only message list and render to text.
            # We do NOT add the generation prompt here; the system prompt sits
            # at position 0 and we want its KV states exactly as they would
            # appear at the beginning of any real conversation.
            sys_messages = [{"role": "system", "content": system_prompt_text}]
            sys_text = self.tokenizer.apply_chat_template(
                sys_messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )

            sys_inputs = self.tokenizer(
                sys_text,
                return_tensors="pt",
                truncation=True,
                max_length=8192,
            ).to(self.device)

            with self._generate_lock, torch.no_grad():
                fwd_out = self.model(
                    **sys_inputs,
                    use_cache=True,
                    return_dict=True,
                )

            past_kv = fwd_out.past_key_values
            self._kv_cache.put(system_prompt_text, past_kv)

            stats = self._kv_cache.stats
            sys_token_len = sys_inputs["input_ids"].shape[-1]
            logger.info(
                f"[KVCache] Stored new entry | "
                f"sys_tokens={sys_token_len} "
                f"cache_size={stats['size']}/{stats['max_entries']} "
                f"hit_rate={stats['hit_rate']:.1%}"
            )
            # Return a deep copy — the stored entry must stay pristine (see the
            # HIT branch above); the copy is what the caller's generate() mutates.
            return copy.deepcopy(past_kv)

        except Exception as exc:
            # Cache failure must never break generation — fall back gracefully.
            logger.warning(f"[KVCache] Prefill forward pass failed, skipping cache: {exc}")
            return None

    def _generate_with_kv_cache(
        self,
        system_prompt_text: str,
        remaining_messages: List[Dict[str, str]],
        max_tokens: int,
    ) -> str:
        """
        Generate a response using cached system-prompt KV states.

        1. Retrieve (or create) past_key_values for `system_prompt_text`.
        2. Tokenize `remaining_messages` (history + user turn, NO system prompt).
        3. Call model.generate() with past_key_values= pre-filled from step 1
           and attention_mask covering the full sequence (prefix + new tokens).
        4. Decode and return only the newly generated tokens.

        Falls back to `_generate()` on any failure so the call always succeeds.

        NOTE: past_key_values (DynamicCache) must NOT be passed to two separate
        generate() calls — it is mutable and gets extended in-place. TTFT is
        measured via a LogitsProcessor hook on the single real generate() call.
        """
        past_kv = self._get_or_create_system_kv_cache(system_prompt_text)
        if past_kv is None:
            # Cache not available — fall back to standard generation
            full_messages = [
                {"role": "system", "content": system_prompt_text}
            ] + remaining_messages
            return self._generate(full_messages, max_tokens)

        try:
            # Render only the non-system portion: history + current user turn.
            # We include the generation prompt here so the model knows to start
            # producing an assistant reply.
            suffix_text = self.tokenizer.apply_chat_template(
                remaining_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            suffix_inputs = self.tokenizer(
                suffix_text,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            ).to(self.device)

            # The model needs an attention_mask that covers the cached prefix
            # tokens too, otherwise it will ignore them during generation.
            if hasattr(past_kv, "get_seq_length"):
                prefix_len = past_kv.get_seq_length()  # transformers Cache API
            else:
                prefix_len = past_kv[0][0].shape[-2]  # legacy tuple format
            suffix_len = suffix_inputs["input_ids"].shape[-1]
            full_attention_mask = torch.ones(
                1, prefix_len + suffix_len,
                dtype=torch.long,
                device=self.device,
            )

            ttft_timer = _TTFTTimer()
            with self._generate_lock, torch.no_grad():
                outputs = self.model.generate(
                    input_ids=suffix_inputs["input_ids"],
                    attention_mask=full_attention_mask,
                    past_key_values=past_kv,
                    max_new_tokens=max_tokens,
                    temperature=cfg.TEMPERATURE,
                    top_p=cfg.TOP_P,
                    top_k=cfg.TOP_K,
                    do_sample=True,
                    repetition_penalty=1.1,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    logits_processor=LogitsProcessorList([ttft_timer]),
                )
            if ttft_timer.ttft_ms is not None:
                logger.info(f"[LLM] TTFT={ttft_timer.ttft_ms:.0f}ms (kv-cache path)")

            # outputs contains [suffix_input_ids | generated_ids]; strip prefix
            generated_ids = outputs[0][suffix_len:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            return response.strip()

        except Exception as exc:
            logger.warning(
                f"[KVCache] KV-cache-assisted generation failed ({exc}), "
                "retrying without cache"
            )
            # Invalidate potentially corrupt cache entry and retry
            if self._kv_cache:
                self._kv_cache.invalidate(system_prompt_text)
            full_messages = [
                {"role": "system", "content": system_prompt_text}
            ] + remaining_messages
            return self._generate(full_messages, max_tokens)
        
    def _generate_with_kv_cache_stream(
        self,
        system_prompt_text: str,
        remaining_messages: List[Dict[str, str]],
        max_tokens: int,
    ):
        """
        Streaming counterpart of _generate_with_kv_cache().
        Yields decoded text chunks token by token.
        Falls back to _generate_stream() if the KV cache is unavailable.
        """
        past_kv = self._get_or_create_system_kv_cache(system_prompt_text)
        if past_kv is None:
            full_messages = [
                {"role": "system", "content": system_prompt_text}
            ] + remaining_messages
            yield from self._generate_stream(full_messages, max_tokens)
            return

        try:
            suffix_text = self.tokenizer.apply_chat_template(
                remaining_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            suffix_inputs = self.tokenizer(
                suffix_text,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            ).to(self.device)

            if hasattr(past_kv, "get_seq_length"):
                prefix_len = past_kv.get_seq_length()
            else:
                prefix_len = past_kv[0][0].shape[-2]
            suffix_len = suffix_inputs["input_ids"].shape[-1]
            full_attention_mask = torch.ones(
                1, prefix_len + suffix_len,
                dtype=torch.long,
                device=self.device,
            )


            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )

            gen_kwargs = dict(
                input_ids=suffix_inputs["input_ids"],
                attention_mask=full_attention_mask,
                past_key_values=past_kv,
                max_new_tokens=max_tokens,
                temperature=cfg.TEMPERATURE,
                top_p=cfg.TOP_P,
                top_k=cfg.TOP_K,
                do_sample=True,
                repetition_penalty=1.1,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                streamer=streamer,
            )

            # See _generate_stream() for why _generate_lock is acquired here:
            # this call must be serialized against every other
            # model.generate()/model() call across threads/sessions.
            def _run_generate():
                with self._generate_lock, torch.no_grad():
                    self.model.generate(**gen_kwargs)

            thread = threading.Thread(target=_run_generate, daemon=True)
            thread.start()

            for token_text in streamer:
                yield token_text

            thread.join()

        except Exception as exc:
            logger.warning(
                f"[KVCache] KV-cache stream failed ({exc}), falling back to plain stream"
            )
            if self._kv_cache:
                self._kv_cache.invalidate(system_prompt_text)
            full_messages = [
                {"role": "system", "content": system_prompt_text}
            ] + remaining_messages
            yield from self._generate_stream(full_messages, max_tokens)

    def generate_response(
        self,
        system_prompt: str,
        history: List[Dict[str, str]],
        user_message: str,
        dynamic_context: str = "",
    ) -> str:
        """
        Generate agent response.

        Args:
            system_prompt: Full STATIC system prompt (agent + node context).
                           Must NOT contain per-turn data (session variables,
                           tool results, etc.) — the KV cache is keyed on this
                           string's hash, so any mutation causes a cache miss.
            history: Previous messages [{role, content}]
            user_message: Current user input
            dynamic_context: Per-turn data block (session variables + tool result).
                             Built by build_session_context() in orchestrator.
                             Injected as a context message BEFORE the user turn
                             so the system prompt stays stable across turns.

        When the system-prompt KV cache is active (CUDA + KV_CACHE_ENABLED),
        the system prompt tokens are processed only once and their key/value
        states are reused on every subsequent turn, reducing prefill latency.
        """
        if not self._loaded:
            self.load()

        max_tokens = cfg.MAX_NEW_TOKENS

        if self._kv_cache is not None:
            # Build the messages list that follows the system prompt
            # (history + current user turn). The system prompt itself is
            # handled separately by the KV cache prefill path.
            #
            # IMPORTANT: dynamic_context is injected here — NOT into system_prompt —
            # so the cache key (hash of system_prompt) stays stable every turn.
            remaining_messages = list(history)
            if dynamic_context:
                remaining_messages.append({
                    "role": "user",
                    "content": f"[Context]\n{dynamic_context}",
                })
                remaining_messages.append({
                    "role": "assistant",
                    "content": "Understood, I have the context.",
                })
            remaining_messages.append({"role": "user", "content": user_message})

            response = self._generate_with_kv_cache(
                system_prompt_text=system_prompt,
                remaining_messages=remaining_messages,
                max_tokens=max_tokens,
            )
        else:
            # KV cache not available — standard full-sequence generation
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(history)
            if dynamic_context:
                messages.append({
                    "role": "user",
                    "content": f"[Context]\n{dynamic_context}",
                })
                messages.append({
                    "role": "assistant",
                    "content": "Understood, I have the context.",
                })
            messages.append({"role": "user", "content": user_message})
            response = self._generate(messages)

        logger.info(f"[LLM] Generated response: {len(response)} chars")
        return response
    
    def _sentence_stream(self, raw_token_gen):
        """
        Pass-through generator that wraps a raw token generator.

        Behaviour
        ---------
        * Yields every token chunk unchanged to the caller — no buffering delay.
        * Simultaneously accumulates chunks into a local buffer.
        * Flushes (logs) the buffer as a completed sentence when:
            - A sentence-ending punctuation is encountered: . ! ?
            - A comma/semicolon is encountered AND the buffer already holds
              >= 6 words (avoids logging every short clause separately).
        * Any leftover text at generator exhaustion is flushed as a final sentence.

        This lets you confirm in the log that streaming is working and see the
        output in natural sentence chunks rather than raw token noise.
        """
        SENTENCE_END = frozenset(".!?")
        CLAUSE_BREAK = frozenset(",;")
        MIN_WORDS_FOR_CLAUSE_FLUSH = 6

        buffer: List[str] = []
        sentence_idx = 0

        for chunk in raw_token_gen:
            yield chunk  # pass token to caller immediately

            buffer.append(chunk)
            joined = "".join(buffer)

            # Determine whether to flush
            stripped = joined.rstrip()
            should_flush = False
            flush_reason = ""

            if stripped and stripped[-1] in SENTENCE_END:
                should_flush = True
                flush_reason = f"punct='{stripped[-1]}'"
            elif stripped and stripped[-1] in CLAUSE_BREAK:
                word_count = len(joined.split())
                if word_count >= MIN_WORDS_FOR_CLAUSE_FLUSH:
                    should_flush = True
                    flush_reason = f"clause-break='{stripped[-1]}' words={word_count}"

            if should_flush:
                sentence_idx += 1
                logger.info(
                    f"[Stream][sentence {sentence_idx}] ({flush_reason}) "
                    f"{joined.strip()!r}"
                )
                buffer.clear()

        # Flush any remaining text
        if buffer:
            sentence_idx += 1
            tail = "".join(buffer).strip()
            if tail:
                logger.info(
                    f"[Stream][sentence {sentence_idx}] (tail) {tail!r}"
                )

        logger.info(
            f"[Stream] Completed — {sentence_idx} sentence segment(s) logged."
        )

    # ─────────────────────────────────────────────────────────────────────────

    def generate_response_stream(
        self,
        system_prompt: str,
        history: List[Dict[str, str]],
        user_message: str,
        dynamic_context: str = "",
    ):
        """
        Streaming version of generate_response().
        Yields decoded text chunks (str) token by token as the model produces
        them. Callers can concatenate the chunks to reconstruct the full reply.

        Sentence-level log lines (DEBUG) are emitted via _sentence_stream()
        so you can confirm streaming is active without any buffering cost.

        Args: identical to generate_response().
        """
        if not self._loaded:
            self.load()

        max_tokens = cfg.MAX_NEW_TOKENS

        if self._kv_cache is not None:
            remaining_messages = list(history)
            if dynamic_context:
                remaining_messages.append({
                    "role": "user",
                    "content": f"[Context]\n{dynamic_context}",
                })
                remaining_messages.append({
                    "role": "assistant",
                    "content": "Understood, I have the context.",
                })
            remaining_messages.append({"role": "user", "content": user_message})

            raw = self._generate_with_kv_cache_stream(
                system_prompt_text=system_prompt,
                remaining_messages=remaining_messages,
                max_tokens=max_tokens,
            )
        else:
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(history)
            if dynamic_context:
                messages.append({
                    "role": "user",
                    "content": f"[Context]\n{dynamic_context}",
                })
                messages.append({
                    "role": "assistant",
                    "content": "Understood, I have the context.",
                })
            messages.append({"role": "user", "content": user_message})

            raw = self._generate_stream(messages, max_tokens)

        logger.info("[Stream] generate_response_stream() started — yielding tokens")
        yield from self._sentence_stream(raw)

    async def agenerate_response_stream(
        self,
        system_prompt: str,
        history: List[Dict[str, str]],
        user_message: str,
        dynamic_context: str = "",
    ):
        """
        Async-generator bridge for generate_response_stream().

        generate_response_stream() is a plain sync generator whose first
        next() call runs tokenization and, on a KV-cache MISS, a full prefill
        forward pass — a heavy call with no internal await point. Iterating
        it directly on the event loop thread (`for token in ...`) blocks the
        entire loop for that call's duration, starving every other coroutine,
        including async-dispatched background tool jobs (asyncio.Task), until
        it happens to return. Iterating it via asyncio.to_thread doesn't work
        either since to_thread expects a single blocking call, not a
        generator to be pulled from repeatedly.

        Instead, the whole sync generator is driven on a background thread;
        each item is handed back across a plain queue.Queue, and the only
        thing the calling coroutine ever awaits is pulling the next queue
        item via run_in_executor — a genuine suspension point on every
        iteration, so the loop stays free for the tool call's real duration,
        not just the brief gaps between already-started tokens.
        """
        import queue as _queue

        q: "_queue.Queue" = _queue.Queue()
        _SENTINEL = object()

        def _worker():
            try:
                for chunk in self.generate_response_stream(
                    system_prompt, history, user_message, dynamic_context
                ):
                    q.put(chunk)
            except Exception as exc:  # noqa: BLE001 — relayed to the caller below
                q.put(exc)
            finally:
                q.put(_SENTINEL)

        threading.Thread(target=_worker, daemon=True).start()

        loop = asyncio.get_running_loop()
        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is _SENTINEL:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    @property
    def cache_stats(self) -> Optional[Dict[str, Any]]:
        """Return KV cache hit/miss statistics, or None if cache is inactive."""
        return self._kv_cache.stats if self._kv_cache else None

    def evaluate_transition(
        self,
        user_message: str,
        condition: str,
        conversation_context: str = "",
    ) -> bool:
        """
        Evaluate if a transition condition is met based on user's message.
        Returns True if condition matches, False otherwise.
        """
        system_prompt = (
            "You are a transition evaluator. Your job is to determine if a customer's "
            "message satisfies a given condition. Respond with ONLY 'YES' or 'NO'. "
            "Nothing else. No explanation."
        )

        user_prompt = (
            f"Condition: \"{condition}\"\n"
            f"Customer said: \"{user_message}\"\n"
        )
        if conversation_context:
            user_prompt += f"Conversation context: {conversation_context}\n"

        user_prompt += "\nDoes the customer's message satisfy the condition? Answer YES or NO only."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self._generate(messages, max_tokens=5, greedy=True)
        result = response.strip().upper()

        is_match = "YES" in result
        logger.info(f"[LLM] Transition eval: condition='{condition[:50]}' | msg='{user_message[:30]}' | result={is_match}")
        return is_match

    def evaluate_all_transitions(
        self,
        user_message: str,
        transitions: List[Dict[str, str]],
        current_node_name: str,
        conversation_context: str = "",
    ) -> Optional[str]:
        """
        Pure LLM transition evaluation — single call, multi-choice.
        Evaluates ALL transitions at once and picks the best match.
        
        Returns to_node if a transition matches, None if should stay in current node.
        
        transitions: List of {"to_node": "...", "condition": "..."}
        """
        if not transitions:
            return None

        system_prompt = (
            "You are a conversation flow router. You decide which transition to take based on "
            "what the customer said. You MUST respond with ONLY a single number. Nothing else.\n\n"
            "Rules:\n"
            "- Pick the transition that BEST matches the customer's message.\n"
            "- If NO transition clearly matches, respond with 0 (stay in current node).\n"
            "- Be STRICT — only transition when the customer's message CLEARLY satisfies a condition.\n"
            "- Giving a reason for not paying is NOT the same as refusing to ever pay.\n"
            "- Saying 'I have no money' means they can't pay NOW, not that they refuse forever.\n"
            "- Only pick a transition if you are VERY confident.\n"
            "- Respond with ONLY the number. No explanation. No text."
        )

        # Build numbered options
        options = []
        for i, t in enumerate(transitions, 1):
            options.append(f"{i}. → {t.get('to_node', '?')}: {t['condition']}")

        options.append(f"0. Stay in current node ({current_node_name}) — none of the above clearly match")

        user_prompt = (
            f"Current node: {current_node_name}\n"
            f"Customer said: \"{user_message}\"\n"
        )
        if conversation_context:
            user_prompt += f"Recent conversation: {conversation_context}\n"

        user_prompt += f"\nTransitions:\n" + "\n".join(options)
        user_prompt += "\n\nWhich number? Reply with ONLY the number:"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self._generate(messages, max_tokens=5, greedy=True)
        result = response.strip()

        # Extract number from response
        # Try to find first digit
        chosen = None
        for char in result:
            if char.isdigit():
                chosen = int(char)
                break

        if chosen is None or chosen == 0:
            logger.info(f"[LLM] Transition: STAY (response='{result}') | msg='{user_message[:30]}'")
            return None

        if 1 <= chosen <= len(transitions):
            target = transitions[chosen - 1]["to_node"]
            logger.info(f"[LLM] Transition: #{chosen} → {target} | msg='{user_message[:30]}'")
            return target

        logger.info(f"[LLM] Transition: invalid number={chosen}, STAY | msg='{user_message[:30]}'")
        return None

    def extract_variables(
        self,
        user_message: str,
        variable_names: List[str],
        conversation_context: str = "",
    ) -> Dict[str, str]:
        """
        Extract structured variables from user's free-text message.
        E.g., extract DOB, PTPDate, PTPAmount from "I'll pay 5000 on next Tuesday".
        Returns dict of {variable_name: extracted_value}.
        """
        if not variable_names:
            return {}

        system_prompt = (
            "You are a data extractor. Extract the requested fields from the customer's message. "
            "Respond in EXACTLY this format — one field per line:\n"
            "FIELD_NAME=value\n\n"
            "Rules:\n"
            "- If a field cannot be extracted, write: FIELD_NAME=UNKNOWN\n"
            "- For dates, use format: YYYY-MM-DD or the exact text the customer said\n"
            "- For amounts, use numbers only (no currency symbols)\n"
            "- No explanation. Only FIELD_NAME=value lines."
        )

        user_prompt = (
            f"Customer said: \"{user_message}\"\n"
        )
        if conversation_context:
            user_prompt += f"Context: {conversation_context}\n"

        user_prompt += f"\nExtract these fields:\n"
        for v in variable_names:
            user_prompt += f"- {v}\n"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self._generate(messages, max_tokens=50, greedy=True)

        # Parse response
        extracted = {}
        for line in response.strip().split("\n"):
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key in variable_names and value and value.upper() != "UNKNOWN":
                    extracted[key] = value

        logger.info(f"[LLM] Extracted variables: {extracted} from msg='{user_message[:30]}'")
        return extracted

    def decide_transition_and_tool(
        self,
        user_message: str,
        transitions: List[Dict[str, str]],
        current_node_name: str,
        available_tools: List[Dict[str, str]],
        conversation_context: str = "",
    ) -> Dict[str, Optional[str]]:
        """
        Combined single-call decision: which transition (if any) applies AND
        which tool (if any) should be called this turn. Replaces two separate
        LLM calls (evaluate_all_transitions + detect_tool_intent) with one,
        to cut per-turn LLM calls in flow mode when both are needed.

        Returns {"transition": to_node_id_or_None, "tool_id": tool_id_or_None}.
        """
        system_prompt = (
            "You are a conversation flow router for a phone call. In ONE response, decide:\n"
            "1) Which transition (if any) the customer's message satisfies.\n"
            "2) Which tool (if any) should be called right now.\n"
            "Respond in EXACTLY this format, nothing else, no explanation:\n"
            "TRANSITION:<number>|TOOL:<tool_id or NONE>\n\n"
            "Transition rules: pick the transition that BEST matches the customer's message. "
            "If none clearly matches, use 0. Be STRICT — only pick a transition if you are VERY confident.\n"
            "Tool rules: pick a tool ONLY if it's clearly needed right now based on what the "
            "customer just said. Otherwise use NONE."
        )

        options = [f"{i}. → {t.get('to_node', '?')}: {t['condition']}" for i, t in enumerate(transitions, 1)]
        options.append(f"0. Stay in current node ({current_node_name}) — none of the above clearly match")

        tools_desc = "\n".join(
            f"- {t['tool_id']}: {t['name']} — {t.get('description', '')}"
            for t in available_tools
        ) or "(no tools available)"

        user_prompt = (
            f"Current node: {current_node_name}\n"
            f"Customer said: \"{user_message}\"\n"
        )
        if conversation_context:
            user_prompt += f"Recent conversation: {conversation_context}\n"
        user_prompt += "\nTransitions:\n" + "\n".join(options)
        user_prompt += f"\n\nAvailable tools:\n{tools_desc}"
        user_prompt += "\n\nRespond as: TRANSITION:<number>|TOOL:<tool_id or NONE>"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self._generate(messages, max_tokens=20, greedy=True)
        result = response.strip()

        transition_target = None
        tool_id = None

        m = re.search(r"TRANSITION:\s*(\d+)", result)
        if m:
            chosen = int(m.group(1))
            if 1 <= chosen <= len(transitions):
                transition_target = transitions[chosen - 1]["to_node"]

        m = re.search(r"TOOL:\s*([\w\-]+)", result)
        if m:
            candidate = m.group(1).strip()
            valid_ids = {t["tool_id"] for t in available_tools}
            if candidate in valid_ids:
                tool_id = candidate

        logger.info(f"[LLM] Combined decision: transition={transition_target} tool={tool_id} | raw='{result}'")
        return {"transition": transition_target, "tool_id": tool_id}

    def flex_mode_evaluate(
        self,
        user_message: str,
        available_nodes: List[Dict[str, str]],
        conversation_context: str = "",
    ) -> Optional[str]:
        """
        Flex Mode: When no edge matches, LLM picks the best node from ALL agent nodes.
        Returns node_id if a strong match is found, None otherwise.

        available_nodes: List of {"node_id": "...", "name": "...", "prompt": "..."}
        """
        if not available_nodes:
            return None

        system_prompt = (
            "You are a conversation router. Given a customer's message and a list of conversation nodes, "
            "determine which node is MOST relevant. If the message clearly fits a specific node, "
            "respond with ONLY the node_id. If no node is clearly relevant, respond with 'NONE'. "
            "No explanation."
        )

        nodes_desc = "\n".join(
            f"- {n['node_id']}: {n['name']} — {n['prompt'][:80]}"
            for n in available_nodes
        )

        user_prompt = (
            f"Customer said: \"{user_message}\"\n\n"
            f"Available nodes:\n{nodes_desc}\n"
        )
        if conversation_context:
            user_prompt += f"\nConversation context: {conversation_context}\n"

        user_prompt += "\nWhich node_id best matches? Respond with node_id or NONE."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self._generate(messages, max_tokens=20, greedy=True)
        result = response.strip()

        # Check if response is a valid node_id
        valid_ids = {n["node_id"] for n in available_nodes}
        if result in valid_ids:
            logger.info(f"[LLM] Flex mode: jumped to '{result}' for msg='{user_message[:30]}'")
            return result

        # Try to extract node_id from response
        for nid in valid_ids:
            if nid in result:
                logger.info(f"[LLM] Flex mode: extracted '{nid}' from response")
                return nid

        logger.info(f"[LLM] Flex mode: no match (response='{result[:30]}')")
        return None

    def detect_tool_intent(
        self,
        user_message: str,
        available_tools: List[Dict[str, str]],
        conversation_context: str = "",
    ) -> Optional[str]:
        """
        Subagent Node: LLM decides if/which tool to call based on conversation.
        Returns tool_id if a tool should be called, None otherwise.

        available_tools: List of {"tool_id": "...", "name": "...", "description": "..."}
        """
        if not available_tools:
            return None

        system_prompt = (
            "You are a tool selector. Given a customer's message and available tools, "
            "determine if any tool should be called NOW to help the conversation. "
            "If yes, respond with ONLY the tool_id. If no tool is needed, respond with 'NONE'. "
            "No explanation."
        )

        tools_desc = "\n".join(
            f"- {t['tool_id']}: {t['name']} — {t.get('description', '')}"
            for t in available_tools
        )

        user_prompt = (
            f"Customer said: \"{user_message}\"\n\n"
            f"Available tools:\n{tools_desc}\n"
        )
        if conversation_context:
            user_prompt += f"\nConversation context: {conversation_context}\n"

        user_prompt += "\nShould any tool be called? Respond with tool_id or NONE."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self._generate(messages, max_tokens=20, greedy=True)
        result = response.strip()

        valid_ids = {t["tool_id"] for t in available_tools}
        if result in valid_ids:
            logger.info(f"[LLM] Subagent: tool '{result}' selected for msg='{user_message[:30]}'")
            return result

        for tid in valid_ids:
            if tid in result:
                logger.info(f"[LLM] Subagent: extracted tool '{tid}'")
                return tid

        return None

    def select_image_tool(
        self,
        user_message: str,
        available_tools: List[Dict[str, str]],
        conversation_context: str = "",
        node_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Force LLM to select the most appropriate image processing tool for an uploaded document
        using a robust Chain of Thought approach.
        """
        if not available_tools:
            return None

        system_prompt = (
            "The customer just uploaded a document. You must select the MOST appropriate tool to process it.\n\n"
            "Follow these steps to decide:\n"
            "1. Analyze the 'Recent conversation' to determine what document the agent most recently asked the customer to upload.\n"
            "2. Look at the 'Available tools' to find the tool that processes that specific document type.\n"
            "3. Use the file extension as an additional hint (.pdf = usually bank statements, .jpg/.png = usually ID cards like PAN or Aadhaar).\n\n"
            "You MUST format your response EXACTLY as follows (do not omit the TOOL line):\n"
            "REASONING: <brief 1-sentence explanation of what was asked for and why you chose the tool>\n"
            "TOOL: <tool_id>"
        )

        tools_desc = "\n".join(
            f"- {t['tool_id']}: {t['name']} — {t.get('description', '')}"
            for t in available_tools
        )

        user_prompt = f"Available tools:\n{tools_desc}\n\n"
        
        if node_name and node_name != "__single_prompt__":
            user_prompt += f"Current conversation node: {node_name}\n"
            
        if conversation_context:
            user_prompt += f"Recent conversation: {conversation_context}\n"
            
        user_prompt += f"\nCustomer said: \"{user_message}\"\n"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Use 100 max tokens to allow for reasoning
        response = self._generate(messages, max_tokens=100, greedy=True)
        result = response.strip()
        logger.info(f"[LLM] select_image_tool raw response:\n{result}")

        # Parse the tool_id from the TOOL: line
        import re
        match = re.search(r"TOOL:\s*(.+)", result)
        if match:
            candidate = match.group(1).strip()
            valid_ids = {t["tool_id"] for t in available_tools}
            if candidate in valid_ids:
                logger.info(f"[LLM] Image tool '{candidate}' selected")
                return candidate
            
            # Fallback: check if the name is mentioned in the candidate string or vice-versa
            for t in available_tools:
                tname = t["name"].lower()
                cand = candidate.lower()
                if tname in cand or cand in tname:
                    logger.info(f"[LLM] Image tool matched by name: '{t['tool_id']}'")
                    return t["tool_id"]

        # Fallback if the LLM didn't format correctly
        for t in available_tools:
            if t["tool_id"] in result or t["name"].lower() in result.lower():
                logger.info(f"[LLM] Image tool extracted via fallback: '{t['tool_id']}'")
                return t["tool_id"]

        return None

    def extract_variables(
        self,
        user_message: str,
        variable_names: List[str],
        conversation_context: str = "",
    ) -> Dict[str, str]:
        """
        Extract structured field values (e.g. DOB, PTPDate, PTPAmount) from the
        customer's message. Returns only fields the model is confident about;
        fields not mentioned in the message are omitted (not guessed).
        """
        if not variable_names:
            return {}

        system_prompt = (
            "You extract structured field values from a customer's message on a phone call. "
            "Respond with ONLY a compact JSON object mapping field name to extracted value. "
            "Only include a field if the customer's message clearly provides it. "
            "Omit fields that were not mentioned — do not guess or invent values. "
            "If nothing can be extracted, respond with {}. No explanation, no markdown, JSON only."
        )

        user_prompt = (
            f"Fields to extract: {', '.join(variable_names)}\n"
            f"Customer said: \"{user_message}\"\n"
        )
        if conversation_context:
            user_prompt += f"Conversation context: {conversation_context}\n"
        user_prompt += "\nJSON:"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self._generate(messages, max_tokens=80, greedy=True)

        import json as _json
        import re as _re

        match = _re.search(r"\{.*\}", response, _re.DOTALL)
        if not match:
            return {}

        try:
            data = _json.loads(match.group(0))
        except (ValueError, _json.JSONDecodeError):
            return {}

        if not isinstance(data, dict):
            return {}

        result = {
            k: str(v) for k, v in data.items()
            if k in variable_names and v not in (None, "", "null")
        }

        if result:
            logger.info(f"[LLM] Extracted variables: {list(result.keys())}")
        return result

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# ── Singleton ────────────────────────────────────────────────────────────────

llm_engine = LLMEngine()


# ── LLM Request Queue ─────────────────────────────────────────────────────────

class LLMRequestQueue:
    """
    Async queue that serialises concurrent LLM calls across multiple users.

    Why this is needed:
      FastAPI handles many simultaneous HTTP requests (each user's message
      arrives as a separate coroutine). Each call eventually needs to run
      model.generate() which is CPU/GPU-bound and NOT concurrency-safe.
      Without a queue:
        - All N user threads block on _generate_lock simultaneously.
        - Memory spikes (N sleeping threads), no timeout, no clean error.
      With this queue:
        - At most `max_concurrent` calls run on the GPU at once (default=1).
        - Up to `max_queue_size` additional calls wait in line (FIFO).
        - Any call waiting longer than `timeout_sec` receives a 503 error
          immediately rather than blocking the user indefinitely.

    Thread safety:
      asyncio.Semaphore is safe within a single event loop. _generate_lock
      in LLMEngine still guards the actual GPU call — this queue sits in
      front of it as the traffic manager.
    """

    def __init__(self, max_concurrent: int = 1, max_queue_size: int = 50, timeout_sec: int = 60):
        self._max_concurrent = max_concurrent
        self._max_queue_size = max_queue_size
        self._timeout_sec = timeout_sec
        # Semaphore created lazily on first use (must be created inside
        # a running event loop to avoid "no current event loop" errors at
        # import time when running under uvicorn).
        self._semaphore: Optional["asyncio.Semaphore"] = None
        self._queued: int = 0       # requests waiting for semaphore
        self._in_flight: int = 0    # requests currently generating
        self._total_served: int = 0
        self._total_rejected: int = 0
        self._lock = threading.Lock()

    def _get_semaphore(self) -> "asyncio.Semaphore":
        """Lazy-init semaphore inside the running event loop."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrent)
        return self._semaphore

    @__import__('contextlib').asynccontextmanager
    async def acquire(self):
        """
        Context manager for acquiring a queue slot (used by streaming endpoints).
        Yields once the semaphore is acquired, releases it on exit.
        Raises HTTPException(503) if the queue is full or wait times out.
        """
        from fastapi import HTTPException

        with self._lock:
            if self._queued >= self._max_queue_size:
                self._total_rejected += 1
                logger.warning(f"[LLMQueue] Queue full ({self._queued} waiting) — rejecting request")
                raise HTTPException(status_code=503, detail="The LLM is currently at capacity. Please retry in a few seconds.")
            self._queued += 1

        sem = self._get_semaphore()
        try:
            try:
                await asyncio.wait_for(sem.acquire(), timeout=self._timeout_sec)
            except asyncio.TimeoutError:
                with self._lock:
                    self._queued -= 1
                    self._total_rejected += 1
                logger.warning(f"[LLMQueue] Request timed out after {self._timeout_sec}s waiting")
                raise HTTPException(status_code=503, detail=f"LLM did not respond within {self._timeout_sec}s. Please retry.")

            with self._lock:
                self._queued -= 1
                self._in_flight += 1

            try:
                yield
                with self._lock:
                    self._total_served += 1
            finally:
                with self._lock:
                    self._in_flight -= 1
                sem.release()

        except HTTPException:
            raise

    async def run(self, fn, *args, **kwargs):
        """
        Enqueue an LLM call and run it on a worker thread.
        """
        async with self.acquire():
            return await asyncio.to_thread(fn, *args, **kwargs)

    @property
    def stats(self) -> Dict[str, Any]:
        """Return current queue metrics (used by /health endpoint)."""
        with self._lock:
            return {
                "max_concurrent": self._max_concurrent,
                "max_queue_size": self._max_queue_size,
                "timeout_sec": self._timeout_sec,
                "queued": self._queued,
                "in_flight": self._in_flight,
                "total_served": self._total_served,
                "total_rejected": self._total_rejected,
            }


# ── LLM Queue Singleton ───────────────────────────────────────────────────────

llm_queue = LLMRequestQueue(
    max_concurrent=cfg.MAX_CONCURRENT_LLM,
    max_queue_size=cfg.LLM_QUEUE_SIZE,
    timeout_sec=cfg.LLM_REQUEST_TIMEOUT_SEC,
)
