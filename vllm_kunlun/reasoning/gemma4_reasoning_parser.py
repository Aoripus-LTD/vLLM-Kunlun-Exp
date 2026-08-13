# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest as _ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike


# ---------------------------------------------------------------------------
# Belt-and-suspenders: force ``skip_special_tokens=False`` for any
# ChatCompletionRequest that hasn't explicitly opted in. This guarantees
# the ``<|channel>`` / ``<channel|>`` boundary tokens survive detokenization
# regardless of which serving path (chat / responses / batch) builds
# SamplingParams and regardless of whether ``adjust_request`` is routed
# through. The parser also implements :meth:`adjust_request` below as the
# canonical hook; the monkey-patch only fires if upstream skipped the hook.
# ---------------------------------------------------------------------------
_PATCH_SENTINEL = "_gemma4_skip_special_tokens_patched"
if not getattr(_ChatCompletionRequest, _PATCH_SENTINEL, False):
    _original_to_sampling_params = _ChatCompletionRequest.to_sampling_params

    def _patched_to_sampling_params(self, *args, **kwargs):
        if self.skip_special_tokens is True:
            fields_set = getattr(self, "model_fields_set", set())
            if "skip_special_tokens" not in fields_set:
                object.__setattr__(self, "skip_special_tokens", False)
        return _original_to_sampling_params(self, *args, **kwargs)

    _ChatCompletionRequest.to_sampling_params = _patched_to_sampling_params
    setattr(_ChatCompletionRequest, _PATCH_SENTINEL, True)


# Role label that Gemma4 emits at the start of the thinking channel.
# Nominally ``thought\n``, but under sampling the model may produce merged
# tokens like ``thoughten-US``, ``thoughte``, etc., so we strip any
# ``thought<suffix>\n`` prefix with a tolerant regex.
_THOUGHT_PREFIX_RE = re.compile(r"^thought[^\n]*\n?")


class Gemma4ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for Google Gemma4 thinking models.

    Gemma4 uses ``<|channel>...<channel|>`` special tokens (ids 100 / 101)
    to delimit reasoning/thinking content within its output. The channel
    structure looks like::

        <|channel>thought
        ...chain of thought reasoning...
        <channel|>
        Final answer text here.

    vLLM defaults ``skip_special_tokens=True``, which would strip the
    channel boundary tokens from decoded text and make reliable splitting
    impossible. This parser overrides :meth:`adjust_request` to force
    ``skip_special_tokens=False`` when the client did not explicitly opt in,
    mirroring the upstream fix in vLLM PR #39027.
    """

    @property
    def start_token(self) -> str:
        return "<|channel>"

    @property
    def end_token(self) -> str:
        return "<channel|>"

    def __init__(self, tokenizer: "TokenizerLike", *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        # Additional boundary tokens used by Gemma4 turn / tool structure.
        self.new_turn_token_id = self.vocab.get("<|turn>")
        self.tool_call_token_id = self.vocab.get("<|tool_call>")
        self.tool_response_token_id = self.vocab.get("<|tool_response>")

    # ------------------------------------------------------------------
    # Request adjustment — the canonical hook for parser-driven sampling
    # tweaks. Called once per request by the serving layer before sampling
    # params are built.
    # ------------------------------------------------------------------

    def adjust_request(
        self, request: "ChatCompletionRequest"
    ) -> "ChatCompletionRequest":
        # Force skip_special_tokens=False unless the client explicitly set
        # it via request body. Without this the ``<|channel>`` /
        # ``<channel|>`` delimiters are dropped from decoded text and
        # reasoning/content splitting degenerates to fragile substring
        # matching on the ``thought*\n`` label.
        try:
            request = super().adjust_request(request)  # type: ignore[misc]
        except AttributeError:
            pass
        try:
            fields_set = getattr(request, "model_fields_set", set())
            if (
                request.skip_special_tokens is True
                and "skip_special_tokens" not in fields_set
            ):
                request.skip_special_tokens = False
        except Exception:
            # Non-fatal: base class will still get a request object.
            pass
        return request

    # ------------------------------------------------------------------
    # is_reasoning_end — used by structured engines (e.g. xgrammar)
    # ------------------------------------------------------------------

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        # Walk backwards looking for the last structural boundary.
        for i in range(len(input_ids) - 1, -1, -1):
            tid = input_ids[i]
            if tid == self.start_token_id:
                return False
            if tid == self.tool_call_token_id:
                return True
            if tid in (self.new_turn_token_id, self.tool_response_token_id):
                return False
            if tid == self.end_token_id:
                return True
        return False

    # ------------------------------------------------------------------
    # extract_content_ids
    # ------------------------------------------------------------------

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        # Locate the first matched <|channel> ... <channel|> pair.
        start_idx = -1
        for i, tid in enumerate(input_ids):
            if tid == self.start_token_id:
                start_idx = i
            elif tid == self.end_token_id and start_idx >= 0:
                return list(input_ids[i + 1 :])
        return []

    # ------------------------------------------------------------------
    # Non-streaming path
    # ------------------------------------------------------------------

    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        """Extract reasoning from the full model output.

        Primary path: with ``skip_special_tokens=False`` enforced by
        :meth:`adjust_request`, ``<|channel>`` / ``<channel|>`` are present
        verbatim in ``model_output`` and we split on them directly.

        Fallback path: if the delimiters were nevertheless stripped (e.g.,
        the client explicitly set ``skip_special_tokens=True``), we anchor
        on the tolerant ``thought*\n`` regex.
        """
        start_tok = self.start_token
        end_tok = self.end_token

        # Primary: delimiters preserved in text.
        if start_tok in model_output:
            prefix, _, after_start = model_output.partition(start_tok)
            if end_tok in after_start:
                reasoning_text, _, content_text = after_start.partition(end_tok)
                reasoning_text = _strip_thought_label(reasoning_text)
                content_text = content_text.lstrip("\n")
                content_text = (prefix or "") + content_text
                return (reasoning_text or None), (content_text or None)
            # Open channel — treat everything after the start token as reasoning.
            reasoning_text = _strip_thought_label(after_start)
            return (reasoning_text or None), (prefix or None)

        # Fallback: delimiters were stripped. Anchor on ``thought*\n`` label.
        match = _THOUGHT_PREFIX_RE.search(model_output)
        if match is None:
            # No thinking prefix found; whole output is content.
            return None, model_output

        idx = match.start()
        pre_content = model_output[:idx]
        rest = model_output[match.end() :]
        # Without the end delimiter, we cannot split reasoning from the
        # final answer reliably. Treat the whole remainder as reasoning and
        # only the text before ``thought*`` as content, matching vLLM
        # upstream behaviour for this degraded case.
        return (rest or None), (pre_content or None)

    # ------------------------------------------------------------------
    # Streaming path — token-id driven
    # ------------------------------------------------------------------

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """Token-id based streaming extraction.

        State is stored on the parser instance, so the serving layer must
        allocate a fresh parser per generation stream / choice.
        """
        # Lazy-init streaming state on first use per-instance.
        if not hasattr(self, "_stream_state_inited"):
            self._in_reasoning: bool = False
            self._reasoning_passed_end: bool = False
            self._reasoning_prefix_buf: str = ""
            self._prefix_stripped: bool = False
            self._stream_state_inited = True

        # Phase transition: start token appears in this delta.
        if self.start_token_id in delta_token_ids:
            self._in_reasoning = True
            self._reasoning_passed_end = False
            self._reasoning_prefix_buf = ""
            self._prefix_stripped = False
            return None

        # Phase transition: end token appears in this delta.
        if self.end_token_id in delta_token_ids:
            self._in_reasoning = False
            self._reasoning_passed_end = True
            return None

        # Backfill: we may have missed a start token that arrived alone.
        if not self._in_reasoning and not self._reasoning_passed_end:
            if self.start_token_id in current_token_ids:
                self._in_reasoning = True

        if self._in_reasoning:
            # Detect that the end token already arrived in a prior delta.
            if self.end_token_id in current_token_ids:
                idx = len(current_token_ids) - 1
                while idx >= 0 and current_token_ids[idx] != self.end_token_id:
                    idx -= 1
                delta_idx_start = len(current_token_ids) - len(delta_token_ids)
                if delta_idx_start > idx:
                    self._in_reasoning = False
                    self._reasoning_passed_end = True
                    return DeltaMessage(content=delta_text) if delta_text else None

            if not delta_text:
                return None

            # Strip the ``thought*\n`` role label once across buffered deltas.
            if self._prefix_stripped:
                return DeltaMessage(reasoning=delta_text)

            self._reasoning_prefix_buf += delta_text

            match = _THOUGHT_PREFIX_RE.match(self._reasoning_prefix_buf)
            if match is not None:
                total_consumed = match.end()
                prev_len = len(self._reasoning_prefix_buf) - len(delta_text)
                self._prefix_stripped = True
                if total_consumed >= len(self._reasoning_prefix_buf):
                    # Entire delta (and earlier buffer) was label — nothing left.
                    return None
                if total_consumed <= prev_len:
                    # Label ended before this delta — emit full delta.
                    return DeltaMessage(reasoning=delta_text)
                # Label ends in the middle of this delta.
                stripped = self._reasoning_prefix_buf[total_consumed:]
                return DeltaMessage(reasoning=stripped) if stripped else None

            # No match yet — could still be mid-label; keep buffering unless
            # a newline already arrived (meaning label must have matched).
            if "\n" in self._reasoning_prefix_buf:
                # Safety valve: shouldn't happen given the regex, but if so
                # just flush what we have as reasoning.
                flushed = self._reasoning_prefix_buf
                self._reasoning_prefix_buf = ""
                self._prefix_stripped = True
                return DeltaMessage(reasoning=flushed)
            return None

        if self._reasoning_passed_end:
            return DeltaMessage(content=delta_text) if delta_text else None

        # Before any start token: content.
        return DeltaMessage(content=delta_text) if delta_text else None


def _strip_thought_label(text: str) -> str:
    """Remove the ``thought*\\n`` role label from the beginning of text.

    Tolerates sampling variants such as ``thoughten-US\\n`` or ``thoughte\\n``.
    """
    return _THOUGHT_PREFIX_RE.sub("", text, count=1)
