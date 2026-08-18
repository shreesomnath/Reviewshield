"""Frontier-API backend for preference-pair generation, used in place of
the local Qwen2.5-72B generator (see process.md Step 17 for why: a real,
reproduced KV-cache-reuse bug in the local transformers/torch stack made
local generation either crash or, once worked around with
use_cache=False, impractically slow at the ~8,800-call scale this project
needs).

Exposes the SAME generate(loaded, prompt, max_new_tokens, temperature,
do_sample) signature as generation/model_utils.py's local generate(), so
generate_preference_pairs.py and response_judge.py can use either backend
interchangeably without other code changes.

Default model is gemini-flash-latest (a maintained alias, not a hardcoded
version, to avoid the exact "model no longer available" deprecation this
project already hit once with gemini-2.0-flash). Handles two frontier-
specific failure modes generic to any API-based pipeline, neither of which
the local pipeline had to worry about: rate limiting (retried with
exponential backoff) and safety/content-policy refusals (expected
occasionally, since we deliberately feed the model adversarial prompt-
injection text - logged and returned as an empty string rather than
raising, so one blocked call does not crash the whole batch run).
"""
import os
from dataclasses import dataclass

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from loguru import logger

DEFAULT_MODEL_ID = "gemini-flash-lite-latest"
# NOT "gemini-flash-latest": that model does invisible extended "thinking"
# by default with no way to disable it (this SDK's GenerationConfig has no
# thinking_config field - confirmed by testing, not assumed) and it ate the
# entire token budget on several real calls, discarding them with zero
# visible output. The "lite" tier does not do this - confirmed directly:
# the same prompt at only 1,500 max_output_tokens completed normally
# (finish_reason=STOP) with candidates_token_count + prompt_token_count
# exactly equal to total_token_count, i.e. zero hidden tokens.


@dataclass
class LoadedFrontierModel:
    model: object
    model_id: str


# Real, exact token accounting (from resp.usage_metadata, not an estimate) -
# module-level so any script using this backend can report precise usage
# after a run, instead of the guessed/derived cost figures used earlier.
_USAGE = {"calls": 0, "prompt_tokens": 0, "output_tokens": 0}


def get_usage_summary() -> dict:
    return dict(_USAGE)


def reset_usage_summary() -> None:
    _USAGE["calls"] = 0
    _USAGE["prompt_tokens"] = 0
    _USAGE["output_tokens"] = 0


def load_frontier_model(model_id: str = DEFAULT_MODEL_ID) -> LoadedFrontierModel:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set - pass --env GEMINI_API_KEY=... at runtime")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_id)
    return LoadedFrontierModel(model=model, model_id=model_id)


class _RetriableAPIError(Exception):
    pass


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    retry=retry_if_exception_type(_RetriableAPIError),
    reraise=True,
)
def _call_with_retry(model, prompt: str, max_new_tokens: int, temperature: float, do_sample: bool):
    try:
        gen_config = genai.types.GenerationConfig(
            max_output_tokens=max_new_tokens,
            temperature=temperature if do_sample else 0.0,
        )
        return model.generate_content(prompt, generation_config=gen_config)
    except Exception as e:
        msg = str(e).lower()
        # Credit exhaustion is a 429 too, but it is PERMANENT - no amount of
        # backoff fixes an empty prepaid balance. Observed directly: the
        # first full run wasted its final several minutes retrying this
        # exact error 5x with growing backoff before finally giving up.
        # Fail immediately instead so a depleted-credit stop is fast and
        # obvious, not disguised as a transient rate-limit retry loop.
        if "prepayment credits are depleted" in msg or "billing" in msg:
            logger.error(f"Permanent billing/credit error, not retrying: {e}")
            raise
        if "resource_exhausted" in msg or "429" in msg or "rate" in msg or "quota" in msg:
            logger.warning(f"Rate/quota hit, retrying with backoff: {e}")
            raise _RetriableAPIError(str(e)) from e
        raise


# Confirmed against the actual installed library
# (google.ai.generativelanguage.Candidate.FinishReason), not guessed: STOP=1
# and MAX_TOKENS=2 are both normal, USABLE outcomes (a max-tokens cutoff
# just means the response is truncated, not blocked) - only the remaining
# values are genuine safety/policy blocks worth skipping. An earlier version
# of this check treated any non-STOP reason as a block, which silently
# discarded every real MAX_TOKENS response (caught immediately in a
# --limit 1 test: 2 calls, 2 skips, 0 usable output, before this was fixed).
_USABLE_FINISH_REASONS = {1, 2}  # STOP, MAX_TOKENS

# 3000, not a few hundred: gemini-flash-latest spends part of max_output_tokens
# on invisible "thinking" tokens not present in resp.text (no thinking_budget
# control is exposed by this SDK's GenerationConfig - checked the actual
# installed signature, not assumed). Measured directly on a real 2,337-token
# review prompt: ~900-1,000 tokens of thinking consumed before any visible
# output; at a 600-token cap the visible review was cut to 22 tokens (104
# chars) before hitting the cap. At 3,000 the same prompt completed normally
# (finish_reason=STOP) with an 805-token, 4,113-character full review.
def generate(loaded: LoadedFrontierModel, prompt: str, max_new_tokens: int = 1500,
             temperature: float = 0.7, do_sample: bool = True) -> str:
    resp = _call_with_retry(loaded.model, prompt, max_new_tokens, temperature, do_sample)

    usage = getattr(resp, "usage_metadata", None)
    if usage is not None:
        _USAGE["calls"] += 1
        _USAGE["prompt_tokens"] += getattr(usage, "prompt_token_count", 0) or 0
        _USAGE["output_tokens"] += getattr(usage, "candidates_token_count", 0) or 0

    # Safety/content-policy blocks: candidates may be empty or finish_reason
    # may indicate a genuine block. Since we deliberately send adversarial
    # prompt-injection text, an occasional real block is expected - log and
    # return empty rather than crash the batch.
    if not resp.candidates:
        logger.warning("Empty candidates (likely a safety block) - skipping this generation")
        return ""
    candidate = resp.candidates[0]
    finish_reason = getattr(candidate, "finish_reason", None)
    reason_value = int(finish_reason) if finish_reason is not None else None
    if reason_value not in _USABLE_FINISH_REASONS:
        logger.warning(f"Non-usable finish_reason={finish_reason} ({reason_value}) - skipping")
        return ""

    try:
        return resp.text
    except Exception as e:
        logger.warning(f"Could not extract text from response ({e}) - skipping this generation")
        return ""
