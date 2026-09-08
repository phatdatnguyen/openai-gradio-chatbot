"""Token counting and history trimming.

The important detail here is the encoder fallback. ``tiktoken.encoding_for_model``
raises KeyError for every gpt-5.x model, and the old code caught that with a bare
``except`` and fell back to ``cl100k_base`` -- the wrong tokenizer. On Vietnamese
text with diacritics cl100k counts ~1.78x more tokens than the correct o200k_base,
so history was being trimmed far more aggressively than necessary.
"""

import base64
import binascii
import io
from functools import lru_cache

import tiktoken

import config
from log import logger
from messages import is_data_url, serialize_message_value

# gpt-4o and everything after it uses o200k_base; cl100k_base is only correct for
# the gpt-4 / gpt-3.5 generation, which tiktoken already knows about by name.
FALLBACK_ENCODING = "o200k_base"

# Rough per-message framing overhead (role, separators) in the chat format.
TOKENS_PER_MESSAGE = 4
TOKENS_PER_REQUEST = 2


class InputTooLargeError(Exception):
    """Raised when even the newest single message cannot fit the context window.

    Without this, trim_history returned an empty list and the API was called with
    no messages at all, producing an opaque 400.
    """

    def __init__(self, model, needed_tokens, budget_tokens):
        self.model = model
        self.needed_tokens = needed_tokens
        self.budget_tokens = budget_tokens
        super().__init__(
            f"This input is about {needed_tokens:,} tokens, which does not fit "
            f"{model}'s usable context of {budget_tokens:,} tokens. Try a shorter "
            "document, or pick a model with a larger context window."
        )


@lru_cache(maxsize=None)
def get_encoding(model):
    """Resolve and cache the tokenizer for a model.

    Cached because trim_history asks for it once per message, and the lookup is
    not free.
    """
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        logger.debug("No tiktoken mapping for %s; using %s.", model, FALLBACK_ENCODING)
        return tiktoken.get_encoding(FALLBACK_ENCODING)


def _count_image_tokens(image_url, model, detail="auto"):
    """Read inline-image dimensions locally; never fetch a remote image to count it."""
    from PIL import Image

    if isinstance(image_url, dict):
        image_url = image_url.get("url")
    dimensions = None
    if (model in config.IMAGE_TILE_TOKEN_COSTS or model in config.IMAGE_PATCH_TOKEN_BUDGETS):
        if isinstance(image_url, str) and is_data_url(image_url):
            try:
                data = base64.b64decode(image_url.split(",", 1)[1], validate=True)
                with Image.open(io.BytesIO(data)) as image:
                    dimensions = image.size
            except (ValueError, binascii.Error, OSError, Image.DecompressionBombError):
                # Invalid or opaque input is left for API validation. Its size
                # estimate still has a finite bound, independent of base64 bytes.
                pass
    return config.image_token_budget(model, detail, dimensions)


def _count_content_tokens(content, model):
    """Count text normally and reserve vision tokens for typed image parts."""
    if not isinstance(content, list):
        return count_text_tokens(serialize_message_value(content), model)

    total = 0
    for part in content:
        if isinstance(part, dict) and part.get("type") in ("image_url", "input_image"):
            image_url = part.get("image_url")
            detail = (image_url.get("detail", "auto") if isinstance(image_url, dict)
                      else part.get("detail", "auto"))
            total += _count_image_tokens(image_url, model, detail)
        else:
            total += count_text_tokens(serialize_message_value(part), model)
    return total


def count_message_tokens(message, model):
    total = TOKENS_PER_MESSAGE
    for key, value in message.items():
        if key == "content":
            total += _count_content_tokens(value, model)
        elif key == "image_url" and value:
            # Also support the internal history shape used before API conversion.
            detail = value.get("detail", "auto") if isinstance(value, dict) else "auto"
            total += _count_image_tokens(value, model, detail)
        else:
            total += count_text_tokens(serialize_message_value(value), model)
    return total


def count_tokens(messages, model):
    total = TOKENS_PER_REQUEST
    for message in messages:
        total += count_message_tokens(message, model)
    return total


def count_text_tokens(text, model):
    if not text:
        return 0
    # Special-token spellings in user text must not trigger tiktoken's local
    # validation: the API receives these as ordinary text, not control tokens.
    return len(get_encoding(model).encode(text, disallowed_special=()))


def reserve_for_output(model):
    """How much of the context window to hold back for the model's own reply.

    For the gpt-5 generation the advertised window covers input *and* output, so
    the old flat 2000-token reserve was far too small. We hold back the model's
    documented max output, capped so a 128k reserve does not eat a 128k window.
    """
    documented = config.max_output_tokens(model)
    if documented is None:
        documented = config.DEFAULT_OUTPUT_RESERVE_TOKENS
    context = config.get_max_context_tokens(model)
    return min(documented, config.MAX_OUTPUT_RESERVE_TOKENS, max(context // 4, 1024))


def trim_history(messages, model, web_search=False, system_prompt="", reserved_tokens=None):
    """Drop the oldest messages until the payload fits the model's context window.

    The system prompt is charged against the budget even though it is prepended
    after trimming, and the newest message is never silently dropped -- if it
    alone cannot fit, InputTooLargeError is raised so the caller can say so.
    If an older user's prompt is dropped, discard its orphaned assistant reply
    as well whenever a newer user turn is retained.
    """
    if not messages:
        return []

    if reserved_tokens is None:
        reserved_tokens = reserve_for_output(model)

    max_context = config.get_max_context_tokens(model, web_search)
    system_tokens = 0
    if system_prompt:
        system_tokens = count_message_tokens(
            {"role": "system", "content": system_prompt}, model
        )

    budget = max_context - reserved_tokens - system_tokens - TOKENS_PER_REQUEST
    if budget <= 0:
        raise InputTooLargeError(model, system_tokens, max(max_context - reserved_tokens, 0))

    trimmed = []
    total = 0
    for index, message in enumerate(reversed(messages)):
        message_tokens = count_message_tokens(message, model)
        if total + message_tokens > budget:
            if index == 0:
                raise InputTooLargeError(model, message_tokens, budget)
            break
        trimmed.insert(0, message)
        total += message_tokens

    if len(trimmed) < len(messages) and trimmed[0].get("role") == "assistant":
        first_user = next(
            (index for index, message in enumerate(trimmed) if message.get("role") == "user"),
            None,
        )
        if first_user is not None:
            total -= sum(count_message_tokens(message, model) for message in trimmed[:first_user])
            trimmed = trimmed[first_user:]

    dropped = len(messages) - len(trimmed)
    if dropped:
        logger.info(
            "Trimmed %d old message(s) to fit %s (%d/%d tokens used).",
            dropped, model, total, budget,
        )

    return trimmed
