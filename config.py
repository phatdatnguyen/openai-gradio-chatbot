"""Model metadata, capability predicates and the shared OpenAI client.

Everything the rest of the app needs to know about *which* model can do *what*
lives here. The capability predicates at the bottom exist because the model
tables kept drifting out of sync as models were added one commit at a time --
``check_model_tables()`` is the guardrail against that.
"""

import os
from math import ceil, floor

from openai import OpenAI

from log import logger

try:
    from api_key import API_KEY as FILE_API_KEY
except ImportError:
    FILE_API_KEY = None


MISSING_KEY_MESSAGE = (
    "No OpenAI API key found. Either set the OPENAI_API_KEY environment variable, "
    "or create api_key.py next to webui.py containing:  API_KEY = \"sk-...\""
)


def _build_client():
    api_key = os.getenv("OPENAI_API_KEY") or FILE_API_KEY
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


# Built eagerly so a bad key surfaces at startup, but a *missing* key no longer
# crashes the import with a raw traceback -- get_client() explains the problem.
client = _build_client()


def get_client():
    if client is None:
        raise RuntimeError(MISSING_KEY_MESSAGE)
    return client


MODEL_TOKEN_LIMITS = {
    "gpt-3.5-turbo": 4096,
    "gpt-4": 8192,
    "gpt-4-turbo": 128000,
    "gpt-4.1": 1047576,
    "gpt-4.1-mini": 1047576,
    "gpt-4.1-nano": 1047576,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-5": 400000,
    "gpt-5-chat-latest": 128000,
    "gpt-5-pro": 400000,
    "gpt-5-mini": 400000,
    "gpt-5-nano": 400000,
    "gpt-5.1": 400000,
    "gpt-5.1-codex": 400000,
    "gpt-5.1-codex-max": 400000,
    "gpt-5.1-codex-mini": 400000,
    "gpt-5.2": 400000,
    "gpt-5.2-codex": 400000,
    "gpt-5.2-pro": 400000,
    "gpt-5.3-codex": 400000,
    "gpt-5.3-chat": 400000,
    "gpt-5.4": 1050000,
    "gpt-5.4-nano": 400000,
    "gpt-5.4-mini": 400000,
    "gpt-5.4-pro": 1050000,
    "gpt-5.5": 1050000,
    "gpt-5.5-pro": 1050000,
    "gpt-5.6-luna": 1050000,
    "gpt-5.6-terra": 1050000,
    "gpt-5.6-sol": 1050000,
    "gpt-6-astra": 1050000,
    "o1": 128000,
    "o1-mini": 128000,
    "o1-pro": 128000,
    "o3": 200000,
    "o3-mini": 200000,
    "o3-pro": 200000,
    "o3-deep-research": 200000,
    "o4-mini": 200000,
    "o4-mini-deep-research": 200000,
}

MODEL_TOKEN_LIMITS_WITH_WEB_SEARCH = {
    "gpt-4.1": 128000,
    "gpt-4.1-mini": 128000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-5": 400000,
    "gpt-5-chat-latest": 128000,
    "gpt-5-pro": 400000,
    "gpt-5-mini": 400000,
    "gpt-5.1": 400000,
    "gpt-5.2": 400000,
    "gpt-5.2-pro": 400000,
    "gpt-5.4": 1050000,
    "gpt-5.4-nano": 400000,
    "gpt-5.4-mini": 400000,
    "gpt-5.4-pro": 1050000,
    "gpt-5.5": 1050000,
    "gpt-5.5-pro": 1050000,
    "gpt-5.6-luna": 128000,
    "gpt-5.6-terra": 128000,
    "gpt-5.6-sol": 128000,
    "gpt-6-astra": 128000,
    "o1": 128000,
    "o3": 128000,
    "o3-pro": 128000,
    "o3-deep-research": 128000,
    "o4-mini-deep-research": 128000,
}

# Used to size the output reserve when trimming history (see tokens.reserve_for_output).
MODEL_MAX_OUTPUT_TOKENS = {
    "gpt-5.1-codex-max": 128000,
    "gpt-5.2-codex": 128000,
    "gpt-5.3-codex": 128000,
    "gpt-5.3-chat": 128000,
    "gpt-5.4": 128000,
    "gpt-5.4-nano": 128000,
    "gpt-5.4-mini": 128000,
    "gpt-5.4-pro": 128000,
    "gpt-5.5": 128000,
    "gpt-5.5-pro": 128000,
    "gpt-5.6-luna": 128000,
    "gpt-5.6-terra": 128000,
    "gpt-5.6-sol": 128000,
    "gpt-6-astra": 128000,
}

# Models only reachable through client.responses.*, never chat.completions.
# The "-pro" family is Responses-only across the board; gpt-5-pro / gpt-5.2-pro /
# gpt-5.4-pro used to be missing here, which silently routed them to the wrong API.
RESPONSES_API_MODELS = {
    "o1-pro",
    "o3-pro",
    "o3-deep-research",
    "o4-mini-deep-research",
    "gpt-5-pro",
    "gpt-5.1-codex",
    "gpt-5.1-codex-max",
    "gpt-5.1-codex-mini",
    "gpt-5.2-codex",
    "gpt-5.2-pro",
    "gpt-5.3-codex",
    "gpt-5.4-pro",
    "gpt-5.5-pro",
}

# These models need a blocking response even when the UI requests streaming.
# See the Features section of each model's official API documentation.
NON_STREAMING_MODELS = {"o1-pro", "o3-pro", "gpt-5.5-pro"}

WEB_SEARCH_MODELS = {
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-5",
    "gpt-5-chat-latest",
    "gpt-5-pro",
    "gpt-5-mini",
    "gpt-5.1",
    "gpt-5.2",
    "gpt-5.2-pro",
    "gpt-5.4",
    "gpt-5.4-nano",
    "gpt-5.4-mini",
    "gpt-5.4-pro",
    "gpt-5.5",
    "gpt-5.5-pro",
    "gpt-5.6-luna",
    "gpt-5.6-terra",
    "gpt-5.6-sol",
    "gpt-6-astra",
    "o3",
    "o3-pro",
}

# Models that must always search (no "None" option offered in the UI).
DEEP_RESEARCH_MODELS = {"o3-deep-research", "o4-mini-deep-research"}

# The current GA tool name. The old "web_search_preview" alias is not accepted by
# the newer model families.
WEB_SEARCH_TOOL_TYPE = "web_search"

WEB_SEARCH_CONTEXT_CHOICES = ["None", "low", "medium", "high"]
WEB_SEARCH_OFF = "None"

# Base gpt-4 is text-only; it used to be missing here, so image uploads produced a
# raw API error instead of a friendly warning.
VISION_DISABLED_MODELS = {"gpt-3.5-turbo", "gpt-4", "o1-mini", "o3-mini"}

# Conservative per-image context budgets, independent of PNG/JPEG compression.
# Source: https://developers.openai.com/api/docs/guides/images-vision
# Tile images fit within 2048px and have a shortest side at most 768px, so no
# more than 4 * 2 tiles are needed. Values are (base tokens, tokens per tile).
IMAGE_TILE_TOKEN_COSTS = {
    "gpt-4o": (85, 170),
    "gpt-4o-mini": (2833, 5667),
    "gpt-4.1": (85, 170),
    "gpt-5": (70, 140),
    "gpt-5.1": (70, 140),
    "o1": (75, 150),
    "o1-pro": (75, 150),
    "o3": (75, 150),
}

# Bounds use the documented patch budgets multiplied by each model's rate,
# rounded up. They intentionally reserve the maximum for the selected detail;
# small images generally use fewer tokens. auto differs between model families.
IMAGE_PATCH_TOKEN_BUDGETS = {
    "gpt-4.1-mini": {"low": 9954, "high": 9954, "auto": 9954},
    "gpt-5.2": {"low": 7373, "high": 7373, "auto": 7373},
    "gpt-5.4": {"low": 7373, "high": 3000, "original": 12000, "auto": 3000},
    "gpt-5.4-mini": {"low": 7373, "high": 3000, "original": 12000, "auto": 3000},
    "gpt-5.4-nano": {"low": 7373, "high": 3000, "original": 12000, "auto": 3000},
    "gpt-5.5": {"low": 308, "high": 3000, "original": 12000, "auto": 12000},
    "gpt-5.6-sol": {"low": 308, "high": 3000, "original": 36000, "auto": 36000},
    "gpt-5.6-terra": {"low": 308, "high": 3000, "original": 36000, "auto": 36000},
    "gpt-5.6-luna": {"low": 308, "high": 3000, "original": 36000, "auto": 36000},
}
IMAGE_PATCH_TOKEN_MULTIPLIERS = {
    "gpt-4.1-mini": 1.62,
    "gpt-5.2": 1.2,
    "gpt-5.4": 1.2,
    "gpt-5.4-mini": 1.2,
    "gpt-5.4-nano": 1.2,
    "gpt-5.5": 1.2,
    "gpt-5.6-sol": 1.2,
    "gpt-5.6-terra": 1.2,
    "gpt-5.6-luna": 1.2,
}

# For variants without documented sizing rules, use the largest documented
# patch cap (30,000) times the largest listed multiplier (2.46). This is a
# conservative fallback estimate, not a documented guarantee for unknown models.
DEFAULT_IMAGE_TOKEN_BUDGET = 73800

# Reasoning models that reject system/developer messages outright.
SYSTEM_MESSAGE_UNSUPPORTED_MODELS = {"o1-mini", "o3-mini"}

# gpt-5-shaped models that are conventional chat models, not reasoning models:
# they accept temperature/top_p and have no reasoning effort control.
NON_REASONING_GPT5_MODELS = {"gpt-5-chat-latest", "gpt-5.3-chat"}

REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5", "gpt-6")

IMAGE_MODEL_CONFIGS = {
    "gpt-image-2": {
        "size_choices": ["auto", "1024x1024", "1536x1024", "1024x1536", "2048x2048", "2048x1152", "3840x2160", "2160x3840"],
        "default_size": "auto",
        "quality_choices": ["auto", "low", "medium", "high"],
        "default_quality": "auto",
        "background_choices": ["auto", "opaque"],
        "default_background": "auto",
        "input_fidelity_choices": ["high"],
        "default_input_fidelity": "high",
        "input_fidelity_interactive": False,
    },
    "gpt-image-1.5": {
        "size_choices": ["auto", "1024x1024", "1536x1024", "1024x1536"],
        "default_size": "auto",
        "quality_choices": ["auto", "low", "medium", "high"],
        "default_quality": "auto",
        "background_choices": ["auto", "opaque", "transparent"],
        "default_background": "auto",
        "input_fidelity_choices": ["low", "high"],
        "default_input_fidelity": "high",
        "input_fidelity_interactive": True,
    },
}

IMAGE_MODEL_CHOICES = list(IMAGE_MODEL_CONFIGS.keys())
IMAGE_OUTPUT_FORMAT_CHOICES = ["png", "jpeg", "webp"]
IMAGE_MODERATION_CHOICES = ["auto", "low"]
COMPRESSIBLE_IMAGE_FORMATS = ("jpeg", "webp")

MODEL_CHOICES = [
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4-turbo",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-5",
    "gpt-5-chat-latest",
    "gpt-5-pro",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5.1",
    "gpt-5.1-codex",
    "gpt-5.1-codex-max",
    "gpt-5.1-codex-mini",
    "gpt-5.2",
    "gpt-5.2-codex",
    "gpt-5.2-pro",
    "gpt-5.3-codex",
    "gpt-5.3-chat",
    "gpt-5.4",
    "gpt-5.4-nano",
    "gpt-5.4-mini",
    "gpt-5.4-pro",
    "gpt-5.5",
    "gpt-5.5-pro",
    "gpt-5.6-luna",
    "gpt-5.6-terra",
    "gpt-5.6-sol",
    "gpt-6-astra",
    "o1",
    "o1-mini",
    "o1-pro",
    "o3",
    "o3-mini",
    "o3-pro",
    "o3-deep-research",
    "o4-mini",
    "o4-mini-deep-research",
]

DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_IMAGE_MODEL = "gpt-image-2"

# Unknown models get a modern default rather than the old 4096, which used to
# crush any newly added model that was missing from MODEL_TOKEN_LIMITS.
DEFAULT_CONTEXT_TOKENS = 128000

# Used when a model has no explicit MODEL_MAX_OUTPUT_TOKENS entry.
DEFAULT_OUTPUT_RESERVE_TOKENS = 8192
MAX_OUTPUT_RESERVE_TOKENS = 32768

NO_REASONING_EFFORT = "auto"

# Documented exceptions to the broad family defaults below. Each model's page
# at https://developers.openai.com/api/docs/models lists its supported efforts.
REASONING_EFFORT_OVERRIDES = {
    "gpt-5-pro": ("high",),
    "gpt-5.2-pro": ("medium", "high", "xhigh"),
    "gpt-5.4-pro": ("medium", "high", "xhigh"),
    "gpt-5.5-pro": ("medium", "high", "xhigh"),
    "gpt-5.2-codex": ("low", "medium", "high", "xhigh"),
    "gpt-5.3-codex": ("low", "medium", "high", "xhigh"),
    "gpt-5.2": ("none", "low", "medium", "high", "xhigh"),
    "gpt-5.4": ("none", "low", "medium", "high", "xhigh"),
    "gpt-5.4-mini": ("none", "low", "medium", "high", "xhigh"),
    "gpt-5.4-nano": ("none", "low", "medium", "high", "xhigh"),
    "gpt-5.5": ("none", "low", "medium", "high", "xhigh"),
}


def get_image_model_config(image_model):
    return IMAGE_MODEL_CONFIGS.get(image_model, IMAGE_MODEL_CONFIGS[DEFAULT_IMAGE_MODEL])


def get_max_context_tokens(model_name, web_search=False):
    if web_search:
        return MODEL_TOKEN_LIMITS_WITH_WEB_SEARCH.get(model_name, DEFAULT_CONTEXT_TOKENS)
    return MODEL_TOKEN_LIMITS.get(model_name, DEFAULT_CONTEXT_TOKENS)


def uses_responses_api(model_name):
    return model_name in RESPONSES_API_MODELS


def supports_streaming(model_name):
    return model_name not in NON_STREAMING_MODELS


def supports_web_search(model_name):
    return model_name in WEB_SEARCH_MODELS or model_name in DEEP_RESEARCH_MODELS


def requires_web_search(model_name):
    return model_name in DEEP_RESEARCH_MODELS


def is_reasoning_model(model_name):
    if model_name in NON_REASONING_GPT5_MODELS:
        return False
    return model_name.startswith(REASONING_MODEL_PREFIXES)


def supports_temperature(model_name):
    """Reasoning models reject temperature/top_p, so we omit them entirely."""
    return not is_reasoning_model(model_name)


def supports_system_message(model_name):
    return model_name not in SYSTEM_MESSAGE_UNSUPPORTED_MODELS


def supports_vision(model_name):
    return model_name not in VISION_DISABLED_MODELS


def image_token_budget(model_name, detail="auto", dimensions=None):
    """Approximate upper-bound image tokens, without counting transport bytes."""
    tile_costs = IMAGE_TILE_TOKEN_COSTS.get(model_name)
    if tile_costs is not None:
        base, per_tile = tile_costs
        if detail == "low":
            return base
        tiles = 8
        if dimensions:
            width, height = dimensions
            scale = min(1, 2048 / max(width, height))
            width, height = width * scale, height * scale
            if min(width, height) > 768:
                scale = 768 / min(width, height)
                width, height = floor(width * scale), floor(height * scale)
            tiles = ceil(width / 512) * ceil(height / 512)
        return base + tiles * per_tile
    patch_budgets = IMAGE_PATCH_TOKEN_BUDGETS.get(model_name)
    if patch_budgets is not None:
        budget = patch_budgets.get(detail, patch_budgets["auto"])
        if dimensions:
            width, height = dimensions
            patches = ceil(width / 32) * ceil(height / 32)
            # Resizing only reduces dimensions. Counting the original patches,
            # capped at the detail budget, stays conservative without reproducing
            # the API's pixel-rounding procedure for every model family.
            return min(budget, ceil(patches * IMAGE_PATCH_TOKEN_MULTIPLIERS[model_name]))
        return budget
    logger.debug(
        "No image-sizing rule for %s; reserving an approximate %d tokens per image.",
        model_name, DEFAULT_IMAGE_TOKEN_BUDGET,
    )
    return DEFAULT_IMAGE_TOKEN_BUDGET


def max_output_tokens(model_name):
    """Documented max output tokens, or None when unknown."""
    return MODEL_MAX_OUTPUT_TOKENS.get(model_name)


def reasoning_effort_choices(model_name):
    """Effort levels this model family accepts, or [] for non-reasoning models."""
    if not is_reasoning_model(model_name):
        return []
    if model_name in REASONING_EFFORT_OVERRIDES:
        return [NO_REASONING_EFFORT, *REASONING_EFFORT_OVERRIDES[model_name]]
    if model_name.startswith("gpt-6"):
        return [NO_REASONING_EFFORT, "low", "medium", "high", "xhigh", "max"]
    if model_name.startswith("gpt-5.6"):
        return [NO_REASONING_EFFORT, "none", "low", "medium", "high", "xhigh", "max"]
    if model_name.startswith(("o1", "o3", "o4")):
        return [NO_REASONING_EFFORT, "low", "medium", "high"]
    if model_name == "gpt-5" or model_name.startswith("gpt-5-"):
        return [NO_REASONING_EFFORT, "minimal", "low", "medium", "high"]
    choices = [NO_REASONING_EFFORT, "none", "low", "medium", "high"]
    if "codex-max" in model_name:
        choices.append("xhigh")
    return choices


def check_model_tables():
    """Fail loudly on model-table drift.

    Every selectable model needs a context limit, and every key in the auxiliary
    tables has to be a model you can actually select. This is the check that
    would have caught gpt-5-pro missing from RESPONSES_API_MODELS.
    """
    known = set(MODEL_CHOICES)
    problems = []

    missing_limits = sorted(known - set(MODEL_TOKEN_LIMITS))
    if missing_limits:
        problems.append(f"missing from MODEL_TOKEN_LIMITS: {missing_limits}")

    auxiliary_tables = {
        "MODEL_TOKEN_LIMITS": MODEL_TOKEN_LIMITS,
        "MODEL_TOKEN_LIMITS_WITH_WEB_SEARCH": MODEL_TOKEN_LIMITS_WITH_WEB_SEARCH,
        "MODEL_MAX_OUTPUT_TOKENS": MODEL_MAX_OUTPUT_TOKENS,
        "RESPONSES_API_MODELS": RESPONSES_API_MODELS,
        "NON_STREAMING_MODELS": NON_STREAMING_MODELS,
        "WEB_SEARCH_MODELS": WEB_SEARCH_MODELS,
        "DEEP_RESEARCH_MODELS": DEEP_RESEARCH_MODELS,
        "VISION_DISABLED_MODELS": VISION_DISABLED_MODELS,
        "IMAGE_TILE_TOKEN_COSTS": IMAGE_TILE_TOKEN_COSTS,
        "IMAGE_PATCH_TOKEN_BUDGETS": IMAGE_PATCH_TOKEN_BUDGETS,
        "IMAGE_PATCH_TOKEN_MULTIPLIERS": IMAGE_PATCH_TOKEN_MULTIPLIERS,
        "SYSTEM_MESSAGE_UNSUPPORTED_MODELS": SYSTEM_MESSAGE_UNSUPPORTED_MODELS,
        "NON_REASONING_GPT5_MODELS": NON_REASONING_GPT5_MODELS,
        "REASONING_EFFORT_OVERRIDES": REASONING_EFFORT_OVERRIDES,
    }
    for table_name, table in auxiliary_tables.items():
        unknown = sorted(set(table) - known)
        if unknown:
            problems.append(f"unknown models in {table_name}: {unknown}")

    if IMAGE_PATCH_TOKEN_BUDGETS.keys() != IMAGE_PATCH_TOKEN_MULTIPLIERS.keys():
        problems.append("image patch budgets and multipliers cover different models")

    if DEFAULT_MODEL not in known:
        problems.append(f"DEFAULT_MODEL {DEFAULT_MODEL!r} is not in MODEL_CHOICES")
    if len(MODEL_CHOICES) != len(known):
        problems.append("MODEL_CHOICES contains duplicates")

    # A model that can search must have a with-web-search context limit, otherwise
    # it silently falls back to the generic default.
    searchable_without_limit = sorted(
        model for model in WEB_SEARCH_MODELS | DEEP_RESEARCH_MODELS
        if model not in MODEL_TOKEN_LIMITS_WITH_WEB_SEARCH
    )
    if searchable_without_limit:
        problems.append(
            "searchable but missing from MODEL_TOKEN_LIMITS_WITH_WEB_SEARCH: "
            f"{searchable_without_limit}"
        )

    # Family rule: the "-pro" and "-codex" families and the deep-research models are
    # all Responses-API-only. This is the check that actually catches the drift that
    # left gpt-5-pro / gpt-5.2-pro / gpt-5.4-pro routing to Chat Completions.
    responses_only_by_family = sorted(
        model for model in known
        if (model.endswith("-pro") or "-codex" in model or model in DEEP_RESEARCH_MODELS)
        and model not in RESPONSES_API_MODELS
    )
    if responses_only_by_family:
        problems.append(
            "pro/codex/deep-research models missing from RESPONSES_API_MODELS: "
            f"{responses_only_by_family}"
        )

    if problems:
        raise ValueError("Model table inconsistencies:\n  - " + "\n  - ".join(problems))

    logger.debug("Model tables consistent (%d models).", len(MODEL_CHOICES))
    return True
