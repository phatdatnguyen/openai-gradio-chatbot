"""Model metadata, capability predicates and the shared OpenAI client.

Everything the rest of the app needs to know about *which* model can do *what*
lives here. The capability predicates at the bottom exist because the model
tables kept drifting out of sync as models were added one commit at a time --
``check_model_tables()`` is the guardrail against that.
"""

import os

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

# Reasoning models that reject system/developer messages outright.
SYSTEM_MESSAGE_UNSUPPORTED_MODELS = {"o1-mini", "o3-mini"}

# gpt-5-shaped models that are conventional chat models, not reasoning models:
# they accept temperature/top_p and have no reasoning effort control.
NON_REASONING_GPT5_MODELS = {"gpt-5-chat-latest", "gpt-5.3-chat"}

REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5")

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

DEFAULT_MODEL = "gpt-5.5"
DEFAULT_IMAGE_MODEL = "gpt-image-2"

# Unknown models get a modern default rather than the old 4096, which used to
# crush any newly added model that was missing from MODEL_TOKEN_LIMITS.
DEFAULT_CONTEXT_TOKENS = 128000

# Used when a model has no explicit MODEL_MAX_OUTPUT_TOKENS entry.
DEFAULT_OUTPUT_RESERVE_TOKENS = 8192
MAX_OUTPUT_RESERVE_TOKENS = 32768

NO_REASONING_EFFORT = "auto"


def get_image_model_config(image_model):
    return IMAGE_MODEL_CONFIGS.get(image_model, IMAGE_MODEL_CONFIGS[DEFAULT_IMAGE_MODEL])


def get_max_context_tokens(model_name, web_search=False):
    if web_search:
        return MODEL_TOKEN_LIMITS_WITH_WEB_SEARCH.get(model_name, DEFAULT_CONTEXT_TOKENS)
    return MODEL_TOKEN_LIMITS.get(model_name, DEFAULT_CONTEXT_TOKENS)


def uses_responses_api(model_name):
    return model_name in RESPONSES_API_MODELS


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


def max_output_tokens(model_name):
    """Documented max output tokens, or None when unknown."""
    return MODEL_MAX_OUTPUT_TOKENS.get(model_name)


def reasoning_effort_choices(model_name):
    """Effort levels this model family accepts, or [] for non-reasoning models."""
    if not is_reasoning_model(model_name):
        return []
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
        "WEB_SEARCH_MODELS": WEB_SEARCH_MODELS,
        "DEEP_RESEARCH_MODELS": DEEP_RESEARCH_MODELS,
        "VISION_DISABLED_MODELS": VISION_DISABLED_MODELS,
        "SYSTEM_MESSAGE_UNSUPPORTED_MODELS": SYSTEM_MESSAGE_UNSUPPORTED_MODELS,
        "NON_REASONING_GPT5_MODELS": NON_REASONING_GPT5_MODELS,
    }
    for table_name, table in auxiliary_tables.items():
        unknown = sorted(set(table) - known)
        if unknown:
            problems.append(f"unknown models in {table_name}: {unknown}")

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
