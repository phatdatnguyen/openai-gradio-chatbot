"""Bounded, cited visual briefs and raster validation for the Graphics tool."""

import base64
import binascii
from io import BytesIO
import re
import warnings

from chatbot import config
from .evidence import require_object, require_text, validate_source_ids


MAX_PROMPT_CHARACTERS = 12000
MAX_CAPTION_CHARACTERS = 4000
MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_IMAGE_PIXELS = 32 * 1024 * 1024
_ASSET_PATH = re.compile(
    r"artifacts/graphics/assets/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}\.(png|jpeg|webp)\Z"
)


def validate_image_options(image_options=None):
    """Resolve defaults and reject bad options before the evidence pipeline runs."""
    if image_options is None:
        image_options = {}
    if not isinstance(image_options, dict):
        raise ValueError("Graphics image options must be an object.")
    model = image_options.get("image_model", config.DEFAULT_IMAGE_MODEL)
    if not isinstance(model, str) or model not in config.IMAGE_MODEL_CONFIGS:
        raise ValueError("Choose a supported image model.")
    settings = config.get_image_model_config(model)
    values = {
        "image_model": model,
        "image_size": settings["default_size"],
        "image_quality": settings["default_quality"],
        "image_background": settings["default_background"],
        "image_output_format": "png",
        "image_output_compression": 100,
        "image_moderation": "auto",
        "image_input_fidelity": settings["default_input_fidelity"],
    }
    if set(image_options) - set(values):
        raise ValueError("Graphics image options contain an unknown setting.")
    values.update(image_options)
    values["image_output_compression"] = config.validate_image_options(**values)
    return values


def validate_graphics(content, *, allowed_source_ids=None, persisted=False):
    expected = {"prompt", "caption", "source_ids"}
    if persisted:
        expected.update({"asset_path", "output_format", "width", "height", "revised_prompt"})
    require_object(content, expected)
    require_text(content["prompt"], limit=MAX_PROMPT_CHARACTERS)
    require_text(content["caption"], limit=MAX_CAPTION_CHARACTERS)
    citations = content["source_ids"]
    if (not isinstance(citations, list) or not citations
            or any(not isinstance(item, str) or not item.strip() for item in citations)
            or len(citations) != len(set(citations))):
        raise ValueError("Graphics must contain nonempty, unique source citations.")
    validate_source_ids(citations, allowed_source_ids if allowed_source_ids is not None else set(citations))
    if persisted:
        path = content["asset_path"]
        match = _ASSET_PATH.fullmatch(path) if isinstance(path, str) else None
        if not match or content["output_format"] != match.group(1):
            raise ValueError("Graphics image paths must refer to a scoped raster asset.")
        width, height = content["width"], content["height"]
        if (type(width) is not int or type(height) is not int or width < 1 or height < 1
                or width * height > MAX_IMAGE_PIXELS):
            raise ValueError("Graphics image dimensions exceed the supported bounds.")
        revised = content["revised_prompt"]
        if not isinstance(revised, str) or len(revised) > MAX_PROMPT_CHARACTERS * 2:
            raise ValueError("Graphics revised prompts exceed the supported bounds.")
    return content


def validate_raster(data, output_format):
    """Decode only bounded, single-frame raster images; never render markup."""
    if not isinstance(output_format, str) or output_format not in {"png", "jpeg", "webp"}:
        raise ValueError("Graphics supports PNG, JPEG, and WebP raster images.")
    if not isinstance(data, bytes) or not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Graphics image exceeds the supported 64 MiB size.")
    from PIL import Image, UnidentifiedImageError

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                width, height = image.size
                if (image.format != {"png": "PNG", "jpeg": "JPEG", "webp": "WEBP"}[output_format]
                        or getattr(image, "n_frames", 1) != 1):
                    raise ValueError("Graphics image does not match the requested raster format.")
                if width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS:
                    raise ValueError("Graphics image dimensions exceed the supported bounds.")
                image.verify()
            # verify() checks structure; a load also rejects truncated pixel data.
            with Image.open(BytesIO(data)) as image:
                image.load()
        return width, height
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError("The image model returned an invalid or oversized raster image.") from exc


def decode_image_asset(value):
    if not isinstance(value, dict):
        raise ValueError("The image model returned no usable raster image.")
    encoded, output_format = value.get("b64_json"), value.get("output_format")
    if not isinstance(encoded, str) or len(encoded) > ((MAX_IMAGE_BYTES + 2) // 3) * 4:
        raise ValueError("The image model returned an invalid or oversized raster image.")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("The image model returned invalid image data.") from exc
    width, height = validate_raster(data, output_format)
    revised = value.get("revised_prompt")
    if revised is None:
        revised = ""
    if not isinstance(revised, str) or len(revised) > MAX_PROMPT_CHARACTERS * 2:
        raise ValueError("The image model returned an invalid revised prompt.")
    return data, {"output_format": output_format, "width": width, "height": height,
                  "revised_prompt": revised}
