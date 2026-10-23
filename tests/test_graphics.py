"""Graphics generation and raster persistence use local fakes, never paid APIs."""

import base64
from copy import deepcopy
from io import BytesIO
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from chatbot import config
from rag_chatbot.config import Settings
from rag_chatbot.store import LibraryStore
from rag_chatbot.tools import artifacts, graphics, service
from rag_chatbot.tools.exports import export_bytes, graphics_export_filename
from rag_chatbot.workspaces import create_workspace, workspace_path


def image_response(format="png"):
    image = Image.new("RGB", (20, 16), "#284b63")
    buffer = BytesIO()
    image.save(buffer, format={"png": "PNG", "jpeg": "JPEG", "webp": "WEBP"}[format])
    return {"b64_json": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "output_format": format, "revised_prompt": "Keep labels readable."}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "data"))
    create_workspace("Research")
    return "Research"


@pytest.fixture
def payload(workspace):
    return {"tool_id": "graphics", "title": "Measured sample", "model": config.DEFAULT_MODEL,
            "content": {"prompt": "Draw a labeled diagram of a 10 nm sample.",
                        "caption": "The sample measured 10 nm.", "source_ids": ["chunk-one"]},
            "sources": [{"id": "chunk-one", "document_id": "doc-one", "filename": "paper.pdf",
                         "locator": "Page 2", "text": "The sample measured 10 nm.",
                         "source_path": str(workspace_path(workspace) / "sources" / "paper.pdf")}],
            "documents": [{"id": "doc-one", "name": "paper.pdf", "fingerprint": "v1", "generation": "g1"}],
            "coverage": {"documents": 1, "chunks": 1, "processed_chunks": 1},
            "settings": {"image_options": graphics.validate_image_options()}, "image_data": image_response()}


def test_options_default_to_shared_model_and_validate_before_generation():
    options = graphics.validate_image_options()
    assert options["image_model"] == config.DEFAULT_IMAGE_MODEL
    assert options["image_size"] == config.get_image_model_config(config.DEFAULT_IMAGE_MODEL)["default_size"]
    assert options["image_output_compression"] == 100
    assert graphics.validate_image_options({"image_output_compression": 42.0})["image_output_compression"] == 42


@pytest.mark.parametrize("options", [[], "png", {"image_model": "bad"}, {"arbitrary": True},
                                      {"image_size": "99999x99999"}, {"image_quality": "bad"},
                                      {"image_output_compression": float("nan")},
                                      {"image_background": "transparent", "image_output_format": "jpeg"}])
def test_invalid_image_options_fail_before_accessing_sources_or_api(monkeypatch, options):
    monkeypatch.setattr(service, "get_service", lambda _: pytest.fail("source access before option validation"))
    monkeypatch.setattr(service.api, "generate_structured", lambda *a, **k: pytest.fail("paid call"))
    with pytest.raises(ValueError):
        list(service.generate_tool("workspace", "graphics", ["doc"], image_options=options))


@pytest.mark.parametrize("field,value", [("prompt", ""), ("caption", " " * 2),
                                        ("prompt", "x" * 12001), ("caption", "x" * 4001),
                                        ("source_ids", []), ("source_ids", ["unknown"]),
                                        ("source_ids", ["chunk-one", "chunk-one"]),
                                        ("source_ids", [[]])])
def test_brief_bounds_and_citations(payload, field, value):
    payload["content"][field] = value
    with pytest.raises(ValueError):
        graphics.validate_graphics(payload["content"], allowed_source_ids={"chunk-one"})


@pytest.mark.parametrize("format", ["png", "jpeg", "webp"])
def test_bounded_raster_decode_accepts_supported_formats(format):
    data, metadata = graphics.decode_image_asset(image_response(format))
    assert data and metadata == {"output_format": format, "width": 20, "height": 16,
                                 "revised_prompt": "Keep labels readable."}


@pytest.mark.parametrize("damage", ["base64", "html", "wrong_format", "empty", "oversize", "dimensions"])
def test_raster_rejects_invalid_content(monkeypatch, damage):
    value = image_response()
    if damage == "base64":
        value["b64_json"] = "not-base64!!"
    elif damage == "html":
        value["b64_json"] = base64.b64encode(b"<svg><script>bad()</script></svg>").decode("ascii")
    elif damage == "wrong_format":
        value["output_format"] = "jpeg"
    elif damage == "empty":
        value["b64_json"] = ""
    elif damage == "oversize":
        monkeypatch.setattr(graphics, "MAX_IMAGE_BYTES", 5)
    else:
        monkeypatch.setattr(graphics, "MAX_IMAGE_PIXELS", 10)
    with pytest.raises(ValueError):
        graphics.decode_image_asset(value)


def test_raster_rejects_animation_and_truncated_pixels():
    frames = [Image.new("RGB", (20, 16), color) for color in ("red", "blue")]
    buffer = BytesIO()
    frames[0].save(buffer, format="PNG", save_all=True, append_images=frames[1:], duration=100)
    with pytest.raises(ValueError, match="format"):
        graphics.validate_raster(buffer.getvalue(), "png")
    data = base64.b64decode(image_response()["b64_json"])
    with pytest.raises(ValueError):
        graphics.validate_raster(data[:len(data) // 2], "png")


def test_current_result_has_immutable_assets_small_manifest_and_isolated_workspace(workspace, payload):
    assert artifacts.load_current_graphics(workspace) is None
    assert not (workspace_path(workspace) / "artifacts").exists()
    original = deepcopy(payload)
    first = artifacts.save_current_graphics(workspace, payload)
    second = artifacts.save_current_graphics(workspace, {**payload, "title": "New graphic"})
    assert payload == original and "image_data" not in first
    assert first["id"] == second["id"] and first["created_at"] == second["created_at"]
    assert first["content"]["asset_path"] != second["content"]["asset_path"]
    assert artifacts.graphics_image_path(workspace, first).is_file()
    assert artifacts.load_current_graphics(workspace) == second
    assert second["sources"][0]["source_path"] == "sources/paper.pdf"
    current = workspace_path(workspace) / "artifacts/graphics/current.json"
    assert "b64_json" not in current.read_text(encoding="utf-8") and current.stat().st_size < 5000
    assert artifacts.list_artifacts(workspace) == []
    create_workspace("Other")
    assert artifacts.load_current_graphics("Other") is None
    with pytest.raises(FileNotFoundError):
        artifacts.graphics_image_path("Other", first)


@pytest.mark.parametrize("stop_at", [1, 2, 3, 4, 5])
def test_cancellation_at_each_save_boundary_preserves_old_result(workspace, payload, stop_at):
    saved = artifacts.save_current_graphics(workspace, payload)
    current = workspace_path(workspace) / "artifacts/graphics/current.json"
    before, calls = current.read_bytes(), []
    assets = set(current.parent.joinpath("assets").iterdir())

    def cancelled():
        calls.append(True)
        return len(calls) == stop_at

    with pytest.raises(RuntimeError, match="cancelled"):
        artifacts.save_current_graphics(workspace, payload, is_cancelled=cancelled)
    assert current.read_bytes() == before
    assert set(current.parent.joinpath("assets").iterdir()) == assets
    assert artifacts.load_current_graphics(workspace) == saved
    assert not list(current.parent.glob(".pending-*"))


def test_invalid_image_and_manifest_failure_preserve_previous(workspace, payload, monkeypatch):
    artifacts.save_current_graphics(workspace, payload)
    current = workspace_path(workspace) / "artifacts/graphics/current.json"
    before = current.read_bytes()
    original = artifacts._atomic_bytes
    images = set(current.parent.joinpath("assets").iterdir())
    bad = deepcopy(payload)
    bad["image_data"]["b64_json"] = "bad"
    with pytest.raises(ValueError):
        artifacts.save_current_graphics(workspace, bad)

    def fail_manifest(path, data, **kwargs):
        if path.name == "current.json":
            raise OSError("disk full")
        return original(path, data, **kwargs)

    monkeypatch.setattr(artifacts, "_atomic_bytes", fail_manifest)
    with pytest.raises(OSError, match="disk full"):
        artifacts.save_current_graphics(workspace, payload)
    assert current.read_bytes() == before and set(current.parent.joinpath("assets").iterdir()) == images


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_fresh_result_can_replace_damaged_previous_image(workspace, payload, damage):
    saved = artifacts.save_current_graphics(workspace, payload)
    path = artifacts.graphics_image_path(workspace, saved)
    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes(b"broken")
    with pytest.raises((ValueError, FileNotFoundError)):
        artifacts.load_current_graphics(workspace)
    replacement = artifacts.save_current_graphics(workspace, payload)
    assert replacement["id"] == saved["id"] and artifacts.load_current_graphics(workspace) == replacement


@pytest.mark.parametrize("path", ["../outside.png", "sources/paper.png", "/tmp/image.png",
                                  "artifacts/graphics/assets/../image.png", "artifacts/graphics/assets/a.svg"])
def test_saved_image_paths_cannot_escape(workspace, payload, path):
    saved = artifacts.save_current_graphics(workspace, payload)
    saved["content"]["asset_path"] = path
    with pytest.raises(ValueError, match="scoped"):
        artifacts.graphics_image_path(workspace, saved)


def test_linked_image_asset_is_rejected(workspace, payload, tmp_path):
    saved = artifacts.save_current_graphics(workspace, payload)
    path = artifacts.graphics_image_path(workspace, saved)
    outside = tmp_path / "outside.png"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    try:
        path.symlink_to(outside)
    except OSError:
        pytest.skip("Creating symlinks requires privileges on this Windows installation.")
    with pytest.raises(ValueError, match="linked"):
        artifacts.load_current_graphics(workspace)


@pytest.mark.parametrize("format", ["png", "jpeg", "webp"])
def test_export_original_raster_and_complete_metadata(workspace, payload, format):
    payload["image_data"] = image_response(format)
    saved = artifacts.save_current_graphics(workspace, payload)
    path = workspace_path(workspace) / "artifacts/graphics/current.json"
    before = path.read_bytes()
    image = artifacts.export_current_graphics(workspace, saved, "image", "Figure.json")
    assert image.name == f"Figure.{format}"
    assert image.read_bytes() == artifacts.graphics_image_path(workspace, saved).read_bytes()
    metadata = artifacts.export_current_graphics(workspace, saved, "json", "Figure")
    assert json.loads(metadata.read_text(encoding="utf-8")) == saved
    markdown = artifacts.export_current_graphics(workspace, saved, "markdown", "Figure")
    text = markdown.read_text(encoding="utf-8")
    assert "## Caption" in text and "## Visual brief" in text and "[1]" in text
    assert "## Provenance" in text and "Page 2" in text and "Keep labels readable" in text
    assert path.read_bytes() == before and artifacts.list_artifacts(workspace) == []


@pytest.mark.parametrize("filename", ["../image", "sub\\image", "NUL", "CON.png", " image", "image.",
                                      "x" * 101, "界" * 85, "\ud800"])
def test_export_portable_filename_checks(filename):
    with pytest.raises(ValueError, match="filename"):
        graphics_export_filename(filename, "png")


def test_export_cancel_keeps_existing_download(workspace, payload):
    saved = artifacts.save_current_graphics(workspace, payload)
    path = artifacts.export_current_graphics(workspace, saved, "json")
    original = path.read_bytes()
    calls = []

    def cancelled():
        calls.append(True)
        return len(calls) == 2

    with pytest.raises(RuntimeError, match="cancelled"):
        artifacts.export_current_graphics(workspace, {**saved, "title": "Late"}, "json", is_cancelled=cancelled)
    assert path.read_bytes() == original and not list(path.parent.glob(".pending-*"))


@pytest.fixture
def catalog(tmp_path):
    store = LibraryStore(Settings(data_dir=tmp_path / "catalog"))
    documents = []
    for number, count in ((1, 12), (2, 2)):
        path = tmp_path / f"source{number}.txt"
        path.write_text(f"Sample data {number}", encoding="utf-8")
        document = store.stage_document(path, "science", store.settings.fingerprint())
        with store._db:
            for index in range(count):
                store._db.execute(
                    "INSERT INTO chunks(id,document_id,text,locator,kind,ordinal) VALUES (?,?,?,?,?,?)",
                    (f"doc{number}-chunk{index}", document["id"], f"Finding {number}/{index}: 10 nm.",
                     f"page {index + 1}", "text", index))
            store._db.execute("UPDATE documents SET status='ready',generation='published',chunk_count=? WHERE id=?",
                              (count, document["id"]))
        documents.append(document["id"])
    yield store, documents
    store.close()


def fake_generation(monkeypatch, catalog, *, transform=None, after_image=None):
    store, documents = catalog
    calls = []
    monkeypatch.setattr(service, "get_service", lambda _: SimpleNamespace(store=store))

    def structured(model, history, schema, **kwargs):
        prompt = json.loads(history[-1]["content"])
        name = kwargs["schema_name"]
        calls.append((name, prompt))
        if name == "tool_evidence_notes":
            return {"notes": [{"source_id": row["source_id"], "text": row["text"]}
                              for row in prompt["evidence"]]}
        assert name == "tool_graphics"
        result = {"title": "Source findings", "prompt": "Draw labeled sample diagrams.",
                  "caption": "Samples measured 10 nm.", "source_ids": [identifier
                      for row in prompt["evidence"]["notes"] for identifier in row["source_ids"]]}
        return transform(result) if transform else result

    def image(prompt, path, **options):
        calls.append(("image", {"prompt": prompt, "path": path, "options": options}))
        if after_image:
            after_image()
        return image_response(options["image_output_format"])

    monkeypatch.setattr(service.api, "generate_structured", structured)
    monkeypatch.setattr(service.api, "generate_image_asset", image)
    return calls, documents


def test_generation_reads_all_chunks_and_creates_cited_brief_before_single_image(catalog, monkeypatch):
    calls, documents = fake_generation(monkeypatch, catalog)
    events = list(service.generate_tool("work", "graphics", documents,
                                       image_options={"image_size": "1024x1024", "image_output_format": "webp"}))
    result = events[-1]["result"]
    assert result["coverage"] == {"documents": 2, "chunks": 14, "processed_chunks": 14}
    assert set(result["content"]["source_ids"]) == {source["id"] for source in result["sources"]}
    assert [name for name, _ in calls] == ["tool_evidence_notes", "tool_evidence_notes", "tool_graphics", "image"]
    assert result["settings"]["model_requests"] == 4
    assert calls[-1][1]["path"] is None and calls[-1][1]["options"]["image_size"] == "1024x1024"
    assert "Draw labeled sample diagrams." in calls[-1][1]["prompt"]
    assert result["image_data"]["output_format"] == "webp"


@pytest.mark.parametrize("damage", ["unknown", "missing_document", "long_prompt", "empty", "extra"])
def test_invalid_brief_never_starts_image_request(catalog, monkeypatch, damage):
    def damage_result(result):
        if damage == "unknown":
            result["source_ids"] = ["unknown"]
        elif damage == "missing_document":
            result["source_ids"] = ["s1"]
        elif damage == "long_prompt":
            result["prompt"] = "x" * 12001
        elif damage == "empty":
            result["caption"] = ""
        else:
            result["html"] = "<script>bad()</script>"
        return result

    calls, documents = fake_generation(monkeypatch, catalog, transform=damage_result)
    with pytest.raises(ValueError):
        list(service.generate_tool("work", "graphics", documents))
    assert calls[-1][0] == "tool_graphics"
    assert not any(name == "image" for name, _ in calls)


def test_cancellation_while_showing_image_progress_prevents_paid_image(catalog, monkeypatch):
    calls, documents = fake_generation(monkeypatch, catalog)
    cancelled = False
    generator = service.generate_tool("work", "graphics", documents, is_cancelled=lambda: cancelled)
    for event in generator:
        assert event["result"] is None
        if event["status"].startswith("Generating the graphic"):
            cancelled = True
            break
    with pytest.raises(service.ToolCancelled):
        next(generator)
    assert not any(name == "image" for name, _ in calls)


def test_source_change_at_image_progress_prevents_paid_image(catalog, monkeypatch):
    calls, documents = fake_generation(monkeypatch, catalog)
    generator = service.generate_tool("work", "graphics", documents)
    for event in generator:
        if event["status"].startswith("Generating the graphic"):
            with catalog[0]._db:
                catalog[0]._db.execute("UPDATE documents SET generation='replaced'")
            break
    with pytest.raises(ValueError, match="changed"):
        next(generator)
    assert not any(name == "image" for name, _ in calls)


@pytest.mark.parametrize("invalidate", ["cancel", "snapshot"])
def test_inflight_image_result_cannot_publish_after_cancel_or_source_change(catalog, monkeypatch, invalidate):
    stopped = False

    def after_image():
        nonlocal stopped
        if invalidate == "cancel":
            stopped = True
        else:
            with catalog[0]._db:
                catalog[0]._db.execute("UPDATE documents SET generation='replaced'")

    calls, documents = fake_generation(monkeypatch, catalog, after_image=after_image)
    events = []
    with pytest.raises((service.ToolCancelled, ValueError)):
        for event in service.generate_tool("work", "graphics", documents, is_cancelled=lambda: stopped):
            events.append(event)
    assert all(event["result"] is None for event in events) and calls[-1][0] == "image"


def test_image_call_counts_towards_preflight_budget(catalog, monkeypatch):
    calls, documents = fake_generation(monkeypatch, catalog)
    monkeypatch.setattr(service, "MAX_MODEL_CALLS", 3)
    with pytest.raises(ValueError, match="too many"):
        list(service.generate_tool("work", "graphics", documents))
    assert calls == []
