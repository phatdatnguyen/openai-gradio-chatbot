"""Current Flashcards/Quizzes persistence and safe, portable study exports."""

from copy import deepcopy
import csv
from io import StringIO
import json
from xml.etree import ElementTree

import pytest

from rag_chatbot.tools import artifacts
from rag_chatbot.tools.exports import export_bytes, learning_export_filename
from rag_chatbot.workspaces import create_workspace, workspace_path


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "data"))
    create_workspace("Research")
    return "Research"


@pytest.fixture(params=["flashcards", "quizzes"])
def payload(workspace, request):
    return _payload(workspace, request.param)


@pytest.fixture
def quiz_payload(workspace):
    return _payload(workspace, "quizzes")


def _payload(workspace, tool_id):
    content = ({"cards": [{"front": "What was measured?", "back": "A peak was measured.",
                           "source_ids": ["chunk-one"]}]}
               if tool_id == "flashcards" else
               {"questions": [{"question": "What was measured?", "options": ["Mass", "A peak", "Volume", "Time"],
                               "correct_option": 1, "explanation": "The source reports a measured peak.",
                               "source_ids": ["chunk-one"]}]})
    return {"tool_id": tool_id, "title": "Research study set", "model": "gpt-6-luna", "content": content,
            "sources": [{"id": "chunk-one", "document_id": "doc-one", "filename": "paper.pdf",
                         "locator": "Page 2", "text": "A peak was measured.",
                         "source_path": str(workspace_path(workspace) / "sources" / "paper.pdf")}],
            "documents": [{"id": "doc-one", "name": "paper.pdf", "fingerprint": "v1", "generation": "g1"}],
            "coverage": {"documents": 1, "chunks": 1, "processed_chunks": 1}, "settings": {"instruction": "Study"}}


def first_item(payload):
    return payload["content"]["cards" if payload["tool_id"] == "flashcards" else "questions"][0]


def test_learning_current_replaces_with_stable_identity_and_isolated_workspace(workspace, payload):
    tool_id = payload["tool_id"]
    assert artifacts.load_current_learning(workspace, tool_id) is None
    assert not (workspace_path(workspace) / "artifacts").exists()
    original = deepcopy(payload)
    first = artifacts.save_current_learning(workspace, payload)
    second = artifacts.save_current_learning(workspace, {**payload, "title": "New study set"})
    assert payload == original
    assert second["id"] == first["id"] and second["created_at"] == first["created_at"]
    assert second["updated_at"] >= first["updated_at"]
    assert second["sources"][0]["source_path"] == "sources/paper.pdf"
    assert artifacts.load_current_learning(workspace, tool_id) == second
    assert artifacts.list_artifacts(workspace) == []
    directory = workspace_path(workspace) / "artifacts"
    assert [path.name for path in directory.iterdir()] == [tool_id]
    assert [path.name for path in (directory / tool_id).iterdir()] == ["current.json"]
    other_tool = "flashcards" if tool_id == "quizzes" else "quizzes"
    assert artifacts.load_current_learning(workspace, other_tool) is None
    create_workspace("Other")
    assert artifacts.load_current_learning("Other", tool_id) is None
    assert not (workspace_path("Other") / "artifacts").exists()


def test_learning_cancel_before_write_or_publish_preserves_previous_output(workspace, payload):
    with pytest.raises(RuntimeError, match="cancelled"):
        artifacts.save_current_learning(workspace, payload, is_cancelled=lambda: True)
    assert not (workspace_path(workspace) / "artifacts").exists()
    saved = artifacts.save_current_learning(workspace, payload)
    path = workspace_path(workspace) / "artifacts" / payload["tool_id"] / "current.json"
    before = path.read_bytes()
    checks = []

    def cancelled():
        checks.append(True)
        return len(checks) == 2

    with pytest.raises(RuntimeError, match="cancelled"):
        artifacts.save_current_learning(workspace, {**saved, "title": "Late output"}, is_cancelled=cancelled)
    assert path.read_bytes() == before
    assert [member.name for member in path.parent.iterdir()] == ["current.json"]
    with pytest.raises(ValueError, match="callable"):
        artifacts.save_current_learning(workspace, payload, is_cancelled=True)


@pytest.mark.parametrize("tool_id", [None, [], "summary", "../quizzes", "Quiz"])
def test_learning_operations_reject_other_tool_ids_without_creating_paths(workspace, tool_id):
    with pytest.raises(ValueError):
        artifacts.load_current_learning(workspace, tool_id)
    with pytest.raises(ValueError):
        artifacts.save_current_learning(workspace, {"tool_id": tool_id})
    with pytest.raises(ValueError):
        artifacts.export_learning(workspace, {"tool_id": tool_id}, "json", "Study")
    assert not (workspace_path(workspace) / "artifacts").exists()


@pytest.mark.parametrize("damage", ["citations", "fields", "empty", "blank", "duplicate"])
def test_learning_invalid_saved_content_rejected_before_replacement(workspace, payload, damage):
    saved = artifacts.save_current_learning(workspace, payload)
    path = workspace_path(workspace) / "artifacts" / payload["tool_id"] / "current.json"
    before = path.read_bytes()
    bad = deepcopy(payload)
    item = first_item(bad)
    key = "cards" if bad["tool_id"] == "flashcards" else "questions"
    if damage == "citations":
        item["source_ids"] = ["invented"]
    elif damage == "fields":
        item["html"] = "<script>bad()</script>"
    elif damage == "empty":
        bad["content"][key] = []
    elif damage == "blank":
        item["front" if bad["tool_id"] == "flashcards" else "question"] = " "
    else:
        bad["content"][key].append(deepcopy(item))
    with pytest.raises(ValueError):
        artifacts.save_current_learning(workspace, bad)
    assert path.read_bytes() == before
    saved["content"] = bad["content"]
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError):
        artifacts.load_current_learning(workspace, payload["tool_id"])


def test_learning_read_rejects_corrupt_json_and_mismatched_tool(workspace, payload):
    saved = artifacts.save_current_learning(workspace, payload)
    path = workspace_path(workspace) / "artifacts" / payload["tool_id"] / "current.json"
    path.write_text('{"tool_id":', encoding="utf-8")
    with pytest.raises(ValueError, match="JSON"):
        artifacts.load_current_learning(workspace, payload["tool_id"])
    other_tool = "flashcards" if payload["tool_id"] == "quizzes" else "quizzes"
    other_dir = path.parent.parent / other_tool
    other_dir.mkdir()
    (other_dir / "current.json").write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="different tool"):
        artifacts.load_current_learning(workspace, other_tool)


@pytest.mark.parametrize("linked_component", ["tool", "manifest", "exports", "export_file"])
def test_learning_rejects_linked_storage_paths(workspace, payload, tmp_path, linked_component):
    directory = workspace_path(workspace) / "artifacts"
    directory.mkdir()
    tool_directory = directory / payload["tool_id"]
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_text("untouched", encoding="utf-8")
    if linked_component == "tool":
        link, target, is_dir = tool_directory, outside, True
    else:
        tool_directory.mkdir()
        if linked_component == "manifest":
            link, target, is_dir = tool_directory / "current.json", sentinel, False
        elif linked_component == "exports":
            link, target, is_dir = tool_directory / "exports", outside, True
        else:
            (tool_directory / "exports").mkdir()
            link, target, is_dir = tool_directory / "exports" / "Study.json", sentinel, False
    try:
        link.symlink_to(target, target_is_directory=is_dir)
    except OSError:
        pytest.skip("Creating symlinks requires privileges on this Windows installation.")
    if linked_component in ("tool", "manifest"):
        with pytest.raises(ValueError, match="linked"):
            artifacts.load_current_learning(workspace, payload["tool_id"])
        with pytest.raises(ValueError, match="linked"):
            artifacts.save_current_learning(workspace, payload)
    if linked_component != "manifest":
        with pytest.raises(ValueError, match="linked"):
            artifacts.export_learning(workspace, payload, "json", "Study")
    assert sentinel.read_text(encoding="utf-8") == "untouched"


def test_learning_exports_keep_saved_output_and_include_full_provenance(workspace, payload):
    output = artifacts.export_learning(workspace, payload, "markdown", "Study.CSV")
    assert output.name == "Study.md"
    assert artifacts.load_current_learning(workspace, payload["tool_id"]) is None
    current = artifacts.save_current_learning(workspace, payload)
    path = output.parent.parent / "current.json"
    before = path.read_bytes()
    text = output.read_text(encoding="utf-8")
    assert "[1]" in text and "Page 2" in text and "A peak was measured" in text
    assert "## Sources" in text and "## Provenance" in text and "fingerprint" not in text
    assert "Model: gpt\\-6\\-luna" in text and '"processed\\_chunks": 1' in text
    export = artifacts.export_learning(workspace, current, "json", "Kết quả.markdown")
    assert export.name == "Kết quả.json"
    assert json.loads(export.read_text(encoding="utf-8")) == current
    csv_output = artifacts.export_learning(workspace, current, "csv", "Study.json")
    rows = list(csv.reader(StringIO(csv_output.read_text(encoding="utf-8-sig"))))
    if payload["tool_id"] == "flashcards":
        assert rows[0] == ["Front", "Back", "References"]
        assert rows[1] == ["What was measured?", "A peak was measured.", "[1]"]
        assert "**Back:** A peak was measured\\." in text
    else:
        assert rows[0] == ["Question", "A", "B", "C", "D", "Correct answer", "Explanation", "References"]
        assert rows[1][1:6] == ["Mass", "A peak", "Volume", "Time", "B"]
        assert rows[1][-1] == "[1]"
        assert "**Correct answer:** B — A peak" in text
    assert ["Sources"] in rows and ["Provenance"] in rows
    assert ["1", "paper.pdf", "Page 2", "A peak was measured."] in rows
    assert ["Model: gpt-6-luna"] in rows
    assert path.read_bytes() == before and artifacts.list_artifacts(workspace) == []


@pytest.mark.parametrize("formula", ["=SUM(1,2)", "+1", "-2", "@SUM(A1)", " \t=1", "\tvalue", "\nvalue"])
def test_learning_csv_escapes_all_untrusted_fields_without_changing_payload(payload, formula):
    item = first_item(payload)
    if payload["tool_id"] == "flashcards":
        item.update(front=formula, back=formula)
    else:
        item.update(question=formula, explanation=formula, options=[formula, "Second", "Third", "Fourth"])
    payload["sources"][0].update(filename=formula, locator=formula, text=formula)
    original = deepcopy(payload)
    extension, data = export_bytes(payload, "csv")
    assert extension == "csv"
    rows = list(csv.reader(StringIO(data.decode("utf-8-sig"))))
    assert rows[1][0] == "'" + formula
    assert rows[1][1] == "'" + formula
    source_row = rows[rows.index(["Citation", "Source document", "Location", "Evidence"]) + 1]
    assert source_row == ["1", "'" + formula, "'" + formula, "'" + formula]
    assert payload == original


def test_learning_markdown_escapes_markup_in_content_sources_and_provenance(payload):
    attack = '<script>alert("x")</script> [link](javascript:bad) &lt;img&gt;\n# injected\n- item'
    payload["title"] = attack
    payload["model"] = attack
    payload["settings"] = {"instruction": attack}
    payload["sources"][0].update(filename=attack, locator=attack, text=attack)
    item = first_item(payload)
    if payload["tool_id"] == "flashcards":
        item.update(front=attack, back=attack)
    else:
        item.update(question=attack, explanation=attack, options=[attack, "Two", "Three", "Four"])
    _, data = export_bytes(payload, "markdown")
    text = data.decode("utf-8")
    assert "<script>" not in text and "[link](javascript:" not in text
    assert "&lt;script&gt;" in text and "&amp;lt;img&amp;gt;" in text
    assert "\\[link\\]\\(javascript:bad\\)" in text
    assert "\n# injected" not in text and "\n- item" not in text
    assert "**References:** [1]" in text


@pytest.mark.parametrize("name", ["../outside", "sub/map", "sub\\map", "CON.csv", "NUL", "name.", ".hidden",
                                  "a" * 101, "界" * 85, " Study", "Study ", "a\x00b", "\ud800"])
def test_learning_export_rejects_nonportable_names_without_storage(workspace, payload, name):
    with pytest.raises(ValueError, match="filename"):
        artifacts.export_learning(workspace, payload, "csv", name)
    assert not (workspace_path(workspace) / "artifacts").exists()


def test_learning_exports_reject_unavailable_formats(workspace, payload):
    with pytest.raises(ValueError, match="available export"):
        artifacts.export_learning(workspace, payload, "html", "Study")
    with pytest.raises(ValueError, match="Unsupported"):
        learning_export_filename("Study", "html")
    assert not (workspace_path(workspace) / "artifacts").exists()


def test_quiz_aiken_contains_only_question_blocks_with_all_correct_answer_letters(quiz_payload):
    questions = quiz_payload["content"]["questions"]
    first = questions.pop()
    expected = []
    for index, letter in enumerate("ABCD"):
        questions.append({**deepcopy(first), "question": f"Question {index + 1}?", "correct_option": index})
        expected.append(f"Question {index + 1}?\nA. Mass\nB. A peak\nC. Volume\nD. Time\nANSWER: {letter}")
    before = deepcopy(quiz_payload)
    extension, data = export_bytes(quiz_payload, "aiken")
    assert extension == "txt"
    assert data.decode("utf-8") == "\n\n".join(expected) + "\n"
    assert not data.startswith(b"\xef\xbb\xbf")
    assert "References" not in data.decode("utf-8") and "Research study set" not in data.decode("utf-8")
    assert quiz_payload == before


@pytest.mark.parametrize("prompt,expected", [
    ("A. Is this an option?", "Question: A. Is this an option?"),
    ("Z)Not an option", "Question: Z)Not an option"),
    ("b.Also a prompt", "Question: b.Also a prompt"),
    ("ANSWER: Explain this term", "Question: ANSWER: Explain this term"),
    ("answer: lowercase syntax", "Question: answer: lowercase syntax"),
    ("X", "Question: X"),
    ("Đo θ?", "Đo θ?"),
])
def test_quiz_aiken_disambiguates_question_lines_for_import(quiz_payload, prompt, expected):
    quiz_payload["content"]["questions"][0]["question"] = prompt
    extension, data = export_bytes(quiz_payload, "aiken")
    assert data.decode("utf-8").splitlines()[0] == expected


def test_quiz_aiken_flattens_all_line_separators_without_changing_content(quiz_payload):
    question = quiz_payload["content"]["questions"][0]
    question.update(question="  Đo θ?\r\nHow\tlarge?\u0085Next\u2028line\u2029  ",
                    options=["<img src=x>\r\nANSWER: D", "\\(λ = 2d sin θ\\)", "\tThird\u2028choice", "Fourth\u2029choice"])
    before = deepcopy(quiz_payload)
    _, data = export_bytes(quiz_payload, "aiken")
    assert data.decode("utf-8") == (
        "Đo θ? How large? Next line\nA. <img src=x> ANSWER: D\nB. \\(λ = 2d sin θ\\)\n"
        "C. Third choice\nD. Fourth choice\nANSWER: B\n")
    assert quiz_payload == before


@pytest.mark.parametrize("correct_option", range(4))
def test_quiz_moodle_xml_is_plain_text_and_preserves_answers_and_cited_feedback(quiz_payload, correct_option):
    question = quiz_payload["content"]["questions"][0]
    question["correct_option"] = correct_option
    before = deepcopy(quiz_payload)
    extension, data = export_bytes(quiz_payload, "moodle_xml")
    assert extension == "xml" and data.startswith(b"<?xml version='1.0' encoding='utf-8'?>\n")
    root = ElementTree.fromstring(data)
    assert root.tag == "quiz" and len(root) == 1
    item = root[0]
    assert item.attrib == {"type": "multichoice"}
    assert item.findtext("name/text") == "Research study set — Question 1"
    assert item.findtext("questiontext/text") == question["question"]
    assert item.find("questiontext").attrib == {"format": "plain_text"}
    assert item.find("generalfeedback").attrib == {"format": "plain_text"}
    assert item.findtext("generalfeedback/text") == (
        "The source reports a measured peak.\n\nReferences:\n\n[1] paper.pdf — Page 2\nA peak was measured.")
    assert item.findtext("single") == "true"
    assert item.findtext("shuffleanswers") == "0"
    assert item.findtext("answernumbering") == "ABCD"
    assert item.findtext("defaultgrade") == "1" and item.findtext("penalty") == "0"
    answers = item.findall("answer")
    assert [answer.findtext("text") for answer in answers] == question["options"]
    assert [answer.attrib for answer in answers] == [
        {"fraction": "100" if index == correct_option else "0", "format": "plain_text"} for index in range(4)]
    assert quiz_payload == before


def test_quiz_moodle_xml_escapes_markup_and_retains_unicode_math_and_multiline_text(quiz_payload):
    attack = '<script>alert("x")</script> &lt;img&gt; & \\(λ < θ\\)\n]]> <answer fraction="100">'
    quiz_payload["title"] = attack
    quiz_payload["sources"][0].update(filename=attack, locator=attack, text=attack)
    question = quiz_payload["content"]["questions"][0]
    question.update(question=attack, explanation=attack, options=[attack, "Two", "Three", "Four"])
    before = deepcopy(quiz_payload)
    _, data = export_bytes(quiz_payload, "moodle_xml")
    root = ElementTree.fromstring(data)
    assert root.findtext("question/questiontext/text") == attack
    assert root.findtext("question/answer/text") == attack
    assert root.findtext("question/generalfeedback/text") == f"{attack}\n\nReferences:\n\n[1] {attack} — {attack}\n{attack}"
    assert not root.findall(".//script") and len(root.findall(".//answer")) == 4
    assert b"&lt;script&gt;" in data and b"&amp;lt;img&amp;gt;" in data
    assert quiz_payload == before


@pytest.mark.parametrize("format,extension", [("aiken", "txt"), ("moodle_xml", "xml")])
def test_quiz_import_exports_use_portable_scoped_names_without_replacing_current(workspace, quiz_payload, format, extension):
    current = artifacts.save_current_learning(workspace, quiz_payload)
    path = workspace_path(workspace) / "artifacts" / "quizzes" / "current.json"
    before = path.read_bytes()
    output = artifacts.export_learning(workspace, current, format, "Bài kiểm tra.TXT")
    assert output.name == f"Bài kiểm tra.{extension}"
    assert output.parent == path.parent / "exports"
    assert output.read_bytes() == export_bytes(current, format)[1]
    assert path.read_bytes() == before
    assert artifacts.list_artifacts(workspace) == []
    assert learning_export_filename("Quiz.XML", extension) == f"Quiz.{extension}"


@pytest.mark.parametrize("tool_id", ["flashcards", "summary", "spreadsheet", "mind_maps"])
@pytest.mark.parametrize("format", ["aiken", "moodle_xml"])
def test_quiz_import_formats_rejected_for_other_tools(quiz_payload, tool_id, format):
    with pytest.raises(ValueError, match="available export format"):
        export_bytes({**quiz_payload, "tool_id": tool_id}, format)


@pytest.mark.parametrize("format", ["aiken", "moodle_xml"])
@pytest.mark.parametrize("name", ["../outside", "sub/quiz", "sub\\quiz", "CON.txt", "NUL.xml", "name.",
                                  "界" * 85, "a\x00b", "\ud800"])
def test_quiz_import_export_rejects_unsafe_names_without_storage(workspace, quiz_payload, format, name):
    with pytest.raises(ValueError, match="filename"):
        artifacts.export_learning(workspace, quiz_payload, format, name)
    assert not (workspace_path(workspace) / "artifacts").exists()


@pytest.mark.parametrize("format", ["aiken", "moodle_xml"])
@pytest.mark.parametrize("damage", ["blank_prompt", "blank_option", "answer", "citation", "control"])
def test_quiz_import_export_rejects_invalid_quiz_content(workspace, quiz_payload, format, damage):
    question = quiz_payload["content"]["questions"][0]
    if damage == "blank_prompt":
        question["question"] = " \n"
    elif damage == "blank_option":
        question["options"][1] = " \n"
    elif damage == "answer":
        question["correct_option"] = 4
    elif damage == "citation":
        question["source_ids"] = ["invented"]
    else:
        question["question"] = "Null\x00character"
    with pytest.raises(ValueError):
        artifacts.export_learning(workspace, quiz_payload, format, "Quiz")
    assert not (workspace_path(workspace) / "artifacts").exists()


def test_quiz_moodle_rejects_invalid_xml_in_reference_metadata(workspace, quiz_payload):
    quiz_payload["sources"][0]["text"] = "Invalid\x00control"
    with pytest.raises(ValueError, match="Moodle XML"):
        artifacts.export_learning(workspace, quiz_payload, "moodle_xml", "Quiz")
    assert not (workspace_path(workspace) / "artifacts").exists()
