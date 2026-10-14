"""Offline cache, containment and Office ownership tests; no real Office is launched."""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile
from xml.etree import ElementTree

import pytest
from pypdf import PdfWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_chatbot import office_preview as preview


def write_pdf(path):
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=400)
    with Path(path).open("wb") as handle:
        writer.write(handle)


def office_file(path, *, relationship=None, field=None):
    if path.suffix in {".docx", ".pptx"}:
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("[Content_Types].xml", "<Types/>")
            if relationship:
                archive.writestr("_rels/.rels", '<Relationships><Relationship TargetMode="External" '
                    f'Type="http://example.org/{relationship}" Target="https://example.org/"/></Relationships>')
            if field:
                archive.writestr("word/document.xml", '<document xmlns:w="urn:word">'
                    f'{field}</document>')
    else:
        path.write_bytes(b"synthetic legacy Office fixture")
    return path


@pytest.mark.parametrize("extension", [".doc", ".docx", ".ppt", ".pptx"])
def test_valid_pdf_is_cached_and_original_is_never_modified(tmp_path, monkeypatch, extension):
    original = office_file(tmp_path / f"original{extension}")
    before = original.read_bytes()
    calls = []

    def convert(source, output, report):
        calls.append(source)
        write_pdf(output)

    monkeypatch.setattr(preview, "_run_worker", convert)
    first = preview.get_pdf_preview(original, data_dir=tmp_path)
    assert first.parent == tmp_path / ".preview"
    assert preview.get_pdf_preview(original, data_dir=tmp_path) == first
    assert calls == [original] and original.read_bytes() == before
    assert not list(first.parent.glob(".converting-*"))


def test_content_change_invalidates_cache_even_when_size_and_mtime_match(tmp_path, monkeypatch):
    import os

    original = office_file(tmp_path / "original.doc")
    monkeypatch.setattr(preview, "_run_worker", lambda source, output, report: write_pdf(output))
    first = preview.get_pdf_preview(original, data_dir=tmp_path)
    initial = original.stat()
    original.write_bytes(b"X" * initial.st_size)
    os.utime(original, ns=(initial.st_atime_ns, initial.st_mtime_ns))
    second = preview.get_pdf_preview(original, data_dir=tmp_path)
    assert first != second and first.is_file() and second.is_file()


def test_failed_conversion_does_not_publish_partial_pdf_or_remove_previous_cache(tmp_path, monkeypatch):
    original = office_file(tmp_path / "original.doc")
    monkeypatch.setattr(preview, "_run_worker", lambda source, output, report: write_pdf(output))
    previous = preview.get_pdf_preview(original, data_dir=tmp_path)
    original.write_bytes(b"changed document")
    monkeypatch.setattr(preview, "_run_worker", lambda source, output, report: output.write_bytes(b"partial"))
    with pytest.raises(preview.PreviewUnavailable, match="valid PDF"):
        preview.get_pdf_preview(original, data_dir=tmp_path)
    assert list(previous.parent.glob("*.pdf")) == [previous]
    assert not list(previous.parent.glob(".converting-*"))


def test_source_changes_during_conversion_do_not_publish_cache(tmp_path, monkeypatch):
    original = office_file(tmp_path / "original.doc")

    def convert(source, output, report):
        write_pdf(output)
        source.write_bytes(b"changed during conversion")

    monkeypatch.setattr(preview, "_run_worker", convert)
    with pytest.raises(preview.PreviewUnavailable, match="changed while"):
        preview.get_pdf_preview(original, data_dir=tmp_path)
    assert not list((tmp_path / ".preview").glob("*.pdf"))


def test_paths_and_file_size_are_checked_before_starting_worker(tmp_path, monkeypatch):
    original = office_file(tmp_path / "original.doc")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(preview, "_run_worker", lambda *args: pytest.fail("Unexpected conversion"))
    with pytest.raises(preview.PreviewUnavailable, match="outside"):
        preview.get_pdf_preview(original, data_dir=workspace)
    with pytest.raises(preview.PreviewUnavailable, match="outside"):
        preview.get_pdf_preview("../original.doc", data_dir=workspace)
    monkeypatch.setattr(preview, "MAX_SOURCE_BYTES", 1)
    with pytest.raises(preview.PreviewUnavailable, match="100 MiB"):
        preview.get_pdf_preview(original, data_dir=tmp_path)


def test_cache_link_is_rejected_even_when_it_targets_inside_workspace(tmp_path, monkeypatch):
    original = office_file(tmp_path / "original.doc")
    cache = tmp_path / ".preview"
    cache.mkdir()
    original_is_symlink = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == cache or original_is_symlink(path))
    with pytest.raises(preview.PreviewUnavailable, match="symbolic links"):
        preview.get_pdf_preview(original, data_dir=tmp_path)


@pytest.mark.parametrize("relationship", ["image", "attachedTemplate", "oleObject"])
def test_external_relationships_are_rejected_before_office_opens(tmp_path, monkeypatch, relationship):
    original = office_file(tmp_path / "original.docx", relationship=relationship)
    monkeypatch.setattr(preview, "_run_worker", lambda *args: pytest.fail("Unexpected conversion"))
    with pytest.raises(preview.PreviewUnavailable, match="external linked content"):
        preview.get_pdf_preview(original, data_dir=tmp_path)


@pytest.mark.parametrize("field", [
    '<w:instrText>DDEAUTO cmd</w:instrText>',
    '<w:instrText>INCLUDE</w:instrText><w:instrText>TEXT remote</w:instrText>',
    '<w:fldSimple w:instr="INCLUDEPICTURE remote"/>',
    '<w:fldChar w:fldCharType="begin"/><w:instrText>PAGE</w:instrText><w:fldChar w:fldCharType="end"/>'
    '<w:fldChar w:fldCharType="begin"/><w:instrText>INCLUDE</w:instrText><w:instrText>TEXT remote</w:instrText>'
    '<w:fldChar w:fldCharType="end"/>',
])
def test_active_word_fields_are_rejected(tmp_path, monkeypatch, field):
    original = office_file(tmp_path / "original.docx", field=field)
    monkeypatch.setattr(preview, "_run_worker", lambda *args: pytest.fail("Unexpected conversion"))
    with pytest.raises(preview.PreviewUnavailable, match="active fields"):
        preview.get_pdf_preview(original, data_dir=tmp_path)


def test_plain_hyperlinks_do_not_prevent_static_preview(tmp_path, monkeypatch):
    original = office_file(tmp_path / "original.docx", relationship="hyperlink")
    monkeypatch.setattr(preview, "_run_worker", lambda source, output, report: write_pdf(output))
    assert preview.get_pdf_preview(original, data_dir=tmp_path).is_file()


def test_worker_timeout_is_bounded_and_hidden_without_shell(tmp_path, monkeypatch):
    calls = []

    def timeout(command, **kwargs):
        calls.append((command, kwargs))
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(preview.subprocess, "run", timeout)
    monkeypatch.setattr(preview, "os", SimpleNamespace(name="nt"))
    with pytest.raises(preview.PreviewUnavailable, match="timed out"):
        preview._run_worker(tmp_path / "original.docx", tmp_path / "out.pdf", tmp_path / "result.json")
    command, options = calls[0]
    assert command[:4] == [sys.executable, "-m", "rag_chatbot.office_preview", "--worker"]
    assert options["timeout"] == preview.CONVERSION_TIMEOUT
    assert not options.get("shell")


def test_worker_failure_preserves_actionable_reason(tmp_path, monkeypatch):
    report = tmp_path / "result.json"
    report.write_text(json.dumps({"error": "Close PowerPoint before preparing this preview."}), encoding="utf-8")
    monkeypatch.setattr(preview.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1))
    monkeypatch.setattr(preview, "os", SimpleNamespace(name="nt"))
    with pytest.raises(preview.PreviewUnavailable, match="Close PowerPoint"):
        preview._run_worker(tmp_path / "original.ppt", tmp_path / "out.pdf", report)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows process job binding")
def test_windows_job_binding_creates_an_empty_kill_on_close_job():
    win32job = pytest.importorskip("win32job")
    job = preview._new_process_job()
    try:
        limits = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
        assert limits["BasicLimitInformation"]["LimitFlags"] & win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        processes = win32job.QueryInformationJobObject(job, win32job.JobObjectBasicProcessIdList)
        assert not processes
    finally:
        job.Close()


@pytest.fixture
def com(monkeypatch):
    class ComError(Exception):
        pass

    events = []
    app = SimpleNamespace(HWND=20, Documents=SimpleNamespace(Count=0), Presentations=SimpleNamespace(Count=0),
                          Quit=lambda: events.append("quit"))
    client = SimpleNamespace(DispatchEx=lambda progid: events.append("dispatch") or app)
    monkeypatch.setattr(preview, "os", SimpleNamespace(name="nt"))
    monkeypatch.setitem(sys.modules, "pythoncom", SimpleNamespace(CoInitialize=lambda: events.append("init"),
        CoUninitialize=lambda: events.append("uninit")))
    monkeypatch.setitem(sys.modules, "pywintypes", SimpleNamespace(error=ComError))
    monkeypatch.setitem(sys.modules, "win32com", SimpleNamespace(client=client))
    monkeypatch.setitem(sys.modules, "win32com.client", client)
    monkeypatch.setitem(sys.modules, "win32process", SimpleNamespace(
        EnumProcesses=lambda: [10], GetWindowThreadProcessId=lambda hwnd: (0, 20)))
    snapshots = iter(({10}, {10, 20}))
    monkeypatch.setattr(preview, "_office_pids", lambda executable: next(snapshots))
    monkeypatch.setattr(preview, "_own_process_job", lambda pid: events.append(("job", pid)) or
                        SimpleNamespace(Close=lambda: events.append("job closed")))
    monkeypatch.setattr(preview, "_export_word", lambda *args: events.append("export"))
    monkeypatch.setattr(preview, "_export_powerpoint", lambda *args: events.append("export"))
    return app, events


def test_owned_word_instance_is_exported_and_cleaned_up(com, tmp_path):
    app, events = com
    preview._convert_with_office(tmp_path / "file.doc", tmp_path / "file.pdf")
    assert events == ["init", "dispatch", ("job", 20), "export", "quit", "job closed", "uninit"]


def test_existing_powerpoint_is_never_dispatched_or_quit(com, tmp_path):
    app, events = com
    with pytest.raises(preview.PreviewUnavailable, match="Close PowerPoint"):
        preview._convert_with_office(tmp_path / "file.ppt", tmp_path / "file.pdf")
    assert events == []


def test_unowned_office_instance_is_never_modified_or_quit(com, tmp_path, monkeypatch):
    app, events = com
    monkeypatch.setattr(preview, "_office_pids", lambda executable: {10, 20})
    with pytest.raises(preview.PreviewUnavailable, match="separate empty"):
        preview._convert_with_office(tmp_path / "file.doc", tmp_path / "file.pdf")
    assert events == ["init", "dispatch", "uninit"]


def test_owned_office_is_cleaned_up_after_export_failure(com, tmp_path, monkeypatch):
    app, events = com

    def failure(*args):
        raise sys.modules["pywintypes"].error("simulated COM error")

    monkeypatch.setattr(preview, "_export_word", failure)
    with pytest.raises(preview.PreviewUnavailable, match="Microsoft Office could not preview"):
        preview._convert_with_office(tmp_path / "file.doc", tmp_path / "file.pdf")
    assert events[-3:] == ["quit", "job closed", "uninit"]


def test_word_opens_read_only_disables_macros_links_and_restores_settings(tmp_path):
    events = []
    options = SimpleNamespace(UpdateLinksAtOpen=True, UpdateLinksAtPrint=True, UpdateFieldsAtPrint=True)
    document = SimpleNamespace(ExportAsFixedFormat=lambda *args, **kwargs: events.append("export"),
                               Close=lambda **kwargs: events.append(("close", kwargs)))
    app = SimpleNamespace(Options=options)

    def open_document(*args, **kwargs):
        assert app.AutomationSecurity == 3 and app.DisplayAlerts == 0 and app.Visible is False
        assert not any(vars(options).values())
        assert kwargs["ReadOnly"] and not kwargs["Visible"] and not kwargs["AddToRecentFiles"]
        return document

    app.Documents = SimpleNamespace(Open=open_document)
    preview._export_word(app, tmp_path / "file.docx", tmp_path / "file.pdf")
    assert all(vars(options).values())
    assert events == ["export", ("close", {"SaveChanges": 0})]


def test_powerpoint_opens_read_only_without_window_or_macro_execution(tmp_path):
    events = []
    presentation = SimpleNamespace(
        ExportAsFixedFormat=lambda *args, **kwargs: events.append(("export", kwargs)),
        Close=lambda: events.append("close"))
    app = SimpleNamespace()

    def open_presentation(*args, **kwargs):
        assert app.AutomationSecurity == 3 and app.DisplayAlerts == 1
        assert kwargs == {"ReadOnly": -1, "Untitled": 0, "WithWindow": 0}
        return presentation

    app.Presentations = SimpleNamespace(Open=open_presentation)
    preview._export_powerpoint(app, tmp_path / "file.pptx", tmp_path / "file.pdf")
    assert events == [("export", {"PrintHiddenSlides": -1, "PrintRange": None}), "close"]


def test_process_check_permission_failure_is_actionable_without_command_text(monkeypatch):
    def denied(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "tasklist private arguments")

    monkeypatch.setattr(preview.subprocess, "run", denied)
    with pytest.raises(preview.PreviewUnavailable, match="current permissions") as error:
        preview._office_pids("WINWORD.EXE")
    assert "tasklist" not in str(error.value)


@pytest.mark.parametrize("platform_name, expected", [("nt", "microsoft"), ("posix", "libreoffice")])
def test_worker_routes_to_platform_converter(tmp_path, monkeypatch, platform_name, expected):
    calls = []
    monkeypatch.setattr(preview, "os", SimpleNamespace(name=platform_name))
    monkeypatch.setattr(sys, "argv", ["office_preview", "--worker", str(tmp_path / "source.docx"),
                                     str(tmp_path / "output.pdf"), str(tmp_path / "report.json")])
    monkeypatch.setattr(preview, "_convert_with_office", lambda *args: calls.append("microsoft"))
    monkeypatch.setattr(preview, "_convert_with_libreoffice", lambda *args: calls.append("libreoffice"))
    assert preview._main() == 0
    assert calls == [expected]


def test_libreoffice_configured_executable_is_one_argument_and_has_priority(tmp_path, monkeypatch):
    executable = tmp_path / "Custom Office" / "soffice"
    monkeypatch.setenv("LIBREOFFICE_PATH", str(executable))
    calls = []
    monkeypatch.setattr(preview.shutil, "which", lambda name: calls.append(name) or str(executable))
    assert preview._libreoffice_executable() == str(executable)
    assert calls == [str(executable)]
    monkeypatch.setattr(preview.shutil, "which", lambda name: None)
    with pytest.raises(preview.PreviewUnavailable, match="without arguments"):
        preview._libreoffice_executable()


def test_libreoffice_is_discovered_on_path_or_reports_optional_dependency(monkeypatch):
    monkeypatch.delenv("LIBREOFFICE_PATH", raising=False)
    monkeypatch.setattr(preview.sys, "platform", "linux")
    monkeypatch.setattr(preview.shutil, "which", lambda name: "/usr/bin/libreoffice" if name == "libreoffice" else None)
    assert preview._libreoffice_executable() == str(Path("/usr/bin/libreoffice").resolve())
    monkeypatch.setattr(preview.shutil, "which", lambda name: None)
    with pytest.raises(preview.PreviewUnavailable, match="Install LibreOffice"):
        preview._libreoffice_executable()


def test_macos_libreoffice_bundle_is_discovered_without_path_entry(tmp_path, monkeypatch):
    monkeypatch.delenv("LIBREOFFICE_PATH", raising=False)
    monkeypatch.setattr(preview.sys, "platform", "darwin")
    monkeypatch.setattr(preview.shutil, "which", lambda name: None)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    executable = tmp_path / "Applications/LibreOffice.app/Contents/MacOS/soffice"
    executable.parent.mkdir(parents=True)
    executable.write_text("fixture executable", encoding="utf-8")
    executable.chmod(0o700)
    # Do not accidentally pick an installed system application on a macOS runner.
    monkeypatch.setattr(Path, "is_file", lambda path: path == executable)
    assert preview._libreoffice_executable() == str(executable)


@pytest.mark.parametrize("extension, pdf_filter", [
    (".doc", "writer_pdf_Export"), (".docx", "writer_pdf_Export"),
    (".ppt", "impress_pdf_Export"), (".pptx", "impress_pdf_Export"),
])
def test_libreoffice_uses_disposable_source_secure_profile_and_pdf_filter(tmp_path, monkeypatch, extension, pdf_filter):
    source = office_file(tmp_path / f"original & source{extension}")
    initial = source.read_bytes(), source.stat().st_mtime_ns
    staging = tmp_path / "staging with spaces & symbols"
    staging.mkdir()
    output = staging / "preview.pdf"
    monkeypatch.setattr(preview, "_libreoffice_executable", lambda: "/opt/Office Suite/soffice")
    calls = []

    def convert(command, **options):
        calls.append((command, options))
        assert command[0] == "/opt/Office Suite/soffice"
        assert command[1] == f"-env:UserInstallation={(staging / 'libreoffice-profile').as_uri()}"
        assert command[2:6] == ["--headless", "--nologo", "--nodefault", "--norestore"]
        assert command[6:8] == ["--convert-to", f"pdf:{pdf_filter}"]
        assert command[8:10] == ["--outdir", str(staging / "libreoffice-output")]
        copied = Path(command[-1])
        assert copied != source and copied.read_bytes() == initial[0]
        settings = ElementTree.parse(staging / "libreoffice-profile/user/registrymodifications.xcu")
        namespace = "{http://openoffice.org/2001/registry}"
        values = {(item.get(namespace + "path"), prop.get(namespace + "name")): prop.findtext("value")
                  for item in settings.getroot() for prop in item}
        scripting = "/org.openoffice.Office.Common/Security/Scripting"
        assert values[scripting, "DisableMacrosExecution"] == "true"
        assert values[scripting, "DisableActiveContent"] == "true"
        assert values[scripting, "BlockUntrustedRefererLinks"] == "true"
        assert values[scripting, "MacroSecurityLevel"] == "3"
        assert values["/org.openoffice.Office.Writer/Content/Update", "Link"] == "2"
        assert values["/org.openoffice.Office.Writer/Content/Update", "Field"] == "false"
        copied.write_bytes(b"Simulate a converter saving its disposable copy")
        write_pdf(staging / "libreoffice-output/input.pdf")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(preview.subprocess, "run", convert)
    preview._convert_with_libreoffice(source, output)
    preview._validate_pdf(output)
    assert (source.read_bytes(), source.stat().st_mtime_ns) == initial
    assert len(calls) == 1
    options = calls[0][1]
    assert options["timeout"] < preview.CONVERSION_TIMEOUT
    assert options["stdin"] == options["stdout"] == options["stderr"] == subprocess.DEVNULL
    assert not options.get("shell")


@pytest.mark.parametrize("failure", ["timeout", "failed", "missing_pdf"])
def test_libreoffice_failures_do_not_publish_pdf(tmp_path, monkeypatch, failure):
    source = office_file(tmp_path / "original.doc")
    monkeypatch.setattr(preview, "_libreoffice_executable", lambda: "soffice")

    def convert(command, **options):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, options["timeout"])
        return SimpleNamespace(returncode=1 if failure == "failed" else 0)

    monkeypatch.setattr(preview.subprocess, "run", convert)
    output = tmp_path / "preview.pdf"
    with pytest.raises(preview.PreviewUnavailable, match="timed out" if failure == "timeout" else "could not preview"):
        preview._convert_with_libreoffice(source, output)
    assert not output.exists()


@pytest.mark.parametrize("timeout", [False, True])
def test_posix_worker_kills_only_its_new_process_group_even_after_exit(monkeypatch, timeout):
    calls = []

    class Process:
        pid = 12345
        returncode = 0

        def wait(self, **options):
            calls.append(("wait", options))
            if timeout and options["timeout"] == preview.CONVERSION_TIMEOUT:
                raise subprocess.TimeoutExpired("owned worker", options["timeout"])
            return self.returncode

    def popen(command, **options):
        calls.append(("start", options))
        return Process()

    monkeypatch.setattr(preview.subprocess, "Popen", popen)
    monkeypatch.setattr(preview, "os", SimpleNamespace(killpg=lambda pid, sig: calls.append(("kill group", pid, sig))))
    # SIGKILL is only available natively on POSIX, but the process lifecycle can
    # also be verified without launching or signaling anything on Windows.
    monkeypatch.setattr(preview, "signal", SimpleNamespace(SIGKILL=9))
    if timeout:
        with pytest.raises(subprocess.TimeoutExpired):
            preview._run_posix_worker(["owned worker"])
    else:
        assert preview._run_posix_worker(["owned worker"]).returncode == 0
    assert calls[0][1]["start_new_session"] is True
    assert not calls[0][1].get("shell")
    assert calls[-2:] == [("kill group", 12345, 9), ("wait", {"timeout": 5})]


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_real_posix_worker_timeout_is_bounded(monkeypatch):
    monkeypatch.setattr(preview, "CONVERSION_TIMEOUT", 0.2)
    with pytest.raises(subprocess.TimeoutExpired):
        preview._run_posix_worker([sys.executable, "-c", "import time; time.sleep(30)"])
