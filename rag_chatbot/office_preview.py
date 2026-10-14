"""Optional local Office-to-PDF previews; never edit the managed original.

Conversion runs in a bounded subprocess: Microsoft Office on Windows, optional
LibreOffice on Linux/macOS. Opening the app or a cached PDF starts neither suite.
"""

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import zipfile
from xml.etree import ElementTree

from chatbot.log import logger


OFFICE_EXTENSIONS = frozenset({".doc", ".docx", ".ppt", ".pptx"})
MAX_SOURCE_BYTES = 100 * 1024 * 1024
MAX_PDF_BYTES = 200 * 1024 * 1024
MAX_PREVIEW_PAGES = 2000
CONVERSION_TIMEOUT = 45
_CONVERSION_LOCK = threading.Lock()
_CACHE_VERSION = b"office-pdf-v1:"


class PreviewUnavailable(RuntimeError):
    """A document can still be downloaded when its visual preview is unavailable."""


def _managed_path(value, root):
    path = Path(value)
    path = Path(os.path.abspath(path if path.is_absolute() else root / path))
    if not path.is_relative_to(root):
        raise PreviewUnavailable("The preview source is outside this workspace.")
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        if current.exists() or current.is_symlink():
            attributes = getattr(current.lstat(), "st_file_attributes", 0)
            if current.is_symlink() or attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
                raise PreviewUnavailable("Preview paths cannot contain symbolic links or junctions.")
    if not path.resolve().is_relative_to(root):
        raise PreviewUnavailable("The preview path is outside this workspace.")
    return path


def _fingerprint(source):
    digest = hashlib.sha256(_CACHE_VERSION + source.suffix.lower().encode())
    total = 0
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            total += len(block)
            if total > MAX_SOURCE_BYTES:
                raise PreviewUnavailable("Office previews are limited to 100 MiB per document.")
            digest.update(block)
    return digest.hexdigest()


def _validate_pdf(path):
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    if not path.is_file() or not 0 < path.stat().st_size <= MAX_PDF_BYTES:
        raise PreviewUnavailable("Office did not produce a usable PDF preview.")
    try:
        with path.open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                raise PreviewUnavailable("Office did not produce a valid PDF preview.")
            handle.seek(0)
            document = PdfReader(handle)
            if document.is_encrypted or not 0 < len(document.pages) <= MAX_PREVIEW_PAGES:
                raise PreviewUnavailable("The PDF preview is encrypted, empty, or exceeds 2,000 pages.")
    except (PdfReadError, ValueError, OSError) as exc:
        raise PreviewUnavailable("Office produced an unreadable PDF preview.") from exc


def _check_external_content(source):
    # Linked images, templates and OLE objects may fetch external files on open.
    # Ordinary hyperlinks remain inert: the converter never follows hyperlinks.
    if source.suffix.lower() not in {".docx", ".pptx"}:
        return
    try:
        with zipfile.ZipFile(source) as archive:
            members = archive.infolist()
            if len(members) > 20000 or sum(item.file_size for item in members) > 500 * 1024 * 1024:
                raise PreviewUnavailable("The Office document is too large to preview safely.")
            for member in members:
                if source.suffix.lower() == ".docx" and member.filename.startswith("word/") and member.filename.endswith(".xml"):
                    if member.file_size > 20 * 1024 * 1024:
                        raise PreviewUnavailable("The Word document has an oversized XML part.")
                    document = ElementTree.fromstring(archive.read(member))
                    fields = []
                    complex_fields = []
                    fragments = []
                    for item in document.iter():
                        tag = item.tag.rsplit("}", 1)[-1]
                        if tag == "fldChar":
                            kind = next((value for name, value in item.attrib.items()
                                         if name.rsplit("}", 1)[-1] == "fldCharType"), "")
                            if kind == "begin":
                                complex_fields.append("")
                            elif kind == "end" and complex_fields:
                                fields.append(complex_fields.pop())
                        elif tag == "instrText":
                            text = item.text or ""
                            fragments.append(text)
                            if complex_fields:
                                complex_fields[-1] += text
                            else:
                                fields.append(text)
                        elif tag == "fldSimple":
                            fields.extend(value for name, value in item.attrib.items()
                                          if name.rsplit("}", 1)[-1] == "instr")
                    fields.extend(complex_fields)
                    fields.append("".join(fragments))
                    if any(re.search(r"\b(?:DDEAUTO|DDE|INCLUDETEXT|INCLUDEPICTURE|DATABASE|LINK)\b",
                                     field, re.IGNORECASE) for field in fields):
                        raise PreviewUnavailable("This document contains external or active fields. "
                                                 "Save a static PDF copy to preview it.")
                if not member.filename.endswith(".rels"):
                    continue
                if member.file_size > 4 * 1024 * 1024:
                    raise PreviewUnavailable("The Office document has an oversized relationship table.")
                root = ElementTree.fromstring(archive.read(member))
                if any(item.get("TargetMode", "").lower() == "external"
                       and not item.get("Type", "").endswith("/hyperlink") for item in root):
                    raise PreviewUnavailable("This document contains external linked content. "
                                             "Save a self-contained copy or a PDF to preview it.")
    except (zipfile.BadZipFile, ElementTree.ParseError) as exc:
        raise PreviewUnavailable("The Office document is damaged or password protected.") from exc


def _run_worker(source, output, report):
    command = [sys.executable, "-m", "rag_chatbot.office_preview", "--worker",
               str(source), str(output), str(report)]
    try:
        if os.name == "nt":
            result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.PIPE, text=True, timeout=CONVERSION_TIMEOUT,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:
            result = _run_posix_worker(command)
    except subprocess.TimeoutExpired as exc:
        raise PreviewUnavailable("Office preview timed out. Download the original or save it as PDF.") from exc
    except OSError as exc:
        raise PreviewUnavailable("The local Office preview worker could not start. Download the original or save it as PDF.") from exc
    if result.returncode:
        if getattr(result, "stderr", None):
            logger.warning("Local Office preview worker diagnostic: %s", result.stderr[-2000:])
        try:
            message = json.loads(report.read_text(encoding="utf-8")).get("error")
        except (OSError, ValueError):
            message = None
        raise PreviewUnavailable(message or "Office could not create this preview. Download the original or save it as PDF.")


def _run_posix_worker(command):
    # A fresh session owns this worker and its LibreOffice descendants. Killing
    # only the worker on timeout would leave soffice.bin and its profile behind.
    # Never discover/terminate a user's existing LibreOffice processes by name.
    with tempfile.TemporaryFile() as diagnostic:
        process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1],
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=diagnostic, start_new_session=True)
        try:
            process.wait(timeout=CONVERSION_TIMEOUT)
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            # Do not use Popen as a context manager here: its exit waits without
            # a timeout, even after a bounded wait fails during cleanup.
            process.wait(timeout=5)
        diagnostic.seek(max(0, diagnostic.seek(0, io.SEEK_END) - 2000))
        return subprocess.CompletedProcess(command, process.returncode,
            stderr=diagnostic.read().decode("utf-8", errors="replace"))


def _libreoffice_executable():
    configured = os.environ.get("LIBREOFFICE_PATH", "").strip()
    if configured:
        candidate = shutil.which(os.path.expanduser(configured))
        if not candidate:
            raise PreviewUnavailable("LIBREOFFICE_PATH must name an executable LibreOffice program, without arguments.")
        return str(Path(candidate).resolve())
    for name in ("soffice", "libreoffice"):
        candidate = shutil.which(name)
        if candidate:
            return str(Path(candidate).resolve())
    if sys.platform == "darwin":
        for directory in (Path("/Applications"), Path.home() / "Applications"):
            candidate = directory / "LibreOffice.app/Contents/MacOS/soffice"
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
    raise PreviewUnavailable("Install LibreOffice for Word/PowerPoint PDF previews, or set LIBREOFFICE_PATH "
                             "to its executable. You can also download the original or save it as PDF.")


def _write_libreoffice_profile(profile):
    # LibreOffice's registry schema documents these settings in Common.xcs and
    # Writer.xcs. Use a new profile, never the user's profile or trusted locations.
    namespace = "http://openoffice.org/2001/registry"
    ElementTree.register_namespace("oor", namespace)
    root = ElementTree.Element(f"{{{namespace}}}items")
    settings = {
        "/org.openoffice.Office.Common/Security/Scripting": {
            "DisableMacrosExecution": "true", "MacroSecurityLevel": "3",
            "DisableActiveContent": "true", "BlockUntrustedRefererLinks": "true",
        },
        "/org.openoffice.Office.Writer/Content/Update": {
            "Link": "2", "Field": "false", "Chart": "false",
        },
        "/org.openoffice.Office.Common/Load": {"UserDefinedSettings": "false"},
    }
    for path, values in settings.items():
        item = ElementTree.SubElement(root, "item", {f"{{{namespace}}}path": path})
        for name, value in values.items():
            prop = ElementTree.SubElement(item, "prop", {
                f"{{{namespace}}}name": name, f"{{{namespace}}}op": "fuse"})
            ElementTree.SubElement(prop, "value").text = value
    user = profile / "user"
    user.mkdir(parents=True)
    ElementTree.ElementTree(root).write(user / "registrymodifications.xcu",
                                        encoding="utf-8", xml_declaration=True)


def _convert_with_libreoffice(source, output):
    executable = _libreoffice_executable()
    profile = output.parent / "libreoffice-profile"
    _write_libreoffice_profile(profile)
    directory = output.parent / "libreoffice-output"
    directory.mkdir()
    # LibreOffice opens only a disposable copy. Even an unexpected save or lock
    # file cannot change the stable source retained for citations and downloads.
    copied = output.parent / f"input{source.suffix.lower()}"
    shutil.copyfile(source, copied)
    pdf_filter = "impress_pdf_Export" if source.suffix.lower() in {".ppt", ".pptx"} else "writer_pdf_Export"
    command = [executable, f"-env:UserInstallation={profile.resolve().as_uri()}",
               "--headless", "--nologo", "--nodefault", "--norestore",
               "--convert-to", f"pdf:{pdf_filter}", "--outdir", str(directory), str(copied)]
    try:
        result = subprocess.run(command, cwd=output.parent, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                timeout=max(1, CONVERSION_TIMEOUT - 5))
    except subprocess.TimeoutExpired as exc:
        raise PreviewUnavailable("LibreOffice preview timed out. Download the original or save it as PDF.") from exc
    generated = directory / "input.pdf"
    if result.returncode or not generated.is_file():
        raise PreviewUnavailable("LibreOffice could not preview this file. It may be protected, damaged, "
                                 "or unsupported; download it or save a PDF copy.")
    os.replace(generated, output)


def get_pdf_preview(source, *, data_dir):
    """Return a validated cached PDF below the original's managed .preview folder."""
    root = Path(data_dir).resolve()
    try:
        source = _managed_path(source, root)
        if source.suffix.lower() not in OFFICE_EXTENSIONS:
            raise PreviewUnavailable("This file type does not use an Office PDF preview.")
        if not source.is_file():
            raise PreviewUnavailable("The original Office document is unavailable.")
        if source.stat().st_size > MAX_SOURCE_BYTES:
            raise PreviewUnavailable("Office previews are limited to 100 MiB per document.")
        if not _CONVERSION_LOCK.acquire(timeout=CONVERSION_TIMEOUT):
            raise PreviewUnavailable("Another Office preview is still being prepared. Try again shortly.")
        try:
            fingerprint = _fingerprint(source)
            directory = _managed_path(source.parent / ".preview", root)
            directory.mkdir(exist_ok=True)
            target = _managed_path(directory / f"{fingerprint}.pdf", root)
            if target.exists():
                try:
                    _validate_pdf(target)
                    return target
                except PreviewUnavailable:
                    pass  # Rebuild a truncated cache without publishing partial output.
            _check_external_content(source)
            with tempfile.TemporaryDirectory(prefix=".converting-", dir=directory) as temporary:
                staging = _managed_path(Path(temporary), root)
                output, report = staging / "preview.pdf", staging / "result.json"
                _run_worker(source, output, report)
                _managed_path(output, root)
                _validate_pdf(output)
                if _fingerprint(source) != fingerprint:
                    raise PreviewUnavailable("The document changed while its preview was being prepared. Try again.")
                os.replace(output, target)
            return target
        finally:
            _CONVERSION_LOCK.release()
    except OSError as exc:
        raise PreviewUnavailable("The Office preview could not access its local source or cache.") from exc


def _office_pids(executable):
    try:
        result = subprocess.run(["tasklist", "/NH", "/FO", "CSV", "/FI", f"IMAGENAME eq {executable}"],
                                capture_output=True, text=True, timeout=10,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise PreviewUnavailable("Office preview cannot check application ownership with the current permissions. "
                                 "Download the original or save it as PDF.") from exc
    return {int(row[1]) for row in csv.reader(io.StringIO(result.stdout))
            if len(row) > 1 and row[0].casefold() == executable.casefold()}


def _new_process_job():
    import win32job

    job = win32job.CreateJobObject(None, "")
    limits = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
    limits["BasicLimitInformation"]["LimitFlags"] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, limits)
    return job


def _own_process_job(pid):
    # A job handle belongs only to this worker. If it times out, closing that
    # handle kills only the verified newly created Office process, never a name.
    import win32api
    import win32job

    job = _new_process_job()
    process = win32api.OpenProcess(0x100 | 0x1 | 0x1000, False, pid)
    try:
        win32job.AssignProcessToJobObject(job, process)
    except Exception:
        job.Close()
        raise
    finally:
        process.Close()
    return job


def _export_word(app, source, output):
    options = ("UpdateLinksAtOpen", "UpdateLinksAtPrint", "UpdateFieldsAtPrint")
    previous = {name: getattr(app.Options, name) for name in options}
    document = None
    try:
        app.Visible = False
        app.DisplayAlerts = 0  # wdAlertsNone
        app.AutomationSecurity = 3  # msoAutomationSecurityForceDisable
        for name in options:
            setattr(app.Options, name, False)
        document = app.Documents.Open(str(source), ConfirmConversions=False, ReadOnly=True,
            AddToRecentFiles=False, Visible=False, NoEncodingDialog=True,
            PasswordDocument="", PasswordTemplate="", WritePasswordDocument="", WritePasswordTemplate="")
        document.ExportAsFixedFormat(str(output), 17, OpenAfterExport=False)  # wdExportFormatPDF
    finally:
        try:
            if document is not None:
                document.Close(SaveChanges=0)
        finally:
            for name, value in previous.items():
                setattr(app.Options, name, value)


def _export_powerpoint(app, source, output):
    presentation = None
    try:
        app.DisplayAlerts = 1  # ppAlertsNone
        app.AutomationSecurity = 3
        presentation = app.Presentations.Open(str(source), ReadOnly=-1, Untitled=0, WithWindow=0)
        # Never call UpdateLinks, activate embedded charts, or run a slideshow.
        presentation.ExportAsFixedFormat(str(output), 2, PrintHiddenSlides=-1, PrintRange=None)
    finally:
        if presentation is not None:
            presentation.Close()


def _convert_with_office(source, output):
    if os.name != "nt":
        raise PreviewUnavailable("Office preview needs Microsoft Word/PowerPoint on Windows. Download the original or save it as PDF.")
    try:
        import pythoncom
        import pywintypes
        import win32com.client
        import win32process
    except ImportError as exc:
        raise PreviewUnavailable("Office preview needs the optional pywin32 package and Microsoft Office.") from exc
    powerpoint = source.suffix.lower() in {".ppt", ".pptx"}
    executable = "POWERPNT.EXE" if powerpoint else "WINWORD.EXE"
    before = _office_pids(executable)
    if powerpoint and before:
        raise PreviewUnavailable("Close PowerPoint before preparing this preview, or download the original. Your open presentations were left untouched.")
    application = job = None
    owned = False
    pythoncom.CoInitialize()
    try:
        application = win32com.client.DispatchEx("PowerPoint.Application" if powerpoint else "Word.Application")
        documents = application.Presentations if powerpoint else application.Documents
        created = _office_pids(executable) - before
        # Word's Application object has no HWND (only its document Windows do).
        # Require one fresh WINWORD process and an empty COM application before
        # assigning ownership. PowerPoint additionally exposes an application HWND.
        if len(created) != 1 or documents.Count:
            raise PreviewUnavailable("Office did not provide a separate empty preview instance. Your open documents were left untouched.")
        pid = created.pop()
        if powerpoint and win32process.GetWindowThreadProcessId(application.HWND)[1] != pid:
            raise PreviewUnavailable("Office did not provide a separate empty preview instance. Your open documents were left untouched.")
        owned = True
        job = _own_process_job(pid)
        (_export_powerpoint if powerpoint else _export_word)(application, source, output)
    except (pywintypes.error, AttributeError) as exc:
        logger.exception("Office preview COM conversion failed")
        raise PreviewUnavailable("Microsoft Office could not preview this file. It may be protected, damaged, or unavailable; download it or save a PDF copy.") from exc
    finally:
        if owned:
            try:
                application.Quit()
            except pywintypes.error:
                pass
        application = None
        if job is not None:
            job.Close()
        pythoncom.CoUninitialize()


def _main():
    parser = argparse.ArgumentParser(description="Local Office preview worker")
    parser.add_argument("--worker", nargs=3, metavar=("SOURCE", "OUTPUT", "REPORT"), required=True)
    source, output, report = map(Path, parser.parse_args().worker)
    try:
        (_convert_with_office if os.name == "nt" else _convert_with_libreoffice)(source, output)
        return 0
    except (PreviewUnavailable, OSError, subprocess.SubprocessError) as exc:
        logger.warning("Office preview conversion unavailable: %s", exc)
        report.write_text(json.dumps({"error": str(exc)}), encoding="utf-8")
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
