"""Session workspace behavior: persistence, attachments-as-files, manifests."""

from pathlib import Path

import pytest

from optimist.agents.orchestrator import Session
from optimist.attachments import AttachmentError
from optimist.config import Settings
from optimist.messages import TextBlock
from optimist.tools.sandbox import run_python


def make_session(tmp_path: Path) -> Session:
    return Session(settings=Settings(workspace_root=tmp_path / "sessions"))


def test_sandbox_workdir_persists_files(tmp_path):
    work = tmp_path / "ws"
    r1 = run_python("open('note.txt', 'w').write('42')", workdir=work)
    assert r1.returncode == 0
    r2 = run_python("print(open('note.txt').read())", workdir=work)
    assert r2.returncode == 0
    assert "42" in r2.stdout
    # scripts are kept, numbered, for auditability
    assert list(work.glob("_optimist_run_*.py"))


def test_workspace_created_lazily_and_reset(tmp_path):
    session = make_session(tmp_path)
    assert session._workspace is None
    ws1 = session.workspace
    assert ws1.is_dir()
    session.reset()
    assert session._workspace is None
    assert session.workspace != ws1


def test_add_data_file_copies_and_previews(tmp_path):
    session = make_session(tmp_path)
    csv = tmp_path / "orders.csv"
    csv.write_text("sku,qty,price\nA,10,2.5\nB,4,1.0\n")
    desc = session.add_file(csv)
    assert "orders.csv" in desc
    assert (session.workspace / "orders.csv").read_text().startswith("sku,qty")
    [block] = session.pending_attachments
    assert isinstance(block, TextBlock)
    assert "sku,qty,price" in block.text            # preview visible to the model
    assert "./orders.csv" in block.text             # and the path solver code uses


def test_add_missing_data_file_raises(tmp_path):
    session = make_session(tmp_path)
    with pytest.raises(AttachmentError, match="No such file"):
        session.add_file(tmp_path / "ghost.csv")


def test_run_python_handler_uses_workspace(tmp_path):
    session = make_session(tmp_path)
    (session.workspace / "data.txt").write_text("7,11")
    result, is_error = session._handle_run_python(
        {"code": "print(sum(int(x) for x in open('data.txt').read().split(',')))"}
    )
    assert not is_error
    assert "18" in result


def test_manifest_lists_files_but_not_run_scripts(tmp_path):
    session = make_session(tmp_path)
    assert session._workspace_manifest() == ""
    (session.workspace / "data.csv").write_text("a,b\n1,2\n")
    session._handle_run_python({"code": "print('hi')"})
    manifest = session._workspace_manifest()
    assert "data.csv" in manifest
    assert "_optimist_run_" not in manifest


def test_read_recipe_handler(tmp_path):
    session = make_session(tmp_path)
    text, is_error = session._handle_read_recipe({"name": "scheduling"})
    assert not is_error
    assert "add_no_overlap" in text
    text, is_error = session._handle_read_recipe({"name": "nope"})
    assert is_error
    assert "Unknown recipe" in text
