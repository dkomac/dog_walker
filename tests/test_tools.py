from dog_walker.tools.base import ToolRegistry, Tool
from dog_walker.tools.builtin import ReadFile, WriteFile, ListFiles, build_registry


def test_read_file(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello")
    assert ReadFile().run(path=str(f)) == "hello"


def test_write_then_read(tmp_path):
    f = str(tmp_path / "out.txt")
    WriteFile().run(path=f, content="data")
    assert ReadFile().run(path=f) == "data"


def test_list_files(tmp_path):
    (tmp_path / "x.txt").write_text("1")
    (tmp_path / "y.txt").write_text("2")
    out = ListFiles().run(path=str(tmp_path))
    assert "x.txt" in out and "y.txt" in out


def test_registry_runs_by_name(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hi")
    reg = build_registry(["read_file"], confirm_bash=False)
    assert reg.run("read_file", {"path": str(f)}) == "hi"


def test_registry_turns_errors_into_data():
    reg = build_registry(["read_file"], confirm_bash=False)
    result = reg.run("read_file", {"path": "/nope/missing.txt"})
    assert result.startswith("Error:")


def test_specs_expose_enabled_tools_only():
    reg = build_registry(["read_file", "write_file"], confirm_bash=False)
    names = {s.name for s in reg.specs()}
    assert names == {"read_file", "write_file"}


def test_set_preference_appends(tmp_path):
    from dog_walker.tools.builtin import SetPreference
    pf = str(tmp_path / "prefs.md")
    SetPreference(pf).run(text="be terse")
    SetPreference(pf).run(text="use tabs")
    content = open(pf).read()
    assert "- be terse" in content
    assert "- use tabs" in content


def test_registry_includes_set_preference(tmp_path):
    pf = str(tmp_path / "prefs.md")
    reg = build_registry(["set_preference"], confirm_bash=False, preferences_file=pf)
    reg.run("set_preference", {"text": "hello"})
    assert "- hello" in open(pf).read()
