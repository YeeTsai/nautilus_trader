"""
Tests that the build script replaces compiled artifacts atomically.

Regression tests for
`docs/bugs/cross-cutting/2026-09-07-nautilus-build-so-inplace-overwrite/bug.md`
(in the yeslab superproject that vendors this fork).

Rewriting a shared library in place kills any process that still has it mapped:
on Linux the truncate window faults an as-yet unread page with SIGBUS, and a
rewritten file leaves the process executing something it never mapped. The build
script writes back into the source tree in three places, and every one of them
used to do exactly that -- two `shutil.copyfile` calls and an in-place `strip`.

These tests drive the real functions from `build.py`. Re-creating their semantics
here instead would keep passing if the production code regressed, which is the
one thing this file exists to prevent. For the same reason the source and the
destination are built from *different* sources, so that copying the wrong one, or
writing the old bytes back, cannot pass.

Linux only: macOS mapping and overwrite semantics differ, so a green run there
would say nothing about the behaviour being pinned.
"""

import os
import platform
import shutil
import stat
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest


pytestmark = [
    pytest.mark.skipif(
        platform.system() != "Linux",
        reason="mmap overwrite semantics are Linux-specific",
    ),
    pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc to build the probe library"),
    pytest.mark.skipif(shutil.which("strip") is None, reason="needs binutils strip"),
]

build = pytest.importorskip("build", reason="build.py needs Cython/numpy importable")

# Far enough apart that the cold function is not faulted in when the warm one runs.
# This is the probe's design rationale, not a measured page state.
PROBE_FUNCTIONS = 1500
COLD_FUNCTION = PROBE_FUNCTIONS - 1

# 0o666 & ~0o077 == 0o600, so a freshly created file is unambiguous.
FRESH_UMASK = 0o077
FRESH_MODE = 0o600
# What the Cython site adds on top: mode |= (mode & 0o444) >> 2.
FRESH_MODE_EXECUTABLE = 0o700

_CHILD = """
import ctypes, sys

lib = ctypes.CDLL({so!r})
lib.f0.restype = ctypes.c_long
lib.f0(1)
cold = getattr(lib, "f{cold}")
cold.restype = ctypes.c_long
print("READY", flush=True)
sys.stdin.readline()
cold(1)
print("SURVIVED", flush=True)
"""


def _compile_probe(workdir: Path, name: str, salt: int) -> Path:
    """
    Build a shared library big enough to have pages that stay cold.

    `salt` changes the emitted code, so two probes are byte-different while
    exporting the same symbols.
    """
    source = workdir / f"{name}.c"
    with source.open("w") as f:
        for i in range(PROBE_FUNCTIONS):
            f.write(f"long f{i}(long x) {{\n")
            for j in range(24):
                f.write(f"  x = x * {3 + salt} + {i * 31 + j + salt}; x ^= (x >> 7);\n")
            f.write("  return x;\n}\n")

    so = workdir / f"{name}.so"
    subprocess.run(  # noqa: S603 (safe - no untrusted input)
        ["gcc", "-O0", "-shared", "-fPIC", "-o", str(so), str(source)],  # noqa: S607
        check=True,
        capture_output=True,
    )
    return so


@pytest.fixture(scope="module")
def probes(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """
    Return (installed, freshly_built) probe libraries with different contents.
    """
    workdir = tmp_path_factory.mktemp("probe")
    return _compile_probe(workdir, "installed", 0), _compile_probe(workdir, "built", 5)


@pytest.fixture
def fresh_umask() -> int:
    previous = os.umask(FRESH_UMASK)
    yield FRESH_UMASK
    os.umask(previous)


def _start_holder(so: Path) -> subprocess.Popen:
    """
    Start a process that maps `so`, warms one function and resolves a cold one.

    The cold symbol is resolved before the caller rewrites the file so that a
    later crash can only come from the call itself, never from dlsym.
    """
    proc = subprocess.Popen(  # noqa: S603 (safe - no untrusted input)
        [sys.executable, "-c", _CHILD.format(so=str(so), cold=COLD_FUNCTION)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == "READY"
    return proc


def _touch_cold_page(proc: subprocess.Popen) -> None:
    """
    Let the holder call into a page it has never executed, and require survival.
    """
    assert proc.stdin is not None
    assert proc.stdout is not None
    proc.stdin.write("\n")
    proc.stdin.flush()
    returncode = proc.wait(timeout=60)
    assert returncode == 0, f"holder died with {returncode} (-7 SIGBUS, -11 SIGSEGV)"
    assert "SURVIVED" in proc.stdout.read()


def _place(content: Path, target: Path, mode: int | None = None) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(content, target)
    if mode is not None:
        target.chmod(mode)
    return target.stat().st_ino


class _StubBuildExt:
    """
    The two attributes `_copy_build_dir_to_project` actually uses.
    """

    def __init__(self, build_lib: Path, outputs: list[Path]) -> None:
        self.build_lib = str(build_lib)
        self._outputs = [str(o) for o in outputs]

    def get_outputs(self) -> list[str]:
        return self._outputs


def _rust_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    cargo_target = tmp_path / "target"
    monkeypatch.setattr(build, "CARGO_TARGET_DIR", str(cargo_target))
    source = cargo_target / f"{build.RUST_LIB_PFX}nautilus_pyo3.{build.RUST_DYLIB_EXT}"
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    destination = tmp_path / "nautilus_trader/core" / f"nautilus_pyo3{ext_suffix}"
    return source, destination


def test_copy_build_dir_to_project_replaces_atomically(
    probes: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed, built = probes
    monkeypatch.chdir(tmp_path)

    relative = Path("nautilus_trader/core/probe.so")
    build_lib = tmp_path / "build" / "lib"
    _place(built, build_lib / relative, 0o644)
    inode_before = _place(installed, tmp_path / relative, 0o755)

    holder = _start_holder(tmp_path / relative)
    try:
        build._copy_build_dir_to_project(_StubBuildExt(build_lib, [build_lib / relative]))
    finally:
        if holder.poll() is not None:  # pragma: no cover - only on a failing run
            holder.kill()

    destination = tmp_path / relative
    assert destination.stat().st_ino != inode_before, "destination was rewritten in place"
    assert destination.read_bytes() == built.read_bytes(), "the freshly built artifact did not land"
    # The pre-existing mode is what the destination keeps, plus the execute bits
    # the build script has always added for the Cython extensions.
    assert stat.S_IMODE(destination.stat().st_mode) == 0o755
    assert not list(tmp_path.rglob("*.tmp")), "temporary artefact left behind"
    _touch_cold_page(holder)


def test_copy_build_dir_to_project_creates_a_missing_destination(
    probes: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fresh_umask: int,
) -> None:
    """
    First build: there is nothing to preserve, so the mode comes from the umask.
    """
    _, built = probes
    monkeypatch.chdir(tmp_path)

    relative = Path("nautilus_trader/core/probe.so")
    build_lib = tmp_path / "build" / "lib"
    _place(built, build_lib / relative, 0o644)
    (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)

    build._copy_build_dir_to_project(_StubBuildExt(build_lib, [build_lib / relative]))

    destination = tmp_path / relative
    assert destination.read_bytes() == built.read_bytes()
    assert stat.S_IMODE(destination.stat().st_mode) == FRESH_MODE_EXECUTABLE
    assert not list(tmp_path.rglob("*.tmp")), "temporary artefact left behind"


def test_copy_rust_dylibs_to_project_replaces_atomically(
    probes: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed, built = probes
    monkeypatch.chdir(tmp_path)

    source, destination = _rust_paths(tmp_path, monkeypatch)
    _place(built, source, 0o644)
    # Deliberately not executable: this site never chmod'd, and the fix must not
    # start adding execute bits that were not there before.
    inode_before = _place(installed, destination, 0o644)

    holder = _start_holder(destination)
    try:
        build._copy_rust_dylibs_to_project()
    finally:
        if holder.poll() is not None:  # pragma: no cover - only on a failing run
            holder.kill()

    assert destination.stat().st_ino != inode_before, "destination was rewritten in place"
    assert destination.read_bytes() == built.read_bytes(), "the freshly built dylib did not land"
    assert stat.S_IMODE(destination.stat().st_mode) == 0o644, "must not gain execute bits"
    assert not list(tmp_path.rglob("*.tmp")), "temporary artefact left behind"
    _touch_cold_page(holder)


def test_copy_rust_dylibs_to_project_creates_a_missing_destination(
    probes: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fresh_umask: int,
) -> None:
    """
    First build: umask decides, and this site must not add execute bits.
    """
    _, built = probes
    monkeypatch.chdir(tmp_path)

    source, destination = _rust_paths(tmp_path, monkeypatch)
    _place(built, source, 0o644)
    destination.parent.mkdir(parents=True, exist_ok=True)

    build._copy_rust_dylibs_to_project()

    assert destination.read_bytes() == built.read_bytes()
    assert stat.S_IMODE(destination.stat().st_mode) == FRESH_MODE
    assert not list(tmp_path.rglob("*.tmp")), "temporary artefact left behind"


def test_strip_unneeded_symbols_replaces_atomically(
    probes: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed, _ = probes
    monkeypatch.chdir(tmp_path)

    target = tmp_path / "nautilus_trader/core/probe.so"
    inode_before = _place(installed, target, 0o755)
    size_before = target.stat().st_size

    holder = _start_holder(target)
    try:
        build._strip_unneeded_symbols()
    finally:
        if holder.poll() is not None:  # pragma: no cover - only on a failing run
            holder.kill()

    assert target.stat().st_ino != inode_before, "file was stripped in place"
    assert target.stat().st_size < size_before, "nothing was actually stripped"
    assert stat.S_IMODE(target.stat().st_mode) == 0o755, "mode not carried over from the original"
    assert not list(tmp_path.rglob("*.tmp")), "temporary artefact left behind"
    _touch_cold_page(holder)


def test_build_script_has_no_in_place_writes_to_the_source_tree() -> None:
    """
    Guard the shape as well: the three call sites must not reacquire a way of
    writing the destination directly. This is a supplement to the behavioural
    tests above, not a substitute for them.
    """
    source = Path(build.__file__).read_text()
    assert "shutil.copyfile(output, relative_extension)" not in source
    assert "shutil.copyfile(src=src, dst=dst)" not in source
    assert source.count("os.replace(") == 3
