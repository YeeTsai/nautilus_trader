# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2026 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------
"""
Re-initializing logging after the last `LogGuard` was dropped must not abort.

The `log` crate's global logger can only be set once per process, while
`LOGGING_INITIALIZED` is reset when the last guard drops. Before the fix the FFI
`logging_init` turned that `Err` into a Rust panic across `extern "C"`, i.e. a
process abort. Every case here therefore runs in its own subprocess: an abort
would otherwise take the whole pytest run with it, and the pre-fix evidence is
exactly "child exits with -SIGABRT".

Bug: yeslab docs/bugs/cross-cutting/2026-09-06-nautilus-ffi-logging-init-abort/bug.md §8
"""

import subprocess
import sys


_HEADER = """
import gc, sys
from nautilus_trader.common.component import LoggingReinitError, init_logging
from nautilus_trader.common.component import is_logging_initialized
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.model.identifiers import TraderId

def _init(tag):
    return init_logging(trader_id=TraderId(tag), machine_id="m", instance_id=UUID4())
"""


def _run(body: str) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", _HEADER + body],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,  # a pre-fix child exits -SIGABRT; that IS the observation
    )


def test_second_init_while_guard_held_raises_runtime_error():
    # T1: the Cython pre-check still owns this case; behaviour is unchanged.
    result = _run(
        """
guard = _init("T-1")
assert is_logging_initialized()
try:
    _init("T-2")
except RuntimeError as e:
    assert not isinstance(e, LoggingReinitError), "should be the already-initialized check"
    print("OK", e)
else:
    raise AssertionError("expected RuntimeError")
""",
    )
    assert result.returncode == 0, result.stderr
    assert "already initialized" in result.stdout


def test_second_init_after_guard_dropped_raises_reinit_error():
    # T2: pre-fix this child aborted with -SIGABRT.
    result = _run(
        """
guard = _init("T-1")
del guard
gc.collect()
assert not is_logging_initialized()
try:
    _init("T-2")
except LoggingReinitError as e:
    assert isinstance(e, RuntimeError)
    print("OK", e)
else:
    raise AssertionError("expected LoggingReinitError")
""",
    )
    assert result.returncode == 0, result.stderr
    assert "cannot be re-initialized" in result.stdout


def test_engines_after_dispose_build_and_warn_once():
    # T3: pre-fix this child aborted with -SIGABRT on the second engine.
    # Two degraded kernels, `always` filter on: exactly one warning proves the
    # once-per-process promise comes from the explicit flag, not from the
    # de-duplication `warnings` does on (message, category, module, lineno).
    result = _run(
        """
import warnings
from nautilus_trader.backtest.engine import BacktestEngine

e1 = BacktestEngine()
e1.dispose()
del e1
gc.collect()
assert not is_logging_initialized()

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    e2 = BacktestEngine()
    e3 = BacktestEngine()

assert e2.kernel.get_log_guard() is None
assert e3.kernel.get_log_guard() is None
reinit = [w for w in caught if issubclass(w.category, RuntimeWarning)
          and "Logging is disabled" in str(w.message)]
assert len(reinit) == 1, [str(w.message) for w in caught]
print("OK", reinit[0].message)
""",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("OK") == 1
