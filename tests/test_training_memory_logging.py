"""Keep optional host-memory telemetry independent of training availability."""

import logging
import runpy
import sys
from types import SimpleNamespace

import pytest

from histoforge_models.pathology.training import train_seg_loop as loop


def test_import_and_memory_log_without_resource(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "resource", None)

    with caplog.at_level(logging.INFO):
        namespace = runpy.run_path(loop.__file__)
        namespace["_log_host_peak_rss"]()

    assert not caplog.records


@pytest.mark.parametrize(
    "platform,peak_rss", [("linux", 3 * 1024**2 // 2), ("darwin", 3 * 1024**3 // 2)]
)
def test_memory_log_uses_platform_units(monkeypatch, caplog, platform, peak_rss):
    requested = []
    rusage_self = object()

    def getrusage(who):
        requested.append(who)
        return SimpleNamespace(ru_maxrss=peak_rss)

    monkeypatch.setitem(
        sys.modules,
        "resource",
        SimpleNamespace(RUSAGE_SELF=rusage_self, getrusage=getrusage),
    )
    monkeypatch.setattr(sys, "platform", platform)

    with caplog.at_level(logging.INFO, logger=loop.train_logger.name):
        loop._log_host_peak_rss()

    assert requested == [rusage_self]
    assert [record.getMessage() for record in caplog.records] == [
        ">>> host RSS after data loading: 1.5 GiB (peak, main process)"
    ]
