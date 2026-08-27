"""Progress bars that stay out of the way in log files.

Interactive runs get a live bar; redirected output (a SLURM job log, a CI run)
gets nothing, leaving only the summary lines each stage prints.
"""

from __future__ import annotations

import os
import sys

from tqdm import tqdm


def _enabled() -> bool:
    override = os.environ.get("AMA_PROGRESS")
    if override is not None:
        return override.lower() not in {"0", "false", "no", "off"}
    return sys.stderr.isatty()


def progress(iterable=None, **kwargs):
    """``tqdm``, disabled when stderr is not a terminal."""
    kwargs.setdefault("leave", False)
    kwargs.setdefault("disable", not _enabled())
    return tqdm(iterable, **kwargs)
