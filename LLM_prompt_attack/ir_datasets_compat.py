"""Compatibility fixes for ir_datasets on native Windows."""

from __future__ import annotations

import os
from typing import Any


class _ClosedNamedTemporaryFileProxy:
    """Proxy tempfile while closing ir_datasets' unnamed download handle."""

    def __init__(self, tempfile_module: Any) -> None:
        self._tempfile_module = tempfile_module

    def NamedTemporaryFile(self, *args: Any, **kwargs: Any) -> Any:
        handle = self._tempfile_module.NamedTemporaryFile(*args, **kwargs)
        handle.close()
        return handle

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tempfile_module, name)


def enable_windows_download_compat(download_module: Any | None = None) -> bool:
    """Close ir_datasets download placeholders before Windows replaces them.

    ir_datasets 0.6.3 leaves a NamedTemporaryFile open in Download.path and
    then calls os.replace on that path. Windows forbids replacing an open
    file. The proxy is scoped to ir_datasets.util.download, leaving Python's
    process-wide tempfile module unchanged.
    """
    if os.name != "nt":
        return False

    if download_module is None:
        import ir_datasets.util.download as download_module

    if getattr(download_module, "_llm_ranker_windows_tempfile_fix", False):
        return False

    download_module.tempfile = _ClosedNamedTemporaryFileProxy(
        download_module.tempfile
    )
    download_module._llm_ranker_windows_tempfile_fix = True
    return True
