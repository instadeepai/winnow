"""Rich helpers that render in terminals and Jupyter notebooks."""

from __future__ import annotations

from rich.console import Console
from rich.logging import RichHandler

# Shared stderr console for CLI error messages.
STDERR_CONSOLE = Console(stderr=True)


def notebook_safe_rich_handler(**kwargs: object) -> RichHandler:
    """Return a Rich log handler that uses ANSI colours, not OSC-8 hyperlinks.

    OSC-8 file links render as escape noise in many notebook frontends; plain
    ANSI styling (including dimmed paths) still works there.
    """
    kwargs.setdefault("enable_link_path", False)
    return RichHandler(console=STDERR_CONSOLE, **kwargs)
