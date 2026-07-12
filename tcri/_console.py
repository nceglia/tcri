"""Single home for tcri's colored console helpers.

Replaces the ``_ok`` / ``_info`` / ``_warn`` / ``_fin`` blocks that were
copy-pasted (with inconsistent ANSI aliases) into ``metrics`` / ``preprocessing``
/ ``plotting``. Import from here; do not redefine.

Behavior is preserved verbatim so the dedup is a no-op at runtime. Both spellings
of each color alias used across the old files are exported (``GRN``/``GREEN``,
``CYN``/``CYAN``, ``MAG``/``MAGENT``) so existing direct uses resolve unchanged.

Note: this stays print-based for the dedup PR. Routing verbosity through scanpy's
logger (silenceable, leveled) is a follow-up refinement (REFACTOR_NOTES).
"""

# ── ANSI palette (all aliases the legacy files used) ─────────────────────────
RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
GREEN = GRN = "\x1b[32m"
CYAN = CYN = "\x1b[36m"
MAGENT = MAG = "\x1b[35m"
YLW = "\x1b[33m"
RED = "\x1b[31m"


def _ok(msg: str, quiet: bool = False):
    """Success mark."""
    if not quiet:
        print(f"{GRN}✅ {msg}{RESET}")


def _info(key: str, txt, quiet: bool = False):
    """Key-value info line."""
    if not quiet:
        print(f"   {CYN}🎯 {key:<22}{DIM}{txt}{RESET}")


def _warn(msg: str, quiet: bool = False):
    """Warning line."""
    if not quiet:
        print(f"{YLW}⚠️  {msg}{RESET}")


def _fin(quiet: bool = False):
    """Final flourish."""
    if not quiet:
        print(f"{MAG}✨  Done!{RESET}")
