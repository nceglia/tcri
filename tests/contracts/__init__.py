"""The frozen contract manifests, and the reason they live under ``tests/``.

These four files are **contract enforcement, not package code**. Nothing in ``tcri`` imports
them — their only consumers are the four conformance tests next door — yet while they sat in
``tcri/tools/`` and ``tcri/model/`` they were collected by ``packages.find`` and shipped inside
every wheel. A user pip-installing tcri received the manifests that police tcri's development,
which is noise at best and, at worst, invites someone to import one and build on a frozen
declaration as though it were an API.

Moving them here makes the role legible from the path: ``tests/contracts/api.pyi`` is obviously
a test fixture in a way that ``tcri/_contract.pyi`` never was.

**Governance is unchanged.** These remain owner-approval-only via ``.github/CODEOWNERS`` (the
paths are updated to match), and the rule in ``CLAUDE.md`` still holds: only @nceglia and
@salehis may change a contract, a conformance test, or a source document. A failing conformance
test still means *stop and decide*, never "loosen the manifest until it passes."
"""
