"""``tl.delta_*`` — the paired (``cov_from`` -> ``cov_to``) forms of the two entropies.

**Why only two.** A cross-covariate comparison belongs in the API when producing it requires
applying a metric at a level the public surface does not expose — the scope principle in the
API contract. The entropies reduce to an item (a phenotype, a clonotype), so their deltas need
the metric evaluated inside the engine's draw loop with identity aligned across two covariate
blocks that hold different row sets; a caller doing it from cached results would have to
reimplement ``H`` itself. ``mutual_information`` has no item axis — it is already the
repertoire-level number, so its "delta" is a subtraction of two cached scalars and belongs to
the caller. There is no ``delta_mutual_information``, deliberately.

**Delta = ``cov_to`` − ``cov_from``, taken within a draw.** Positive means it increased. The
per-draw evaluation is what makes the interval meaningful: HDIs do not subtract, so the
interval on a delta cannot be recovered from the intervals on its endpoints.

**Support is the intersection**, within each replicate — clones present at BOTH levels, as
``phenotypic_flux`` already requires. It does two different jobs depending on where the item
axis sits, which is worth stating because they look unrelated:

* ``delta_phenotypic_entropy`` — items ARE clones, so the intersection decides which rows
  exist. This is the pairing itself: the same clonotype at two timepoints, a biological
  barcode rather than a category measured twice.
* ``delta_clonotypic_entropy`` — items are phenotypes, present at both levels by construction.
  The intersection instead constrains the clone set summed over *inside* ``H(c|φ)``, which
  makes ``log2(C)`` identical on both sides so the normalizer cancels out of the difference.
  Without it a repertoire contracting 150 -> 90 clones reports **+0.078** normalized entropy
  having not redistributed at all — and the artefact points the same way as treatment does.

That cancellation is why ``n_clones_ref`` keeps its inherited ``None`` default here rather
than gaining a special one.
"""
from __future__ import annotations

import warnings

import numpy as np

from .._state import keys as K
from .._state import schemas
from .._state.storage import tl_result, with_resolved_params
from ._common import (
    build_result,
    build_stats,
    clone_col,
    joint_draws,
    metric_table,
    resolve_groupby,
    validate_splitby,
)
from ._entropy import _clonotypic_one, _phenotypic_one

__all__ = ["delta_clonotypic_entropy", "delta_phenotypic_entropy"]


def _clones_at(adata, covariate, group_clones=None):
    """Clone ids with cells at ``covariate`` (optionally restricted to a group's clones)."""
    meta = adata.uns[K.METADATA]
    obs = adata.obs
    at = obs.loc[obs[meta[K.Config.COVARIATE_COL]].astype(str) == str(covariate),
                 clone_col(adata)].dropna().unique().tolist()
    if group_clones is None:
        return at
    allowed = set(group_clones)
    return [c for c in at if c in allowed]


def _delta_metric(adata, *, kind, cov_from, cov_to, groupby, splitby, n_samples, temperature,
                  clones, weighted, normalized, random_state, n_clones_ref=None, device=None):
    """Shared body. Mirrors ``_entropy_metric`` with the covariate axis contracted."""
    item_col = "phenotype" if kind == "clonotypic" else "clonotype"

    def _one(clone_ids, J, cols):
        if kind == "clonotypic":
            return _clonotypic_one(J, cols, normalized=normalized, n_clones_ref=n_clones_ref)
        return _phenotypic_one(clone_ids, J, cols, normalized=normalized)

    gkey, resolved = resolve_groupby(adata, groupby)
    validate_splitby(adata.obs, gkey, splitby)

    # Both sides must come from ONE shared sample. The engine draws over every ct row and then
    # selects a covariate's block, so two calls carrying the same seed realise the same
    # underlying draw. `phenotypic_flux` learned this the hard way: at random_state=None the
    # flux of a covariate against ITSELF -- exactly 0 by construction -- came back as 0.209 at
    # n_samples=16, which was the sampling noise floor being reported as a result.
    if random_state is None:
        random_state = int(np.random.SeedSequence().generate_state(1)[0])

    dropped = []

    def _compute(clone_subset):
        at_from = _clones_at(adata, cov_from, clone_subset)
        at_to = _clones_at(adata, cov_to, clone_subset)
        shared = [c for c in at_from if c in set(at_to)]
        n_union = len(set(at_from) | set(at_to))
        if n_union > len(shared):
            dropped.append((n_union - len(shared), n_union))
        if not shared:
            return []

        kw = dict(n_samples=n_samples, weighted=weighted, device=device,
                  temperature=temperature, clones=shared, random_state=random_state)
        draws_from, cols = joint_draws(adata, cov_from, **kw)
        draws_to, _ = joint_draws(adata, cov_to, **kw)

        per = []
        for (ids_f, Jf), (ids_t, Jt) in zip(draws_from, draws_to):
            h_from = _one(ids_f, Jf, cols)
            h_to = _one(ids_t, Jt, cols)
            per.append({
                item: {"value": h_to[item] - h_from[item],
                       "value_from": h_from[item], "value_to": h_to[item]}
                for item in h_from if item in h_to
            })
        return per

    table = metric_table(adata, covariate=None, groupby=gkey, splitby=splitby, clones=clones,
                         item_col=item_col, compute=_compute,
                         extra_labels={"cov_from": cov_from, "cov_to": cov_to})

    if dropped:
        n_drop = sum(d for d, _ in dropped)
        n_tot = sum(t for _, t in dropped)
        warnings.warn(
            f"delta_{kind}_entropy: {n_drop} of {n_tot} clones are absent from "
            f"{cov_from!r} or {cov_to!r} and were dropped — a delta needs both endpoints. "
            f"This changes the support the metric is computed over, and where a replicate "
            f"loses all of its clones it also leaves the contrast, moving n.",
            UserWarning, stacklevel=3,
        )

    result = build_result(table, extra_values=("value_from", "value_to"))
    stats = build_stats(result, groupby=gkey, splitby=splitby)

    payload = {"table": table, "result": result, "stats": stats}
    return with_resolved_params(payload, groupby=gkey) if resolved else payload


@tl_result(key=K.DELTA_CLONOTYPIC_ENTROPY, version=1, schema=schemas.DeltaClonotypicEntropy)
def delta_clonotypic_entropy(adata, *, cov_from, cov_to, groupby=None, splitby=None,
                             n_samples=0, temperature=1.0, clones=None, weighted=False,
                             normalized=True, n_clones_ref=None, random_state=None,
                             device=None, key_added=None, inplace=True):
    """ΔH[P(c|φ)] per phenotype: ``cov_to`` minus ``cov_from``, in bits.

    "Did this phenotype draw on a wider or narrower clone pool?" The item is a phenotype — a
    category measured twice, not an entity that persisted — so the intersection here acts on
    the clone set summed over, not on which rows exist. See the module docstring.
    """
    return _delta_metric(adata, kind="clonotypic", cov_from=cov_from, cov_to=cov_to,
                         groupby=groupby, splitby=splitby, n_samples=n_samples,
                         temperature=temperature, clones=clones, weighted=weighted,
                         normalized=normalized, random_state=random_state,
                         n_clones_ref=n_clones_ref, device=device)


@tl_result(key=K.DELTA_PHENOTYPIC_ENTROPY, version=1, schema=schemas.DeltaPhenotypicEntropy)
def delta_phenotypic_entropy(adata, *, cov_from, cov_to, groupby=None, splitby=None,
                             n_samples=0, temperature=1.0, clones=None, weighted=False,
                             normalized=True, random_state=None, device=None,
                             key_added=None, inplace=True):
    """ΔH[P(φ|c)] per clone: ``cov_to`` minus ``cov_from``, in bits.

    "Did this clone become more or less plastic?" The item is a clonotype, so each row is the
    same entity at two timepoints — the one metric here whose pairing is a biological barcode.
    Complementary to ``phenotypic_flux``, which measures how FAR a clone moved without saying
    whether it spread or concentrated.
    """
    return _delta_metric(adata, kind="phenotypic", cov_from=cov_from, cov_to=cov_to,
                         groupby=groupby, splitby=splitby, n_samples=n_samples,
                         temperature=temperature, clones=clones, weighted=weighted,
                         normalized=normalized, random_state=random_state, device=device)
