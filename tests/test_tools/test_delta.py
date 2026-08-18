"""The paired (``cov_from`` -> ``cov_to``) entropies.

What these pin, in order of how much they'd cost to get wrong:

1. the delta is taken WITHIN a draw, so ``result``'s interval is the interval of the
   difference — not something reconstructed from the endpoints' intervals, which is not
   possible because HDIs do not subtract;
2. both sides come from ONE shared sample, so a self-delta is exactly 0 rather than the
   sampling noise floor (``phenotypic_flux`` reported 0.209 for a self-flux before its seed
   was pinned);
3. the support is the intersection, within each replicate, and the drop is warned about
   because it moves ``n``;
4. there is no ``delta_mutual_information`` — the scope principle, not an oversight.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import tcri
from tcri import _keys as K

DELTAS = ["delta_clonotypic_entropy", "delta_phenotypic_entropy"]
ITEM = {"delta_clonotypic_entropy": "phenotype", "delta_phenotypic_entropy": "clonotype"}


def _covs(adata):
    return list(adata.uns[K.COVARIATE_CATEGORIES])[:2]


@pytest.fixture(scope="module")
def ragged():
    """A fitted AnnData where some clones are present at only ONE covariate.

    Built rather than carved: slicing an AnnData is not an option here, because the package
    deliberately refuses a sliced object -- its full-space ``uns`` registration arrays
    misalign against the subset, and the error says so. Thin the cells instead and the
    raggedness arises the way it does in real data.
    """
    import contextlib
    import io
    import logging

    import pyro

    from tcri.datasets import simulate_tcri
    from tcri.model._model import TCRIModel

    logging.disable(logging.INFO)
    adata = simulate_tcri(n_clones=30, n_phenotypes=4, n_genes=25, n_cells=260,
                          n_covariates=2, omega_concentration=0.5, seed=3)
    adata.obs["patient"] = "P0"
    adata.obs["patient"] = adata.obs["patient"].astype("category")
    adata.layers["counts"] = adata.X.copy()
    pyro.clear_param_store()
    TCRIModel.setup_anndata(adata, layer="counts", clonotype_key="clone_id",
                            phenotype_key="phenotype", covariate_key="covariate",
                            batch_key="patient")
    model = TCRIModel(adata, n_latent=8, n_hidden=16, n_layers=1, classifier_n_layers=1,
                      classifier_hidden=16, K=4, seed=0)
    with contextlib.redirect_stdout(io.StringIO()):
        model.train(max_epochs=10, batch_size=128, n_steps_kl_warmup=8, accelerator="cpu",
                    enable_progress_bar=False, enable_model_summary=False)
        model.to_anndata(adata)
    logging.disable(logging.NOTSET)

    a, b = list(adata.uns[K.COVARIATE_CATEGORIES])[:2]
    cc = adata.uns[K.METADATA][K.Config.CLONE_COL]
    cov_col = adata.uns[K.METADATA][K.Config.COVARIATE_COL]
    at = {lv: set(adata.obs.loc[adata.obs[cov_col].astype(str) == str(lv), cc].dropna())
          for lv in (a, b)}
    assert at[a] ^ at[b], "fixture is not ragged; nothing to test"
    return adata, a, b, at


# ── the estimand ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", DELTAS)
def test_delta_is_to_minus_from_within_a_draw(name, cohort):
    """Positive means it increased, and the subtraction happens per draw.

    The per-draw part is the whole reason these functions exist rather than being a
    subtraction the caller does: the distribution of a difference is not recoverable from two
    marginal summaries.
    """
    _, adata = cohort
    a, b = _covs(adata)
    res = getattr(tcri.tl, name)(adata, cov_from=a, cov_to=b, groupby="patient",
                                 n_samples=20, random_state=0)

    t = res["table"]
    assert {"value", "value_from", "value_to"} <= set(t.columns)
    assert np.allclose(t["value"], t["value_to"] - t["value_from"], equal_nan=True), (
        "the delta is not to - from"
    )
    assert t["draw"].nunique() == 20, "the endpoints were reduced before the subtraction"

    # `result.value` is the mean of the per-draw deltas, and its HDI is the delta's own
    key = ["patient", ITEM[name]]
    agg = t.groupby(key, observed=True)["value"].agg(["mean", "median"]).reset_index()
    got = res["result"].merge(agg, on=key)
    assert np.allclose(got["value"], got["mean"], equal_nan=True)

    # The HDI brackets the MEDIAN, not necessarily the mean. An HDI is the *narrowest*
    # interval holding 94% of the mass, so a single outlying draw is excluded from the
    # interval while still pulling the mean. Measured on this fixture: draws
    # [-0.0002, 0.0 x18, +0.0038] give mean 0.000218 and hdi (-0.000192, 0.0000396) --
    # the mean sits outside by construction, not by error. Any interval holding >50% of
    # the mass must contain the median, so that is the invariant worth asserting.
    fin = got.dropna(subset=["hdi_low", "hdi_high", "median"])
    assert len(fin)
    assert (fin["hdi_low"] <= fin["median"]).all() and (fin["median"] <= fin["hdi_high"]).all()
    assert (fin["hdi_low"] <= fin["hdi_high"]).all()


@pytest.mark.parametrize("name", DELTAS)
def test_a_self_delta_is_exactly_zero(name, cohort):
    """Both sides must come from one shared sample.

    ``phenotypic_flux`` learned this: at ``random_state=None`` the flux of a covariate against
    itself — exactly 0 by construction — came back as 0.209 at n_samples=16, which was the
    sampling noise floor being reported as a result. Independent draws would give a delta
    whose spread grows with n_samples and whose mean is only asymptotically 0.
    """
    _, adata = cohort
    a, _ = _covs(adata)
    res = getattr(tcri.tl, name)(adata, cov_from=a, cov_to=a, groupby="patient",
                                 n_samples=16, random_state=0, inplace=False)
    v = res["table"]["value"].to_numpy(dtype=float)
    assert np.nanmax(np.abs(v)) == 0.0, (
        f"a self-delta is {np.nanmax(np.abs(v)):.6f}, not 0 — the two sides are not the "
        f"same draw"
    )


@pytest.mark.parametrize("name", DELTAS)
def test_the_support_is_the_intersection_and_the_drop_is_reported(name, ragged):
    """A delta needs both endpoints, and dropping clones moves the n a contrast is built on."""
    adata, a, b, at = ragged
    one_sided = at[a] ^ at[b]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = getattr(tcri.tl, name)(adata, cov_from=a, cov_to=b, inplace=False)
    messages = [str(w.message) for w in caught]
    assert any("absent from" in m and "both endpoints" in m for m in messages), (
        f"the {len(one_sided)} dropped clones were not reported: {messages}"
    )

    if name == "delta_phenotypic_entropy":
        assert not (set(res["result"]["clonotype"]) & one_sided), "a one-sided clone survived"
        assert set(res["result"]["clonotype"]) == at[a] & at[b]


def test_the_intersection_changes_the_clonotypic_answer(ragged):
    """It is not cosmetic: intersecting fixes ``log2(C)`` on both sides so it cancels.

    Computing H(c|phi) at each level over that level's OWN clones and subtracting is a
    different quantity — the normalizer moves with the clone count, so a repertoire that
    contracts reports an entropy change it did not have. This asserts the two disagree, which
    is what makes the choice load-bearing rather than a detail.
    """
    adata, a, b, _at = ragged

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matched = tcri.tl.delta_clonotypic_entropy(adata, cov_from=a, cov_to=b,
                                                   inplace=False)["result"]
    naive_a = tcri.tl.clonotypic_entropy(adata, covariate=a, inplace=False)["result"]
    naive_b = tcri.tl.clonotypic_entropy(adata, covariate=b, inplace=False)["result"]
    naive = (naive_b.set_index("phenotype")["value"] - naive_a.set_index("phenotype")["value"])

    got = matched.set_index("phenotype")["value"]
    assert not np.allclose(got.reindex(naive.index), naive, equal_nan=True), (
        "the intersection made no difference; the fixture cannot discriminate"
    )


# ── the scope principle ──────────────────────────────────────────────────────

def test_there_is_no_delta_mutual_information():
    """MI has no item axis, so a cross-covariate comparison of it is a subtraction of two
    cached scalars — the caller's, not the package's. See the scope principle in the API
    contract. Its absence is a decision, and this test is where that decision lives."""
    assert not hasattr(tcri.tl, "delta_mutual_information")
    assert "delta_mutual_information" not in tcri.tl.__all__
    assert not hasattr(tcri.pl, "delta_mutual_information")


@pytest.mark.parametrize("name", DELTAS)
def test_the_delta_reduces_to_a_named_item(name, cohort):
    """Every delta names its unit of reduction. That is the test a proposed metric has to
    pass to be here at all."""
    _, adata = cohort
    a, b = _covs(adata)
    res = getattr(tcri.tl, name)(adata, cov_from=a, cov_to=b, inplace=False)
    assert ITEM[name] in res["result"].columns


# ── the payload ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", DELTAS)
def test_no_direction_probability_column(name, cohort):
    """`p_gt` is deliberately absent, so the delta schema stays the parents' schema.

    It reads as a frequentist p-value, its resolution caps at 1/n_samples (1.0 means "no draw
    crossed zero", not certainty), and per-item it invites filtering on the survivors.
    `hdi_*` answers direction already, and the graded version is one line off `table`.
    """
    _, adata = cohort
    a, b = _covs(adata)
    res = getattr(tcri.tl, name)(adata, cov_from=a, cov_to=b, groupby="patient",
                                 n_samples=10, random_state=0)
    assert not ({"p_gt", "p_lt", "prob_increase"} & set(res["result"].columns))

    # ...and it is genuinely one line away for a caller who wants it
    t = tcri.get.table(adata, name, which="table")
    graded = t.groupby(ITEM[name], observed=True)["value"].apply(lambda v: (v > 0).mean())
    assert graded.between(0, 1).all()


@pytest.mark.parametrize("name", DELTAS)
def test_the_contrast_counts_replicates(name, cohort):
    """Same guarantee as every other metric: `stats` n is patients, never items."""
    _, adata = cohort
    a, b = _covs(adata)
    res = getattr(tcri.tl, name)(adata, cov_from=a, cov_to=b, groupby="patient",
                                 splitby="response")
    row = res["stats"].iloc[0]
    assert row["n_a"] == 3 and row["n_b"] == 3
    assert row["replicate_unit"] == "patient"
    assert len(res["result"]) > 6, "fixture too small to discriminate"


@pytest.mark.parametrize("name", DELTAS)
def test_stored_and_returned_agree(name, cohort):
    """The store-once invariant, extended to the new tools."""
    _, adata = cohort
    a, b = _covs(adata)
    returned = getattr(tcri.tl, name)(adata, cov_from=a, cov_to=b, groupby="patient")
    cached = tcri.get.result(adata, name)
    assert set(cached) == set(returned)
    for slot, frame in returned.items():
        if frame is None:
            assert cached[slot] is None
        else:
            pd.testing.assert_frame_equal(cached[slot], frame, check_dtype=False, obj=slot)
    params = tcri.get.params(adata, name)
    assert params["cov_from"] == a and params["cov_to"] == b
