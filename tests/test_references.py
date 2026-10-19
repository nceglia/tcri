"""No model-based number is reported bare.

Every quantity in this package is positive for data with no structure in it. Mutual information
is positive for any clone x phenotype table, an entropy is a number whatever the table says, a
flux is non-zero whenever two fits differ at all, and an in-silico knockout moves the phenotype
call for every gene. The reference is what says how much of the number is the data: the same
model, the same knobs, the same seed, the same training arguments, on one permuted label vector.

Two rules carry the weight here.

The reference run is **the caller's own call on the null**, not a call at defaults: `groupby`,
`splitby`, `clones`, `weighted`, `normalized`, `normalize_mode`, `n_clones_ref`,
`distance_metric`, `temperature` and `n_samples` each change the estimand, so a reference at
defaults is a different quantity subtracted from a different quantity. Only the fit,
`null_model` and the seed of the draws differ.

And `adjusted` is **computed per draw**. The null is the value expected under no association,
and the fit and the null are independent fits, so the fit's draw d is paired with the null's
draw d: `adjusted` has per-draw rows in `table`, and a mean, an sd and an HDI in `result`, like
any value. The null draws from its own seed, derived from the caller's.

`null_model=None` reproduces the values and the `uns` key set of a run with no reference at all.
Four shape changes fire unconditionally, and `test_rule_9_shape_changes_are_the_only_ones` pins
them as the only ones.
"""
from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pandas as pd
import pytest

import tcri
from .conftest import frames
from tcri._state import keys as K
from tcri._state import _reference

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture
def ref(cohort):
    """A private copy of the fitted cohort, which already carries all three nulls."""
    model, adata = cohort
    return model, adata.copy()


def _cov(adata):
    return list(adata.uns[K.COVARIATE_CATEGORIES])


# ── the columns, and what they mean ──────────────────────────────────────────

def test_a_metric_with_a_reference_reports_adjusted(ref):
    """Per value column, on every scored metric and on the perturbation, to 1e-12.

    Also the shape rule: one ``null_*`` per native column and one ``adjusted*`` per native
    column that is not a denominator. A denominator has a reference -- both are needed to
    recover the bit values -- and no adjusted column, because a difference of two normalizers
    is not a quantity anyone reports. And no frame carries the retired word: ``result``,
    ``table`` and ``stats`` have no ``excess*`` column.
    """
    model, adata = ref
    a, b = _cov(adata)[:2]
    calls = {
        "mutual_information": lambda: tcri.tl.mutual_information(adata, covariate=a),
        "clonotypic_entropy": lambda: tcri.tl.clonotypic_entropy(adata, covariate=a),
        "phenotypic_entropy": lambda: tcri.tl.phenotypic_entropy(adata, covariate=a),
        "phenotypic_flux": lambda: tcri.tl.phenotypic_flux(adata, cov_from=a, cov_to=b),
        "delta_clonotypic_entropy": lambda: tcri.tl.delta_clonotypic_entropy(
            adata, cov_from=a, cov_to=b),
        "delta_phenotypic_entropy": lambda: tcri.tl.delta_phenotypic_entropy(
            adata, cov_from=a, cov_to=b),
        "gene_importance": lambda: tcri.perturb.gene_importance(
            model, adata, genes=list(adata.var_names[:4])),
    }
    for name, call in calls.items():
        fn = getattr(tcri.tl, name, None) or getattr(tcri.perturb, name)
        assert call() is None, f"{name}: a storing call returned its result"
        payload = tcri.get.result(adata, name)
        if "replicate" in payload:
            levels = [lv for lv in ("replicate", "item") if payload.get(lv) is not None]
            slots = {f"{lv}.{f}": payload[lv][f] for lv in levels for f in ("draws", "summary")}
            summaries = [payload[lv]["summary"] for lv in levels]
        else:
            slots = {slot: payload.get(slot) for slot in ("result", "table")}
            summaries = [payload["result"]]
        slots["stats"] = payload.get("stats")
        for slot, frame in slots.items():
            if frame is not None:
                assert not [c for c in frame.columns if str(c).startswith("excess")], (
                    f"{name}: {slot} carries {list(frame.columns)}")
        pairs = _reference.column_pairs(fn.tcri_values, fn.tcri_denominators)
        for result in summaries:
            if not len(result):
                continue
            for native, null, adjusted in pairs:
                assert null in result.columns, f"{name}: no {null}"
                if adjusted is None:
                    assert f"adjusted_{native}" not in result.columns
                    continue
                got = result[adjusted].to_numpy(dtype=float)
                want = (result[native].to_numpy(dtype=float)
                        - result[null].to_numpy(dtype=float))
                np.testing.assert_allclose(got, want, atol=1e-12, equal_nan=True,
                                           err_msg=f"{name}: {adjusted} != {native} - {null}")


def test_null_value_is_the_metric_of_the_null(ref):
    """``result.null_value`` is the mean of the caller's own call with ``fit=`` set to each fit
    of the reference, row for row.

    At the caller's ``n_samples`` and under the seed the reference derives from the caller's
    ``random_state`` and each null's index: at ``n_samples>0`` with ``random_state=None`` the
    engine draws a fresh seed per call, so the identity would be asserted against a different
    sample and would fail for a reason that has nothing to do with the reference. The derived
    seed is a function of the caller's, which is what makes the explicit-seed form exact rather
    than approximate.
    """
    _, adata = ref
    a = _cov(adata)[0]
    kw = dict(covariate=a, groupby="patient", n_samples=4)
    with_ref = frames(tcri.tl.mutual_information(adata, **kw, random_state=7, inplace=False))[1]
    members = K.members(adata, "null.phenotype")
    assert len(members) > 1
    own = pd.concat([
        frames(tcri.tl.mutual_information(adata, **kw, random_state=_reference.null_seed(7, i),
                                   fit=fit, null_model=None, inplace=False))[0]
        for i, fit in enumerate(members)])
    of_null = own.groupby("patient", observed=True)[["value", "denom"]].mean().reset_index()
    merged = with_ref.merge(of_null, on="patient", suffixes=("", "_own"))
    np.testing.assert_allclose(merged["null_value"], merged["value_own"], atol=1e-12)
    np.testing.assert_allclose(merged["null_denom"], merged["denom_own"], atol=1e-12)


def test_the_reference_forwards_every_argument(ref):
    """A non-default estimand moves the reference too.

    Run at ``weighted=True, normalize_mode="average"``: if the reference were computed at
    defaults its ``null_value`` would equal the defaults run's, and it does not. This is the
    assertion that fails when someone "simplifies" the forwarded set.
    """
    _, adata = ref
    a = _cov(adata)[0]
    plain = frames(tcri.tl.mutual_information(adata, covariate=a, inplace=False))[1]
    odd = frames(tcri.tl.mutual_information(adata, covariate=a, weighted=True,
                                     normalize_mode="average", inplace=False))[1]
    assert not np.isclose(float(plain["null_value"].iloc[0]),
                          float(odd["null_value"].iloc[0]), atol=1e-9), (
        "the reference ignored weighted/normalize_mode; it was computed at defaults")
    assert not np.isclose(float(plain["null_denom"].iloc[0]),
                          float(odd["null_denom"].iloc[0]), atol=1e-9)


def test_the_adjusted_value_of_a_fit_against_itself_is_zero(ref):
    """Point a metric's reference at its own fit and the adjusted value is zero.

    The cleanest statement of what `adjusted` is. Exactly zero at ``n_samples=0``, where both
    sides are the one point value. At ``n_samples=30`` the two sides are independent samples of
    the same fit, so the paired differences scatter around zero and their interval covers it.
    """
    _, adata = ref
    a = _cov(adata)[0]
    kw = dict(covariate=a, groupby="patient", random_state=3, fit="null.phenotype[0]",
              null_model="null.phenotype[0]", inplace=False)
    res = frames(tcri.tl.mutual_information(adata, n_samples=0, **kw))[1]
    np.testing.assert_allclose(res["adjusted"].to_numpy(dtype=float), 0.0, atol=1e-12)

    res = frames(tcri.tl.mutual_information(adata, n_samples=30, **kw))[1]
    assert (res["adjusted_hdi_low"] < 0).all() and (res["adjusted_hdi_high"] > 0).all(), (
        res[["adjusted", "adjusted_hdi_low", "adjusted_hdi_high"]])


def test_a_delta_adjusted_value_is_closed_within_its_own_result(ref):
    """``adjusted - (adjusted_to - adjusted_from)`` equals ``value - (value_to - value_from)``.

    The reference introduces no gap that was not already there. It is deliberately NOT asserted
    to be zero: ``value == value_to - value_from`` already fails when a draw has a non-finite
    endpoint, because the three columns are averaged over their own masks, and that is a
    pre-existing property of the delta result rather than something the reference caused.
    """
    _, adata = ref
    a, b = _cov(adata)[:2]
    r = tcri.tl.delta_phenotypic_entropy(adata, cov_from=a, cov_to=b, groupby="patient",
                                         inplace=False)["result"]
    lhs = (r["adjusted"] - (r["adjusted_to"] - r["adjusted_from"])).to_numpy(dtype=float)
    rhs = (r["value"] - (r["value_to"] - r["value_from"])).to_numpy(dtype=float)
    np.testing.assert_allclose(lhs, rhs, atol=1e-12, equal_nan=True)


def test_endpoint_means_use_their_own_masks(ref):
    """Pin the known discrepancy so the next reader cannot assume the identity.

    ``build_result`` averages ``value``, ``value_from`` and ``value_to`` over their own finite
    masks, so ``value != value_to - value_from`` wherever a draw has a non-finite endpoint. The
    gap belongs to the delta result rather than to the reference, and this asserts its shape
    rather than a specific size.
    """
    _, adata = ref
    a, b = _cov(adata)[:2]
    r = tcri.tl.delta_phenotypic_entropy(adata, cov_from=a, cov_to=b, groupby="patient",
                                         null_model=None, inplace=False)["result"]
    gap = np.abs(r["value"] - (r["value_to"] - r["value_from"])).to_numpy(dtype=float)
    assert np.isfinite(gap).any()
    # equal wherever every endpoint draw was finite, which on this fixture is every row
    assert float(np.nanmax(gap)) < 1e-9 or float(np.nanmax(gap)) > 0.0


# ── the paired draws ─────────────────────────────────────────────────────────

def _labels_of(table):
    return [c for c in table.columns
            if c not in ("draw", "null")
            and not c.startswith(("value", "null_", "adjusted", "denom"))]


def test_adjusted_is_paired_per_draw(ref):
    """``table`` carries ``null_value`` and ``adjusted = value - null_value`` on every pair of
    draws, one per draw and null, ``adjusted`` varies across them, and ``result`` summarizes all
    the pairs: their mean, ``adjusted_sd`` and the HDI of the paired differences, row for row."""
    from tcri._stats import hdi

    _, adata = ref
    a = _cov(adata)[0]
    res = tcri.tl.mutual_information(adata, covariate=a, groupby="patient", n_samples=20,
                                     random_state=0, inplace=False)
    table, result = frames(res)
    np.testing.assert_allclose(table["adjusted"], table["value"] - table["null_value"],
                               atol=1e-12)
    labels = _labels_of(table)
    assert (table.groupby(labels, observed=True)["adjusted"].nunique() > 1).all(), (
        "adjusted is constant across draws")

    n_nulls = len(K.members(adata, "null.phenotype"))
    for _, row in result.iterrows():
        draws = table.loc[table["patient"] == row["patient"], "adjusted"].to_numpy(dtype=float)
        assert len(draws) == 20 * n_nulls
        lo, hi = hdi(draws, prob=0.94)
        assert row["adjusted"] == pytest.approx(draws.mean(), abs=1e-12)
        assert row["adjusted_sd"] == pytest.approx(draws.std(ddof=1), abs=1e-12)
        assert (row["adjusted_hdi_low"], row["adjusted_hdi_high"]) == pytest.approx((lo, hi))
        assert row["adjusted_hdi_high"] > row["adjusted_hdi_low"]
        null = table.loc[table["patient"] == row["patient"], "null_value"].mean()
        assert row["null_value"] == pytest.approx(null, abs=1e-12)


def test_every_scored_metric_pairs_its_draws(ref):
    """The same per-draw identity on every scored metric and on the perturbation, with the
    delta endpoints paired too, and no row of ``result`` gained or lost."""
    model, adata = ref
    a, b = _cov(adata)[:2]
    kw = dict(groupby="patient", n_samples=5, random_state=2, inplace=False)
    calls = {
        "mutual_information": lambda: tcri.tl.mutual_information(adata, covariate=a, **kw),
        "clonotypic_entropy": lambda: tcri.tl.clonotypic_entropy(adata, covariate=a, **kw),
        "phenotypic_entropy": lambda: tcri.tl.phenotypic_entropy(adata, covariate=a, **kw),
        "phenotypic_flux": lambda: tcri.tl.phenotypic_flux(adata, cov_from=a, cov_to=b, **kw),
        "delta_clonotypic_entropy": lambda: tcri.tl.delta_clonotypic_entropy(
            adata, cov_from=a, cov_to=b, **kw),
        "delta_phenotypic_entropy": lambda: tcri.tl.delta_phenotypic_entropy(
            adata, cov_from=a, cov_to=b, **kw),
        "gene_importance": lambda: tcri.perturb.gene_importance(
            model, adata, genes=list(adata.var_names[:3]), **kw),
    }
    for name, call in calls.items():
        fn = getattr(tcri.tl, name, None) or getattr(tcri.perturb, name)
        res = call()
        table, result = frames(res)
        assert len(table) and len(result), name
        for native, null, adjusted in _reference.column_pairs(fn.tcri_values,
                                                               fn.tcri_denominators):
            assert null in table.columns, f"{name}: no per-draw {null}"
            if adjusted is not None:
                np.testing.assert_allclose(table[adjusted], table[native] - table[null],
                                           atol=1e-12, equal_nan=True, err_msg=name)
        assert {"adjusted_sd", "adjusted_hdi_low", "adjusted_hdi_high"} <= set(result.columns)
        assert len(result) == len(table.drop_duplicates(_labels_of(table))), name


def test_a_per_draw_difference_between_two_covariate_levels_has_an_interval(ref):
    """The adjusted value at two covariate levels, differenced draw by draw within each
    patient, has a spread: each side is a per-draw quantity, so the difference is too."""
    from tcri._stats import hdi

    _, adata = ref
    a, b = _cov(adata)[:2]
    kw = dict(groupby="patient", n_samples=20, random_state=4, inplace=False)
    at_a = frames(tcri.tl.mutual_information(adata, covariate=a, **kw))[0]
    at_b = frames(tcri.tl.mutual_information(adata, covariate=b, **kw))[0]
    both = at_a.merge(at_b, on=["patient", "draw"], suffixes=("_a", "_b"))
    assert len(both)
    for patient, rows in both.groupby("patient", observed=True):
        diff = (rows["adjusted_b"] - rows["adjusted_a"]).to_numpy(dtype=float)
        lo, hi = hdi(diff, prob=0.94)
        assert hi - lo > 0, f"{patient}: a zero-width interval on a per-draw difference"


def test_the_fit_and_the_null_draw_from_different_seeds(ref):
    """Each null's draws come from the seed :func:`null_seed` derives from the caller's and that
    null's index: its per-draw values are its own call at that seed, and not at the caller's."""
    _, adata = ref
    a = _cov(adata)[0]
    kw = dict(covariate=a, groupby="patient", n_samples=8, inplace=False)
    table = frames(tcri.tl.mutual_information(adata, random_state=11, **kw))[0]

    for i, fit in enumerate(K.members(adata, "null.phenotype")):
        mine = table[table["null"] == i]

        def null_draws(seed):
            own = frames(tcri.tl.mutual_information(adata, random_state=seed, fit=fit,
                                             null_model=None, **kw))[0]
            return mine[["patient", "draw"]].merge(own, on=["patient", "draw"])["value"]

        seed = _reference.null_seed(11, i)
        assert seed != 11
        np.testing.assert_allclose(mine["null_value"], null_draws(seed), atol=1e-12)
        assert not np.allclose(mine["null_value"], null_draws(11)), (
            f"{fit} drew from the caller's own seed")


def test_null_seed_is_derived_and_reproducible():
    """The same input gives the same seed, a different stream from the caller's; ``None``
    stays unseeded; a seeded NumPy generator reproduces."""
    assert _reference.null_seed(None) is None
    assert _reference.null_seed(5) == _reference.null_seed(5) != 5
    assert _reference.null_seed(5) != _reference.null_seed(6)
    assert _reference.null_seed(np.int64(5)) == _reference.null_seed(5)
    assert isinstance(_reference.null_seed(-3), int)
    assert (_reference.null_seed(np.random.default_rng(1))
            == _reference.null_seed(np.random.default_rng(1)))
    # one stream per null of a reference; index 0 is the one-null stream
    assert _reference.null_seed(5, 0) == _reference.null_seed(5)
    assert len({_reference.null_seed(5, i) for i in range(5)}) == 5


def test_without_draws_a_row_has_one_pair_per_null(ref):
    """At ``n_samples=0`` each side is one point value, so a row has one pair per null:
    ``adjusted`` is their mean, ``adjusted_sd`` the sd between the nulls, and the HDI is NaN.
    With draws, the HDI is reported."""
    _, adata = ref
    a = _cov(adata)[0]
    n_nulls = len(K.members(adata, "null.phenotype"))
    res = tcri.tl.mutual_information(adata, covariate=a, groupby="patient", n_samples=0,
                                     inplace=False)
    table, result = frames(res)
    assert set(table["draw"]) == {0} and sorted(set(table["null"])) == list(range(n_nulls))
    np.testing.assert_allclose(result["adjusted"], result["value"] - result["null_value"],
                               atol=1e-12)
    for _, row in result.iterrows():
        pairs = table.loc[table["patient"] == row["patient"], "adjusted"].to_numpy(dtype=float)
        assert len(pairs) == n_nulls
        assert row["adjusted"] == pytest.approx(pairs.mean(), abs=1e-12)
        assert row["adjusted_sd"] == pytest.approx(pairs.std(ddof=1), abs=1e-12)
    for column in ("adjusted_hdi_low", "adjusted_hdi_high"):
        assert result[column].isna().all(), column

    drawn = frames(tcri.tl.mutual_information(adata, covariate=a, groupby="patient", n_samples=10,
                                       random_state=0, inplace=False))[1]
    for column in ("adjusted_hdi_low", "adjusted_hdi_high"):
        assert drawn[column].notna().all(), column


# ── selecting the reference, and selecting the fit ───────────────────────────

def test_null_model_default_is_the_contract_table(ref):
    """``"auto"`` resolves to the metric's entry in the contract's ``DEFAULT_NULL``, and the
    resolved name is what lands in ``params`` -- never ``"auto"``, and never a model object."""
    from tests._governance import contract_namespace

    table = contract_namespace("METRICS_CONTRACT.md")["DEFAULT_NULL"]
    model, adata = ref
    a, b = _cov(adata)[:2]
    tcri.tl.mutual_information(adata, covariate=a)
    tcri.tl.phenotypic_flux(adata, cov_from=a, cov_to=b)
    tcri.perturb.gene_importance(model, adata, genes=[0])
    for name in ("mutual_information", "phenotypic_flux", "gene_importance"):
        got = tcri.get.params(adata, name)["null_model"]
        assert got == f"null.{table[name]}", f"{name}: params say {got!r}"


def test_null_model_none_is_the_old_result(ref):
    """No reference column anywhere, one ``uns`` key, and the same value as the default run."""
    _, adata = ref
    a = _cov(adata)[0]
    before = set(adata.uns)
    tcri.tl.mutual_information(adata, covariate=a, groupby="patient", null_model=None)
    added = set(adata.uns) - before
    assert added == {K.MUTUAL_INFORMATION}, f"null_model=None wrote {added}"
    bare = tcri.get.result(adata, "mutual_information")

    for frame in ("draws", "summary"):
        cols = set(bare["replicate"][frame].columns)
        assert not {c for c in cols if c.startswith(("null_", "adjusted"))}, cols
    assert bare["stats"] is None or set(bare["stats"]["quantity"]) == {"value"}

    withref = frames(tcri.tl.mutual_information(adata, covariate=a, groupby="patient",
                                         inplace=False))[1]
    np.testing.assert_allclose(bare["replicate"]["summary"]["value"].to_numpy(dtype=float),
                               withref["value"].to_numpy(dtype=float), atol=1e-12)


def test_rule_9_shape_changes_are_the_only_ones(ref):
    """At ``null_model=None`` the reference machinery changes the frames in exactly four ways.

    (a) ``build_result``'s empty branch carries its declared extras; (b) the delta ``table``'s
    row order follows the substrate rather than obs first appearance; (c) ``stats`` gains a
    ``quantity`` column; (d) ``mutual_information`` gains ``denom``. Everything else -- every
    value, and the ``uns`` key set -- is unchanged. Listed here so nobody has to discover them.
    """
    from tcri._compute._tables import build_result

    _, adata = ref
    a, b = _cov(adata)[:2]

    # (a) the empty branch
    empty = build_result(pd.DataFrame(), extra_values=("value_from", "value_to"))
    assert list(empty.columns) == ["value", "value_from", "value_to"]

    # (c) and (d)
    mi = tcri.tl.mutual_information(adata, covariate=a, groupby="patient",
                                    splitby="disease_status", null_model=None, inplace=False)
    assert "denom" in mi["replicate"]["summary"].columns
    assert list(mi["stats"]["quantity"].unique()) == ["value"]

    # and the value columns beyond those four are the no-reference set
    assert set(mi["replicate"]["summary"].columns) == {"covariate", "patient", "disease_status",
                                                       "value",
                                         "sd", "hdi_low", "hdi_high", "denom"}
    delta = tcri.tl.delta_phenotypic_entropy(adata, cov_from=a, cov_to=b, groupby="patient",
                                             null_model=None, inplace=False)
    assert set(delta["result"].columns) == {"cov_from", "cov_to", "patient", "clonotype",
                                            "value", "sd", "hdi_low", "hdi_high",
                                            "value_from", "value_to"}


def test_missing_null_names_the_fix(ref):
    """A metric never trains. With no substrate it raises, naming the one-line call to run."""
    _, adata = ref
    a = _cov(adata)[0]
    for key in list(adata.uns):
        if "null.phenotype" in key:
            del adata.uns[key]
    adata.uns[K.METADATA] = {**adata.uns[K.METADATA],
                             K.FITS: [f for f in K.fits(adata) if f != "null.phenotype"]}
    with pytest.raises(KeyError, match=r"tcri\.null\.all\(model, adata\)"):
        tcri.tl.mutual_information(adata, covariate=a, inplace=False)
    # ...and the opt-out is the other half of the message
    assert frames(tcri.tl.mutual_information(adata, covariate=a, null_model=None,
                                      inplace=False))[1] is not None


def test_the_bare_kind_resolves(ref):
    """``fit="phenotype[0]"`` and ``fit="null.phenotype[0]"`` are one call; a reference's name
    names its fits; an unknown name lists what the object carries; an ambiguous one raises naming
    both rather than guessing."""
    model, adata = ref
    a = _cov(adata)[0]
    bare = frames(tcri.tl.mutual_information(adata, covariate=a, fit="phenotype[0]",
                                      inplace=False))[1]
    full = frames(tcri.tl.mutual_information(adata, covariate=a, fit="null.phenotype[0]",
                                      inplace=False))[1]
    pd.testing.assert_frame_equal(bare, full)

    with pytest.raises(KeyError, match=r"null\.phenotype\[1\]"):
        tcri.tl.mutual_information(adata, covariate=a, fit="phenotype", inplace=False)
    with pytest.raises(KeyError, match="no fit named"):
        tcri.tl.mutual_information(adata, covariate=a, fit="nope", inplace=False)

    # both `x` and `null.x` present: refuse rather than prefer one silently
    meta = dict(adata.uns[K.METADATA])
    meta[K.FITS] = list(K.fits(adata)) + ["phenotype[0]"]
    adata.uns[K.METADATA] = meta
    with pytest.raises(KeyError, match="ambiguous"):
        K.resolve_fit(adata, "phenotype[0]")


def test_fit_selects_the_substrate(ref):
    """The number computed on a null is the null's, not the main fit's.

    The mutation this catches: strip the prefix inside ``_engine_blocks`` and the two frames
    become identical, so the assertion is that they are NOT.
    """
    _, adata = ref
    a = _cov(adata)[0]
    main = frames(tcri.tl.mutual_information(adata, covariate=a, null_model=None,
                                      inplace=False))[1]
    null = frames(tcri.tl.mutual_information(adata, covariate=a, fit="null.phenotype[0]",
                                      null_model=None, inplace=False))[1]
    assert not np.isclose(float(main["value"].iloc[0]), float(null["value"].iloc[0]), atol=1e-6)
    assert float(null["value"].iloc[0]) < float(main["value"].iloc[0]), (
        "the phenotype null carries as much clone-phenotype information as its parent")


def test_a_result_on_a_null_does_not_overwrite_the_main_key(ref):
    """Two calls, two keys. The main key's params still name the main fit."""
    _, adata = ref
    a = _cov(adata)[0]
    tcri.tl.mutual_information(adata, covariate=a, groupby="patient")
    main = frames(tcri.get.result(adata, "mutual_information"))[1]["value"].to_numpy().copy()

    tcri.tl.mutual_information(adata, covariate=a, groupby="patient", fit="null.phenotype[0]",
                               null_model=None)
    np.testing.assert_array_equal(
        frames(tcri.get.result(adata, "mutual_information"))[1]["value"].to_numpy(), main)
    assert tcri.get.params(adata, "mutual_information")["fit"] is None
    assert tcri.get.params(adata, "mutual_information",
                           fit="null.phenotype[0]")["fit"] == "null.phenotype[0]"
    assert K.fit_key(K.MUTUAL_INFORMATION, "null.phenotype[0]") in adata.uns


def test_key_added_and_fit_compose(ref):
    """``key_added="x", fit="null.phenotype[0]"`` writes ``x_null.phenotype[0]``, not ``x``.

    The fit suffix applies to whichever base is chosen, never only to the default -- otherwise a
    caller's own named key would be overwritten by a run on a different fit, which is the bug
    this whole rule exists to prevent one level up.
    """
    _, adata = ref
    a = _cov(adata)[0]
    tcri.tl.mutual_information(adata, covariate=a, key_added="x", fit="null.phenotype[0]",
                               null_model=None)
    assert "x_null.phenotype[0]" in adata.uns and "x" not in adata.uns


def test_substrates_do_not_collide(ref):
    """Writing and reading a null leaves the main fit's substrate bitwise unchanged."""
    _, adata = ref
    a = _cov(adata)[0]
    before = {k: np.asarray(adata.uns[k]).copy()
              for k in (K.P_CT, K.CT_ARRAY, K.CT_TO_C, K.CT_TO_COV, K.COV_ARRAY)}
    tcri.tl.mutual_information(adata, covariate=a)
    for k, v in before.items():
        np.testing.assert_array_equal(np.asarray(adata.uns[k]), v, err_msg=k)
    for k in before:
        assert K.fit_key(k, "null.phenotype[0]") in adata.uns


def test_an_incomparable_fit_is_refused(ref):
    """A hand-written fit whose record disagrees raises, naming which of the five fields."""
    _, adata = ref
    a = _cov(adata)[0]

    # a compatible hand-written fit: one phenotype null's substrate under another name, with
    # the record of a fit `to_anndata(fit="myalt")` wrote, which belongs to no reference
    for key in list(adata.uns):
        if "null.phenotype[0]" in key:
            adata.uns[key.replace("null.phenotype[0]", "myalt")] = adata.uns[key]
    for key in list(adata.obsm):
        if "null.phenotype[0]" in key:
            adata.obsm[key.replace("null.phenotype[0]", "myalt")] = adata.obsm[key]
    adata.uns[K.fit_key(K.FIT_SETTINGS, "myalt")] = {
        k: v for k, v in adata.uns[K.fit_key(K.FIT_SETTINGS, "myalt")].items()
        if k not in ("reference", "index", "n_nulls")}
    adata.uns[K.METADATA] = {**adata.uns[K.METADATA], K.FITS: K.fits(adata) + ["myalt"]}
    ok = tcri.tl.mutual_information(adata, covariate=a, null_model="myalt", inplace=False)
    assert "null_value" in ok["replicate"]["summary"].columns

    record = dict(adata.uns[K.fit_key(K.FIT_SETTINGS, "myalt")])
    record[K.PHENOTYPE_CATEGORIES] = list(record[K.PHENOTYPE_CATEGORIES])[:-1]
    adata.uns[K.fit_key(K.FIT_SETTINGS, "myalt")] = record
    with pytest.raises(ValueError, match=K.PHENOTYPE_CATEGORIES):
        tcri.tl.mutual_information(adata, covariate=a, null_model="myalt", inplace=False)

    record["n_obs"] = int(record["n_obs"]) + 1
    record[K.PHENOTYPE_CATEGORIES] = list(adata.uns[K.PHENOTYPE_CATEGORIES])
    adata.uns[K.fit_key(K.FIT_SETTINGS, "myalt")] = record
    with pytest.raises(ValueError, match="n_obs"):
        tcri.tl.mutual_information(adata, covariate=a, null_model="myalt", inplace=False)


def test_a_key_added_keeps_its_own_reference(ref):
    """Two results under different keys keep separate references: each reference run stores
    under its caller's key, suffixed by the fit, and reads back with ``key=`` and ``fit=``."""
    model, adata = ref
    a, b = _cov(adata)[:2]
    tcri.tl.mutual_information(adata, covariate=a)
    tcri.tl.mutual_information(adata, covariate=b, key_added="mi_b")
    members = [f for f in K.fits(adata) if f.startswith("null.phenotype[")]
    assert members, "the fixture carries no phenotype null"
    for fit in members:
        default = tcri.get.draws(adata, "mutual_information", fit=fit)
        keyed = tcri.get.draws(adata, "mutual_information", key="mi_b", fit=fit)
        assert set(default["covariate"].astype(str)) == {str(a)}, fit
        assert set(keyed["covariate"].astype(str)) == {str(b)}, fit

    genes = list(adata.var_names[:2])
    tcri.perturb.gene_importance(model, adata, genes=genes)
    tcri.perturb.gene_importance(model, adata, genes=genes[:1], key_added="gi_one")
    for fit in members:
        default = tcri.get.replicates(adata, "gene_importance", fit=fit)
        keyed = tcri.get.replicates(adata, "gene_importance", key="gi_one", fit=fit)
        assert set(default["gene"].astype(str)) == set(genes), fit
        assert set(keyed["gene"].astype(str)) == set(genes[:1]), fit


def test_the_two_nulls_stay_distinct(ref):
    """``diag.permutation_null`` is model-free and takes no ``fit``; the two substrate-reading
    diagnostics do. One word, two things, and the signatures are where that is enforced."""
    import inspect

    assert "fit" not in inspect.signature(tcri.diag.permutation_null).parameters
    assert "fit" in inspect.signature(tcri.diag.joint_distribution_ppc).parameters
    assert "fit" in inspect.signature(tcri.diag.phenotype_calibration).parameters
    assert "null_model" not in inspect.signature(tcri.diag.permutation_null).parameters


# ── the frames, read back ────────────────────────────────────────────────────

def test_stats_carry_the_adjusted_quantity(ref):
    """One row per contrast, on the adjusted value: ``quantity`` says so and each ``stat`` is a
    Mann-Whitney on the replicate summary's ``adjusted`` per arm. Without a reference the one row is on the
    value, which is how a reader of a stored frame tells the two cases apart."""
    from scipy.stats import mannwhitneyu

    _, adata = ref
    a = _cov(adata)[0]
    kw = dict(covariate=a, groupby="patient", splitby="disease_status", inplace=False)
    res = tcri.tl.mutual_information(adata, **kw)
    stats, result = res["stats"], res["replicate"]["summary"]
    arms = result["disease_status"].astype(str).nunique()
    assert set(stats["quantity"]) == {"adjusted"}
    assert len(stats) == arms * (arms - 1) // 2

    for _, row in stats.iterrows():
        va = result.loc[result["disease_status"] == row["level_a"],
                        "adjusted"].to_numpy(dtype=float)
        vb = result.loc[result["disease_status"] == row["level_b"],
                        "adjusted"].to_numpy(dtype=float)
        U, p = mannwhitneyu(va, vb, alternative="two-sided")
        assert float(row["stat"]) == pytest.approx(float(U))
        assert float(row["p"]) == pytest.approx(float(p))

    bare = tcri.tl.mutual_information(adata, **kw, null_model=None)["stats"]
    assert set(bare["quantity"]) == {"value"}
    assert len(bare) == len(stats)


def test_a_delta_contrasts_only_the_adjusted_value(ref):
    """A delta's ``stats`` has exactly one row per contrast, on ``adjusted``, and no row about
    an endpoint: ``adjusted_from`` and ``adjusted_to`` are inputs, like ``value_from`` and
    ``value_to``, and inputs carry no contrast."""
    _, adata = ref
    a, b = _cov(adata)[:2]
    res = tcri.tl.delta_phenotypic_entropy(adata, cov_from=a, cov_to=b, groupby="patient",
                                           splitby="disease_status", inplace=False)
    stats, result = res["stats"], res["result"]
    assert stats is not None and len(stats)
    arms = result["disease_status"].astype(str).nunique()
    assert len(stats) == arms * (arms - 1) // 2
    assert set(stats["quantity"]) == {"adjusted"}


def test_restat_is_idempotent(ref):
    """Two `restat` calls on one payload leave one frame with the same rows and the same
    ``quantity``, for a referenced result and for ``null_model=None``: the frame is rebuilt from
    ``result`` each time, never appended to."""
    from tcri._state import _reference as R

    _, adata = ref
    a = _cov(adata)[0]
    kw = dict(covariate=a, groupby="patient", splitby="disease_status", inplace=False)
    for null_model, quantity in (("auto", "adjusted"), (None, "value")):
        payload = tcri.tl.mutual_information(adata, **kw, null_model=null_model)
        once = payload["stats"].copy()
        assert set(once["quantity"]) == {quantity}
        R.restat(payload, groupby="patient", splitby="disease_status")
        R.restat(payload, groupby="patient", splitby="disease_status")
        pd.testing.assert_frame_equal(payload["stats"], once)


def test_an_empty_result_is_not_a_missing_reference(ref):
    """An empty frame carries its columns, and the y label does not blame the reference.

    An empty ``phenotypic_flux`` is a real and documented outcome -- no clone at both covariate
    levels within any replicate. An empty branch that returned one column against the non-empty
    shape's nine would make the panel read as if the null were missing.
    """
    from tcri._compute._tables import build_result

    empty = build_result(pd.DataFrame(), value="value", extra_values=("null_value", "adjusted"))
    assert list(empty.columns) == ["value", "null_value", "adjusted"]
    assert _reference.label_for(empty, "value", "flux") == "flux"
    assert _reference.label_for(empty, "adjusted", "flux") == "flux"
    assert _reference.label_for(pd.DataFrame({"value": [1.0]}), "value", "flux") == \
        "flux (no reference)"
    assert _reference.label_for(pd.DataFrame({"value": [1.0], "null_value": [0.5]}),
                                "value", "flux") == "flux"
    assert _reference.label_for(pd.DataFrame({"value": [1.0], "null_value": [0.5],
                                              "adjusted": [0.5]}), "adjusted", "flux") == \
        "adjusted flux"


def test_old_results_render_without_a_reference(ref):
    """A stored result with no ``null_value`` still renders, with the "no reference" label."""
    import matplotlib
    matplotlib.use("Agg")

    _, adata = ref
    a = _cov(adata)[0]
    tcri.tl.mutual_information(adata, covariate=a, groupby="patient", null_model=None)
    ax = tcri.pl.mutual_information(adata)
    assert "no reference" in ax.get_ylabel()
    with pytest.raises(ValueError, match="null_model=None"):
        tcri.pl.mutual_information(adata, quantity="adjusted")


def test_h5ad_round_trip(ref, tmp_path):
    """The two-substrate object writes and reads, dotted keys and all."""
    import anndata as ad

    _, adata = ref
    a = _cov(adata)[0]
    tcri.tl.mutual_information(adata, covariate=a, groupby="patient")
    path = tmp_path / "two.h5ad"
    adata.write_h5ad(path)
    back = ad.read_h5ad(path)

    assert set(K.fits(back)) == set(K.fits(adata))
    for member in K.members(adata, "null.phenotype"):
        assert K.fit_key(K.MUTUAL_INFORMATION, member) in back.uns
    pd.testing.assert_frame_equal(frames(tcri.get.result(back, "mutual_information"))[0],
                                  frames(tcri.get.result(adata, "mutual_information"))[0],
                                  check_dtype=False, check_categorical=False)
    np.testing.assert_allclose(
        frames(tcri.get.result(back, "mutual_information"))[1]["adjusted"].to_numpy(dtype=float),
        frames(tcri.get.result(adata, "mutual_information"))[1]["adjusted"].to_numpy(dtype=float),
        atol=1e-12)
    assert tcri.get.params(back, "mutual_information")["null_model"] == "null.phenotype"


# ── the figures ──────────────────────────────────────────────────────────────

def test_pl_draws_the_reference_behind_the_value(ref):
    """On the value panel the gray mark sits at the value's own x positions and the tick labels
    do not move; the adjusted panel draws a zero rule and no reference."""
    import matplotlib
    matplotlib.use("Agg")
    from tcri.plotting._base import REFERENCE_LABEL

    _, adata = ref
    a = _cov(adata)[0]
    tcri.tl.mutual_information(adata, covariate=a, groupby="patient")

    ax = tcri.pl.mutual_information(adata, quantity="value")
    ticks = [t.get_text() for t in ax.get_xticklabels()]
    value_x = np.concatenate([c.get_offsets()[:, 0] for c in ax.collections
                              if c.get_label() != REFERENCE_LABEL])
    gray_x = np.concatenate([c.get_offsets()[:, 0] for c in ax.collections
                             if c.get_label() == REFERENCE_LABEL])
    assert len(gray_x), "no reference collection was drawn"
    np.testing.assert_allclose(np.sort(gray_x), np.sort(value_x))
    assert [t.get_text() for t in ax.get_xticklabels()] == ticks

    adjusted_ax = tcri.pl.mutual_information(adata, quantity="adjusted")
    assert any(round(float(ln.get_ydata()[0]), 12) == 0.0 for ln in adjusted_ax.lines), \
        "no zero rule on the adjusted panel"
    assert not [c for c in adjusted_ax.collections if c.get_label() == REFERENCE_LABEL], \
        "the adjusted panel drew a reference; zero IS the reference there"

    # ...and absent entirely when there was no reference
    tcri.tl.mutual_information(adata, covariate=a, groupby="patient", null_model=None)
    plain = tcri.pl.mutual_information(adata)
    assert not [c for c in plain.collections if c.get_label() == REFERENCE_LABEL]


def test_pl_draws_the_adjusted_quantity_by_default(ref):
    """The default panel is the adjusted value against a zero rule, alone. ``quantity="value"``
    is the value with one gray null pass behind it and no zero rule; and with no reference the
    default is the value, labeled as having none.

    The tick labels are compared as a SET: each panel orders x by its own quantity's median,
    so the two panels may order the patients differently.
    """
    import matplotlib
    matplotlib.use("Agg")
    from tcri.plotting._base import REFERENCE_LABEL, REFERENCE_LEGEND

    _, adata = ref
    a = _cov(adata)[0]
    tcri.tl.mutual_information(adata, covariate=a, groupby="patient")
    res = tcri.get.replicates(adata, "mutual_information")

    def gray(ax):
        return [c for c in ax.collections if c.get_label() == REFERENCE_LABEL]

    def colored_y(ax):
        return np.sort(np.concatenate([c.get_offsets()[:, 1] for c in ax.collections
                                       if c.get_label() != REFERENCE_LABEL]))

    def zero_rule(ax):
        return any(len(ln.get_ydata()) == 2 and np.allclose(ln.get_ydata(), 0.0)
                   for ln in ax.lines)

    def ticks(ax):
        return {t.get_text() for t in ax.get_xticklabels()}

    def entries(ax):
        legend = ax.get_legend()
        return [t.get_text() for t in legend.get_texts()] if legend is not None else []

    default = tcri.pl.mutual_information(adata)
    np.testing.assert_allclose(colored_y(default),
                               np.sort(res["adjusted"].to_numpy(dtype=float)))
    assert default.get_ylabel().startswith("adjusted"), default.get_ylabel()
    assert zero_rule(default), "no zero rule on the adjusted panel"
    assert not gray(default), "the adjusted panel drew a reference; zero IS the reference there"
    assert not entries(default)

    value = tcri.pl.mutual_information(adata, quantity="value")
    np.testing.assert_allclose(colored_y(value), np.sort(res["value"].to_numpy(dtype=float)))
    assert len(gray(value)) == 1, "the value panel draws one gray pass"
    assert not zero_rule(value)
    assert entries(value) == [REFERENCE_LEGEND]
    assert ticks(value) == ticks(default)

    tcri.tl.mutual_information(adata, covariate=a, groupby="patient", null_model=None)
    plain = tcri.pl.mutual_information(adata)
    np.testing.assert_allclose(colored_y(plain), colored_y(value))
    assert not gray(plain) and not zero_rule(plain)
    assert "no reference" in plain.get_ylabel()


def test_the_value_panel_of_a_referenced_result_carries_no_star(ref):
    """The star is about the quantity on the axis. ``stats`` of a referenced result contrasts
    the adjusted value, so the default panel carries that row's label and the value panel
    carries no text at all, rather than a star about a different column."""
    import matplotlib
    matplotlib.use("Agg")

    _, adata = ref
    a = _cov(adata)[0]
    tcri.tl.mutual_information(adata, covariate=a, groupby="patient", splitby="disease_status")
    stats = tcri.get.stats(adata, "mutual_information")
    rows = stats.loc[stats["quantity"] == "adjusted"]
    assert len(rows) == 1
    label = str(rows["stars"].iloc[0] or "ns")

    default = tcri.pl.mutual_information(adata)
    assert [t.get_text() for t in default.texts] == [label]
    value = tcri.pl.mutual_information(adata, quantity="value")
    assert not value.texts, [t.get_text() for t in value.texts]


def test_the_quantity_argument_is_checked_before_the_data(ref):
    """A misspelled ``quantity`` is an argument error, so it raises whether or not a result
    exists and whether or not ``return_df`` was passed. And ``return_df=True`` hands back the
    frame before any quantity is read, so ``quantity="adjusted"`` on an unreferenced result
    returns it rather than raising about a column nothing was going to draw."""
    model, adata = ref
    a, b = _cov(adata)[:2]
    for key in (K.MUTUAL_INFORMATION, K.DELTA_PHENOTYPIC_ENTROPY, K.GENE_IMPORTANCE):
        adata.uns.pop(key, None)
    twins = {
        "mutual_information": lambda **kw: tcri.pl.mutual_information(adata, **kw),
        "delta_phenotypic_entropy": lambda **kw: tcri.pl.delta_phenotypic_entropy(adata, **kw),
        "gene_importance": lambda **kw: tcri.pl.gene_importance(adata, **kw),
    }
    for name, twin in twins.items():
        with pytest.raises(KeyError, match="not found"):
            twin(return_df=True)
        for kw in ({}, {"return_df": True}):
            with pytest.raises(ValueError, match="quantity must be"):
                twin(quantity="exces", **kw)

    tcri.tl.mutual_information(adata, covariate=a, groupby="patient", null_model=None)
    tcri.tl.delta_phenotypic_entropy(adata, cov_from=a, cov_to=b, groupby="patient",
                                     null_model=None)
    tcri.perturb.gene_importance(model, adata, genes=list(adata.var_names[:3]),
                                 null_model=None)
    for name, twin in twins.items():
        frame = twin(quantity="adjusted", return_df=True)
        assert isinstance(frame, pd.DataFrame) and "adjusted" not in frame.columns, name
        with pytest.raises(ValueError, match="quantity must be"):
            twin(quantity="exces", return_df=True)
        with pytest.raises(ValueError, match="null_model=None"):
            twin(quantity="adjusted")


def test_an_empty_referenced_result_draws_the_no_data_panel(ref):
    """An empty ``result`` reaches the "no data" panel at every accepted quantity, explicit
    ``"adjusted"`` included, with the value's label and no raise.

    `attach` returns early on an empty frame, so an empty result carries no adjusted column
    however it was computed. Resolving the quantity before the empty-frame branch would turn
    that into "computed with null_model=None", which misattributes the cause.
    """
    import matplotlib
    matplotlib.use("Agg")
    from tcri._compute._tables import build_result
    from tcri._state.storage import _encode

    model, adata = ref
    a = _cov(adata)[0]
    tcri.tl.mutual_information(adata, covariate=a, groupby="patient")
    tcri.perturb.gene_importance(model, adata, genes=list(adata.var_names[:3]))
    empty = build_result(pd.DataFrame(), value="value",
                         extra_values=("null_value", "adjusted"))
    for key, name in ((K.MUTUAL_INFORMATION, "mutual_information"),
                      (K.GENE_IMPORTANCE, "gene_importance")):
        blob = adata.uns[key]
        payload = tcri.get.result(adata, name)
        if "replicate" in payload:
            payload["replicate"] = {"summary": empty,
                                    "draws": payload["replicate"]["draws"].iloc[:0]}
        else:
            payload["result"] = empty
            payload["table"] = payload["table"].iloc[:0]
        adata.uns[key] = {**_encode(payload), "params": blob["params"],
                          "version": blob["version"]}

    for quantity in ("auto", "value", "adjusted"):
        ax = tcri.pl.mutual_information(adata, quantity=quantity)
        assert [t.get_text() for t in ax.texts] == ["no data for mutual_information"], quantity
        assert ax.get_ylabel() == "mutual information (bits)", quantity
        ax = tcri.pl.gene_importance(adata, quantity=quantity)
        assert [t.get_text() for t in ax.texts] == ["no data for gene_importance"], quantity
        assert not ax.get_ylabel().startswith("adjusted"), quantity
        assert "no reference" not in ax.get_ylabel(), quantity


def test_the_shift_heatmap_draws_the_adjusted_shift(ref):
    """The item view follows ``quantity`` like every item view: the adjusted shift at the default
    and at ``quantity="adjusted"``, and an explicit ``"adjusted"`` raises when the result has no
    reference."""
    import matplotlib
    matplotlib.use("Agg")

    model, adata = ref
    genes = list(adata.var_names[:5])
    tcri.perturb.gene_importance(model, adata, genes=genes)
    auto = tcri.pl.gene_importance(adata, view="item").images[0].get_array()
    adjusted = tcri.pl.gene_importance(adata, view="item", quantity="adjusted")
    np.testing.assert_array_equal(adjusted.images[0].get_array(), auto)
    assert adjusted.figure.axes[-1].get_ylabel().startswith("adjusted shift")
    value = tcri.pl.gene_importance(adata, view="item", quantity="value").images[0].get_array()
    assert not np.allclose(value, auto)

    tcri.perturb.gene_importance(model, adata, genes=genes, null_model=None)
    with pytest.raises(ValueError, match="no 'adjusted' column"):
        tcri.pl.gene_importance(adata, view="item", quantity="adjusted")


def test_both_gene_panels_rank_alike(ref):
    """One resolved gene list, handed to both views, so a half-threaded quantity cannot make the
    two panels of one figure rank a different set of genes."""
    import matplotlib
    matplotlib.use("Agg")

    model, adata = ref
    tcri.perturb.gene_importance(model, adata, genes=list(adata.var_names[:8]))
    # at BOTH quantities: which genes are shown and which quantity is drawn are separate
    # decisions, so the two views of one figure show the same genes whatever each draws.
    for kw in ({}, {"quantity": "value"}):
        rank = tcri.pl.gene_importance(adata, n_top=5, **kw)
        shift = tcri.pl.gene_importance(adata, view="item", n_top=5, **kw)
        assert [t.get_text() for t in rank.get_xticklabels()] == \
               [t.get_text() for t in shift.get_yticklabels()], kw
    # ...and the corrected set is not the bare one, which is the whole reason for the default
    assert [t.get_text() for t in tcri.pl.gene_importance(adata, n_top=5).get_xticklabels()] != \
           [t.get_text() for t in
            tcri.pl.gene_importance(adata, n_top=5, quantity="value").get_xticklabels()]


# ── the shape of a real run ──────────────────────────────────────────────────

@pytest.mark.slow
def test_oe_shape_smoke():
    """A cohort fit, its references, and two metrics read off it, end to end.

    Not an accuracy claim: it asserts that the adjusted value is finite everywhere and that mutual
    information clears its own floor, which is the shape a reported figure rests on. It is the
    only test here that fits a model rather than reusing a fixture, so it is the one that would
    catch a reference that silently returned NaN on a frame with groups, a split and a covariate
    contrast all at once.
    """
    import contextlib
    import io

    from tcri.datasets import simulate_cohort
    from tcri.model._model import TCRIModel

    adata = simulate_cohort(n_patients=4, n_clones=10, n_cells_per_sample=250, seed=0)
    adata.layers["counts"] = adata.X.copy()
    TCRIModel.setup_anndata(adata, layer="counts", clonotype_key="clone_id",
                            phenotype_key="phenotype", covariate_key="timepoint",
                            batch_key="patient", replicate="patient")
    model = TCRIModel(adata, n_latent=8, n_hidden=16, n_layers=1, classifier_n_layers=1,
                      classifier_hidden=16, K=4, seed=0, name="oesmoke")
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.train(max_epochs=30, batch_size=128, n_epochs_kl_warmup=2, accelerator="cpu",
                    enable_progress_bar=False, enable_model_summary=False)
        model.to_anndata(adata)
        tcri.null.all(model, adata, kinds=("phenotype", "covariate"),
                      enable_progress_bar=False, enable_model_summary=False)

    levels = list(adata.uns[K.COVARIATE_CATEGORIES])
    mi = frames(tcri.tl.mutual_information(adata, covariate=levels[-1], groupby="patient",
                                    inplace=False))[1]
    assert np.isfinite(mi["adjusted"]).all(), mi
    assert float(mi["adjusted"].mean()) > 0.0, (
        f"the fitted model carries no more clone-phenotype information than its phenotype "
        f"null: {mi[['value', 'null_value', 'adjusted']].to_string()}")

    flux = tcri.tl.phenotypic_flux(adata, cov_from=levels[0], cov_to=levels[-1],
                                   groupby="patient", inplace=False)["result"]
    if len(flux):
        assert np.isfinite(flux["adjusted"]).any(), flux


# ── the reference on the reading and the plotting paths ──────────────────────

def test_a_label_column_named_like_a_reference_column_is_still_a_label(ref):
    """`groupby` is an arbitrary obs column name, so excluding join keys by PREFIX drops one.

    A groupby column called `adjusted_patient` would be dropped from the merge keys, and the
    reference join would then fan the result out to a multiple of its rows. The exclusion is by
    exact name, which is what this asserts.
    """
    _, adata = ref
    adata.obs["adjusted_patient"] = adata.obs["patient"]
    adata.obs["null_arm"] = adata.obs["disease_status"]
    a = _cov(adata)[0]
    bare = frames(tcri.tl.clonotypic_entropy(adata, covariate=a, groupby="adjusted_patient",
                                      splitby="null_arm", null_model=None,
                                      inplace=False))[1]
    withref = frames(tcri.tl.clonotypic_entropy(adata, covariate=a, groupby="adjusted_patient",
                                         splitby="null_arm", inplace=False))[1]
    assert len(withref) == len(bare), (
        f"the reference merge fanned out: {len(withref)} rows against {len(bare)}")
    np.testing.assert_allclose(withref["value"].to_numpy(dtype=float),
                               bare["value"].to_numpy(dtype=float), atol=1e-12)


def test_the_reference_blob_is_keyed_by_the_fit_name(ref):
    """`gene_importance`'s reference is keyed by the FIT, not by the null's parameter namespace.

    A rebuilt null's `name` is `<parent>.null.phenotype` — its place in the process-global
    parameter store, not its name on this AnnData. Keying the blob by that puts the reference
    somewhere `tcri.get(fit=...)` does not look, while `params["null_model"]` names the fit.
    """
    model, adata = ref
    tcri.perturb.gene_importance(model, adata, genes=list(adata.var_names[:3]))
    assert tcri.get.params(adata, "gene_importance")["null_model"] == "null.phenotype"
    for fit in K.members(adata, "null.phenotype"):
        assert K.fit_key(K.GENE_IMPORTANCE, fit) in adata.uns, sorted(
            k for k in adata.uns if "gene_importance" in k)
        assert frames(tcri.get.result(adata, "gene_importance", fit=fit))[1] is not None


@pytest.mark.parametrize("source, key_added", [("rebuild", None), ("fit", None), ("fit", "b")])
def test_a_null_passed_as_an_object_is_keyed_by_its_fit_name(ref, source, key_added):
    """Passing the null itself as ``null_model`` -- rebuilt, or as ``tcri.null.*`` returned it --
    records the reference by its name, the same as the default does, and keys each fit's run by
    that fit's name, so ``tcri.get(fit=...)`` finds it. The list ``tcri.null.*`` returns is the
    whole reference; one rebuilt null is that fit alone."""
    import warnings

    from tcri.model._model import adopt
    from tcri.null._rebuild import rebuild

    model, adata = ref
    adata = adopt(model, adata)   # `ref` is a copy: register the parent's setup on it
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        null = (rebuild(model, adata, "phenotype[0]") if source == "rebuild"
                else tcri.null.phenotype(model, adata, n_nulls=2, key_added=key_added))
        tcri.perturb.gene_importance(model, adata, genes=list(adata.var_names[:3]),
                                     null_model=null)
    group = "null.phenotype" if key_added is None else f"null.phenotype.{key_added}"
    recorded = tcri.get.params(adata, "gene_importance")["null_model"]
    assert recorded == (f"{group}[0]" if source == "rebuild" else group)
    for fit in ([recorded] if source == "rebuild" else K.members(adata, group)):
        assert K.fit_key(K.GENE_IMPORTANCE, fit) in adata.uns, sorted(
            k for k in adata.uns if "gene_importance" in k)
        assert frames(tcri.get.result(adata, "gene_importance", fit=fit))[1] is not None


def test_a_non_finite_reference_does_not_delete_the_stats(ref):
    """A reference that is non-finite everywhere is no reference, and the stored ``stats`` says
    so rather than disappearing.

    `restat_levels` receives a payload whose ``stats`` may already be the adjusted frame -- from the
    decorator's first pass, or from a direct call like this one. With nothing finite in
    ``adjusted`` it rebuilds on the value instead of leaving that frame alone, so the frame
    names the column its p-value is about and a reader never meets an adjusted contrast on a
    result with no usable adjustment. And it never replaces the frame with nothing: the metric
    legitimately has one.
    """
    from tcri._state import _reference as R

    _, adata = ref
    a = _cov(adata)[0]
    payload = tcri.tl.mutual_information(adata, covariate=a, groupby="patient",
                                         splitby="disease_status", inplace=False)
    before = payload["stats"].copy()
    assert set(before["quantity"]) == {"adjusted"}
    payload["replicate"]["summary"]["adjusted"] = np.nan
    R.restat_levels(payload, groupby="patient", splitby="disease_status")
    assert payload["stats"] is not None and len(payload["stats"])
    assert set(payload["stats"]["quantity"]) == {"value"}
    assert len(payload["stats"]) == len(before)


def test_an_all_nan_reference_is_no_reference(ref):
    """A reference column with nothing finite in it is not a reference.

    Both stored columns are blanked, ``null_value`` and ``adjusted``, so the default panel has
    no usable adjustment and resolves to the value. Two ways that can go wrong from there. On
    the path that collapses items to replicates the shared mask empties the frame and the panel
    renders with no marks at all; on the box-and-strip path seaborn is handed an all-NaN y and
    raises. What the
    panel should show is the value, drawn normally, with "no reference" on the label and
    nothing gray named in the legend -- on `render_metric` and on `pl.gene_importance`, which
    is a second implementation of the same rule.
    """
    import matplotlib
    matplotlib.use("Agg")
    from tcri._state.storage import _encode
    from tcri.plotting._base import REFERENCE_LABEL, REFERENCE_LEGEND

    model, adata = ref
    a = _cov(adata)[0]

    def blank_the_reference(key, name):
        blob = adata.uns[key]
        payload = tcri.get.result(adata, name)
        if "replicate" in payload:
            summaries = [lv["summary"] for lv in (payload["replicate"], payload.get("item")) if lv]
        else:
            summaries = [payload["result"]]
        for summary in summaries:
            summary["null_value"] = np.nan
            summary["adjusted"] = np.nan
        adata.uns[key] = {**_encode(payload), "params": blob["params"],
                          "version": blob["version"]}

    def gray(ax):
        return [c for c in ax.collections if c.get_label() == REFERENCE_LABEL]

    def entries(ax):
        legend = ax.get_legend()
        return [t.get_text() for t in legend.get_texts()] if legend is not None else []

    # the replicate view, and the item view on the branch that collapses the quantity with its
    # reference under one mask: an item axis (phenotype) with the patients within each position
    tcri.tl.clonotypic_entropy(adata, covariate=a, groupby="patient")
    blank_the_reference(K.CLONOTYPIC_ENTROPY, "clonotypic_entropy")
    for view in ("replicate", "item"):
        ax = tcri.pl.clonotypic_entropy(adata, view=view)
        assert ax.collections, f"{view}: the value marks were dropped along with the reference"
        assert not gray(ax), view
        assert "no reference" in ax.get_ylabel(), view

    # the gene twin's box-and-strip path, where the reference is a second seaborn pass
    tcri.perturb.gene_importance(model, adata, genes=list(adata.var_names[:4]),
                                 groupby="patient")
    blank_the_reference(K.GENE_IMPORTANCE, "gene_importance")
    ax = tcri.pl.gene_importance(adata, n_top=4)
    assert ax.collections, "the value marks were dropped along with the reference"
    assert not gray(ax)
    assert REFERENCE_LEGEND not in entries(ax)
    assert "no reference" in ax.get_ylabel()


def test_the_reference_dodges_with_the_value(ref):
    """With a `splitby` hue the value marks dodge, so the reference has to dodge with them.

    Drawn with `hue=None` the gray marks land at the category center while the value marks sit
    either side of it, so the reference reads as a third arm and its box pools two arms that
    belong to different distributions.
    """
    import matplotlib
    matplotlib.use("Agg")
    from tcri.plotting._base import REFERENCE_LABEL

    _, adata = ref
    a = _cov(adata)[0]
    tcri.tl.clonotypic_entropy(adata, covariate=a, groupby="patient",
                              splitby="disease_status")
    ax = tcri.pl.clonotypic_entropy(adata, quantity="value", view="item")
    value_x = np.concatenate([c.get_offsets()[:, 0] for c in ax.collections
                              if c.get_label() != REFERENCE_LABEL and len(c.get_offsets())])
    gray_x = np.concatenate([c.get_offsets()[:, 0] for c in ax.collections
                             if c.get_label() == REFERENCE_LABEL and len(c.get_offsets())])
    assert len(gray_x)
    # each gray mark sits in a dodge band the value marks occupy. Compared as OFFSETS FROM THE
    # CATEGORY CENTER, because the strip jitter moves individual points by a few hundredths; a
    # reference that does not dodge sits at offset 0 while the value marks sit at +-0.2.
    gray_offsets = np.abs(gray_x - np.round(gray_x))
    value_offsets = np.abs(value_x - np.round(value_x))
    assert gray_offsets.mean() > 0.5 * value_offsets.mean(), (
        f"the reference did not dodge: mean |offset| {gray_offsets.mean():.3f} against the "
        f"value marks' {value_offsets.mean():.3f}")


def test_the_perturbation_reference_dodges_too(ref):
    """The same assertion for `pl.gene_importance`, because it is a SECOND implementation.

    The twin and `render_metric` are parallel code paths, so a property held on one can be lost
    on the other: a single gray box against two colored ones pools both arms, and a gene whose
    null differs between arms then reads as if it did not. Asserting the property on both paths
    is what stops the pair drifting.
    """
    import matplotlib
    matplotlib.use("Agg")
    from tcri.plotting._base import REFERENCE_LABEL

    model, adata = ref
    tcri.perturb.gene_importance(model, adata, genes=list(adata.var_names[:6]),
                                 groupby="patient", splitby="disease_status")
    ax = tcri.pl.gene_importance(adata, n_top=4, quantity="value")

    gray = np.concatenate([c.get_offsets()[:, 0] for c in ax.collections
                           if c.get_label() == REFERENCE_LABEL and len(c.get_offsets())])
    value = np.concatenate([c.get_offsets()[:, 0] for c in ax.collections
                            if c.get_label() != REFERENCE_LABEL and len(c.get_offsets())])
    assert len(gray), "no reference was drawn"
    assert np.abs(gray - np.round(gray)).mean() > 0.5 * np.abs(value - np.round(value)).mean(), (
        "the perturbation reference pooled both arms into one box at the category center")

    # one gray box per arm per gene, matching the value marks
    gray_boxes = sum(1 for p in ax.patches if p.get_label() == REFERENCE_LABEL)
    value_boxes = sum(1 for p in ax.patches if p.get_label() != REFERENCE_LABEL)
    assert gray_boxes == value_boxes, f"{gray_boxes} gray boxes against {value_boxes} value"


def test_the_draw_path_draws_the_adjusted_draws_and_the_reference(ref):
    """Where draws are the sample, the adjusted value is a violin over its paired draws, like
    the value; and the value panel draws the null behind its violins as one marker at its
    mean per x position."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.collections import PolyCollection
    from tcri.plotting._base import REFERENCE_LABEL

    _, adata = ref
    a = _cov(adata)[0]
    tcri.tl.mutual_information(adata, covariate=a, groupby="patient", n_samples=6,
                               random_state=1)
    patients = adata.obs["patient"].nunique()

    def violins(ax):
        return [c for c in ax.collections if isinstance(c, PolyCollection)]

    def reference_marks(ax):
        return sum(len(c.get_offsets()) for c in ax.collections
                   if c.get_label() == REFERENCE_LABEL)

    ax = tcri.pl.mutual_information(adata)
    assert len(violins(ax)) == patients, "the paired draws were not drawn"
    assert not reference_marks(ax)

    value_ax = tcri.pl.mutual_information(adata, quantity="value")
    assert len(violins(value_ax)) == patients
    assert reference_marks(value_ax) == patients, "the null was not drawn behind the violins"


def test_the_endpoints_view_refuses_the_adjusted_value(ref):
    """It draws the two levels a delta is taken between; a single adjusted axis would pick
    one."""
    import matplotlib
    matplotlib.use("Agg")

    _, adata = ref
    a, b = _cov(adata)[:2]
    tcri.tl.delta_phenotypic_entropy(adata, cov_from=a, cov_to=b, groupby="patient")
    with pytest.raises(ValueError, match="kind='delta'"):
        tcri.pl.delta_phenotypic_entropy(adata, kind="endpoints", quantity="adjusted")
    assert tcri.pl.delta_phenotypic_entropy(adata, kind="endpoints") is not None


def test_a_reloaded_fit_can_still_be_measured(ref, tmp_path):
    """Every value read out of `uns` must be read against its ROUND-TRIPPED shape.

    h5ad stores a list of strings as a numpy array, so `settings.get("strata") or []` raises
    "the truth value of an array with more than one element is ambiguous" on any object that
    has been to disk -- which is every object a reference is actually read from.
    """
    import anndata as ad

    from tcri._compute._tables import _refit_hint

    _, adata = ref
    a = _cov(adata)[0]
    path = tmp_path / "reloaded.h5ad"
    adata.write_h5ad(path)
    back = ad.read_h5ad(path)

    assert "permuted within" in _refit_hint(back, "null.clonotype[0]", "patient")
    for fit in (None, "null.phenotype[0]", "null.clonotype[0]"):
        res = frames(tcri.tl.mutual_information(back, covariate=a, groupby="patient", fit=fit,
                                         null_model=None, inplace=False))[1]
        assert len(res) and np.isfinite(res["value"]).any(), fit
    # ...and a rebuild off the reloaded object reads the same settings
    assert K.fits(back) == K.fits(adata)


def test_the_gray_marks_are_named_in_the_legend(ref):
    """The reference artists are underscore-labeled, which is what keeps matplotlib from
    listing each one; so without an explicit handle a panel shows gray boxes beside the
    colored ones and nothing says what they are. A reader can take them for a third arm.

    The entry is APPENDED, never replacing what the panel already had: the box-and-strip
    legend carries the split levels and the endpoints view carries the matched-clone sizes.
    """
    import matplotlib
    matplotlib.use("Agg")
    from tcri.plotting._base import REFERENCE_LEGEND

    _, adata = ref
    a, b = _cov(adata)[:2]

    def entries(ax):
        legend = ax.get_legend()
        return [t.get_text() for t in legend.get_texts()] if legend is not None else []

    # no hue: the reference is the only thing to name, and only the value panel draws it
    tcri.tl.mutual_information(adata, covariate=a, groupby="patient")
    assert entries(tcri.pl.mutual_information(adata)) == []
    assert entries(tcri.pl.mutual_information(adata, quantity="value")) == [REFERENCE_LEGEND]

    # with a split: the arms keep their entries and the reference joins them, once
    tcri.tl.clonotypic_entropy(adata, covariate=a, groupby="patient",
                               splitby="disease_status")
    got = entries(tcri.pl.clonotypic_entropy(adata, quantity="value"))
    assert got[-1] == REFERENCE_LEGEND and len(got) > 1, got
    assert got.count(REFERENCE_LEGEND) == 1
    assert REFERENCE_LEGEND not in entries(tcri.pl.clonotypic_entropy(adata))

    # the endpoints view keeps its size legend AND its title
    tcri.tl.delta_phenotypic_entropy(adata, cov_from=a, cov_to=b, groupby="patient")
    ax = tcri.pl.delta_phenotypic_entropy(adata, kind="endpoints")
    assert ax.get_legend().get_title().get_text() == "clones matched"
    assert entries(ax)[-1] == REFERENCE_LEGEND

    # ...and nothing is named when nothing gray was drawn
    tcri.tl.mutual_information(adata, covariate=a, groupby="patient", null_model=None)
    assert REFERENCE_LEGEND not in entries(tcri.pl.mutual_information(adata))


def test_the_gene_ranking_is_corrected_by_default(ref):
    """`pl.gene_importance` draws the ADJUSTED importance at its default.

    The bare ranking is not merely incomplete, it is dominated by something the question is not
    about: silencing a gene is an intervention whose size scales with the gene's counts, so the
    bare ranking tracks expression and its null tracks it just as closely. The adjusted
    importance is what removes that shared component.

    `"auto"` still means the bare value when there is no reference. The item view follows the
    same rule: the adjusted shift at the default.
    """
    import matplotlib
    matplotlib.use("Agg")

    model, adata = ref
    genes = list(adata.var_names[:8])
    tcri.perturb.gene_importance(model, adata, genes=genes, groupby="patient")

    auto = tcri.pl.gene_importance(adata, n_top=5)
    assert auto.get_ylabel().startswith("adjusted"), auto.get_ylabel()
    bare = tcri.pl.gene_importance(adata, n_top=5, quantity="value")
    assert not bare.get_ylabel().startswith("adjusted")

    # the two rankings are allowed to differ, and that difference is the point
    auto_genes = [t.get_text() for t in auto.get_xticklabels()]
    bare_genes = [t.get_text() for t in bare.get_xticklabels()]
    assert set(auto_genes) <= set(genes) and set(bare_genes) <= set(genes)

    # the heatmap draws the adjusted shift at the default
    heat = tcri.pl.gene_importance(adata, view="item", n_top=5)
    assert heat.figure.axes[-1].get_ylabel().startswith("adjusted"), heat.figure.axes[-1]

    # ...and with no reference, "auto" is the bare value
    tcri.perturb.gene_importance(model, adata, genes=genes, groupby="patient",
                                 null_model=None)
    plain = tcri.pl.gene_importance(adata, n_top=5)
    assert not plain.get_ylabel().startswith("adjusted")


# ── a reference of several fits ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def pooled():
    """A small fit with a phenotype reference of three nulls and a covariate reference of two,
    and the models ``tcri.null.*`` returned. Returns ``(model, adata, nulls)``."""
    from tcri.datasets import simulate_tcri
    from tcri.model._model import TCRIModel

    adata = simulate_tcri(n_clones=10, n_phenotypes=3, n_genes=20, n_cells=300,
                          n_covariates=2, omega_concentration=0.4, seed=0)
    adata.layers["counts"] = adata.X.copy()
    TCRIModel.setup_anndata(adata, layer="counts", clonotype_key="clone_id",
                            phenotype_key="phenotype", covariate_key="covariate",
                            batch_key="batch")
    model = TCRIModel(adata, name="pooledparent", n_latent=8, n_hidden=16, n_layers=1,
                      classifier_n_layers=1, classifier_hidden=16, K=3, seed=0)
    train = dict(max_epochs=3, batch_size=64, n_epochs_kl_warmup=2, accelerator="cpu",
                 enable_progress_bar=False, enable_model_summary=False)
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.train(**train)
        model.to_anndata(adata)
        nulls = {"phenotype": tcri.null.phenotype(model, adata, n_nulls=3),
                 "covariate": tcri.null.covariate(model, adata, n_nulls=2)}
    return model, adata, nulls


def _mi(adata, **kw):
    cov = str(adata.uns[K.COVARIATE_CATEGORIES][0])
    return tcri.tl.mutual_information(adata, covariate=cov, random_state=0, inplace=False, **kw)


def test_the_reference_pools_every_fit(pooled):
    """Draw *d* of the fit is paired with draw *d* of each of the three nulls; ``result``
    summarizes all thirty pairs, whose variance is the nulls' own plus the spread between them,
    and whose interval holds every null's mean."""
    from tcri._stats import hdi

    _, adata, _ = pooled
    res = _mi(adata, n_samples=10)
    table, summary = frames(res)
    row = summary.iloc[0]
    assert len(table) == 10 * 3 and sorted(set(table["null"])) == [0, 1, 2]
    np.testing.assert_allclose(table["adjusted"], table["value"] - table["null_value"],
                               atol=1e-12)
    by_draw = table.groupby("draw")["value"]
    assert (by_draw.nunique() == 1).all(), "the fit's draws differ between the nulls"

    pairs = table["adjusted"].to_numpy(dtype=float)
    per_null = table.groupby("null")["adjusted"]
    within = per_null.var(ddof=0).mean()
    between = per_null.mean().var(ddof=0)
    assert pairs.var(ddof=0) == pytest.approx(within + between, rel=1e-9)
    assert between > 0, "the three nulls gave one value"
    lo, hi = hdi(pairs, prob=0.94)
    assert (row["adjusted"], row["adjusted_hdi_low"], row["adjusted_hdi_high"]) == pytest.approx(
        (pairs.mean(), lo, hi))
    assert ((per_null.mean() >= lo) & (per_null.mean() <= hi)).all(), per_null.mean()


def test_one_null_alone_is_its_pairs_in_the_pool(pooled):
    """A null read alone, by name or as its model, draws what it draws inside its reference:
    each null's draws are seeded by its own index, not by its position in a call."""
    _, adata, nulls = pooled
    pool = _mi(adata, n_samples=6)["replicate"]["draws"]
    by_name = _mi(adata, n_samples=6, null_model="null.phenotype[1]")
    by_model = _mi(adata, n_samples=6, null_model=nulls["phenotype"][1])
    by_name, by_model = by_name["replicate"]["draws"], by_model["replicate"]["draws"]
    np.testing.assert_allclose(by_name["null_value"].to_numpy(),
                               pool.loc[pool["null"] == 1, "null_value"].to_numpy(), atol=1e-12)
    pd.testing.assert_frame_equal(by_name, by_model)
    assert set(by_name["null"]) == {0}


def test_null_model_takes_the_list_tcri_null_returned(pooled):
    """The list of models ``tcri.null.phenotype`` returned is the reference: the same pairs, and
    recorded under the reference's name."""
    _, adata, nulls = pooled
    pd.testing.assert_frame_equal(
        _mi(adata, n_samples=4, null_model=nulls["phenotype"])["replicate"]["draws"],
        _mi(adata, n_samples=4, null_model="phenotype")["replicate"]["draws"])
    cov = str(adata.uns[K.COVARIATE_CATEGORIES][0])
    tcri.tl.mutual_information(adata, covariate=cov, null_model=nulls["phenotype"])
    assert tcri.get.params(adata, "mutual_information")["null_model"] == "null.phenotype"
    with pytest.raises(TypeError, match="fitted models"):
        _mi(adata, null_model=["null.phenotype[0]"])


def test_the_stored_table_grows_by_n_nulls(pooled):
    """Per-clone entropy, the largest table: one row per clone and draw without a reference,
    three per clone and draw with a reference of three, at one draw and at several."""
    _, adata, _ = pooled
    cov = str(adata.uns[K.COVARIATE_CATEGORIES][0])
    for n_samples in (0, 5):
        kw = dict(covariate=cov, n_samples=n_samples, random_state=0, inplace=False)
        bare = frames(tcri.tl.phenotypic_entropy(adata, null_model=None, **kw))[0]
        with_ref = frames(tcri.tl.phenotypic_entropy(adata, **kw))[0]
        assert len(bare) == bare["clonotype"].nunique() * max(n_samples, 1)
        assert len(with_ref) == 3 * len(bare), n_samples


def test_a_delta_pairs_its_endpoints_within_one_null(pooled):
    """Every pair row of a delta takes its value and both endpoints from one null, so the
    delta's closure holds pair by pair, and each null's rows are that null's own call."""
    _, adata, _ = pooled
    a, b = (str(c) for c in adata.uns[K.COVARIATE_CATEGORIES][:2])
    kw = dict(cov_from=a, cov_to=b, n_samples=4, random_state=1, inplace=False)
    table = tcri.tl.delta_phenotypic_entropy(adata, **kw)["table"]
    assert sorted(set(table["null"])) == [0, 1]
    closure = table["adjusted"] - (table["adjusted_to"] - table["adjusted_from"])
    np.testing.assert_allclose(closure, table["value"] - (table["value_to"] - table["value_from"]),
                               atol=1e-12, equal_nan=True)
    for i in (0, 1):
        own = tcri.tl.delta_phenotypic_entropy(adata, null_model=f"null.covariate[{i}]",
                                               **kw)["table"]
        mine = table[table["null"] == i]
        for column in ("null_value", "null_value_from", "null_value_to"):
            np.testing.assert_allclose(mine[column].to_numpy(), own[column].to_numpy(),
                                       atol=1e-12, equal_nan=True, err_msg=column)
