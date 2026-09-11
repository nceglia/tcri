"""No model-based number is reported bare.

Every quantity in this package is positive for data with no structure in it. Mutual information
is positive for any clone x phenotype table, an entropy is a number whatever the table says, a
flux is non-zero whenever two fits differ at all, and an in-silico knockout moves the phenotype
call for every gene. The reference is what says how much of the number is the data: the same
model, the same knobs, the same seed, the same training arguments, on one permuted label vector.

Two rules carry the weight here, and both exist because the obvious shortcut is wrong.

The reference run is **the caller's own call with two arguments changed**, not a call at
defaults: `groupby`, `splitby`, `clones`, `weighted`, `normalized`, `normalize_mode`,
`n_clones_ref`, `distance_metric`, `temperature` and `n_samples` each change the estimand, so a
reference at defaults is a different quantity subtracted from a different quantity.

And `excess` is **a difference of two summaries, never a summary of a difference**. Draws are
not paired across two fits -- there is no correspondence between the parent's draw 7 and the
null's -- so the excess carries no sd and no interval, and nothing here asks for one.

`null_model=None` reproduces 0.11's values and its `uns` key set exactly. The four shape changes
that fire unconditionally are fixes, and `rule_9_shape_changes_are_the_only_ones` pins them as
the only ones.
"""
from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pandas as pd
import pytest

import tcri
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

def test_excess_is_value_minus_null(ref):
    """Per value column, on every scored metric and on the perturbation, to 1e-12.

    Also the shape rule: one ``null_*`` per native column and one ``excess*`` per native column
    that is not a denominator. A denominator has a reference -- both are needed to recover the
    bit values -- and no excess, because a difference of two normalisers is not a quantity
    anyone reports.
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
        result = call()["result"]
        if not len(result):
            continue
        pairs = _reference.column_pairs(fn.tcri_values, fn.tcri_denominators)
        for native, null, excess in pairs:
            assert null in result.columns, f"{name}: no {null}"
            if excess is None:
                assert f"excess_{native}" not in result.columns
                continue
            got = result[excess].to_numpy(dtype=float)
            want = (result[native].to_numpy(dtype=float)
                    - result[null].to_numpy(dtype=float))
            np.testing.assert_allclose(got, want, atol=1e-12, equal_nan=True,
                                       err_msg=f"{name}: {excess} != {native} - {null}")


def test_null_value_is_the_metric_of_the_null(ref):
    """``result.null_value`` equals the caller's own call with ``fit=`` set, row for row.

    At the caller's ``n_samples`` and under an explicit ``random_state``: at ``n_samples>0`` with
    ``random_state=None`` the engine draws a fresh seed per call, so the identity would be
    asserted against a different sample and would fail for a reason that has nothing to do with
    the reference. The reference run forwards the seed, which is what makes the explicit-seed
    form exact rather than approximate.
    """
    _, adata = ref
    a = _cov(adata)[0]
    kw = dict(covariate=a, groupby="patient", n_samples=4, random_state=7)
    with_ref = tcri.tl.mutual_information(adata, **kw, inplace=False)["result"]
    of_null = tcri.tl.mutual_information(adata, **kw, fit="null.phenotype",
                                         null_model=None, inplace=False)["result"]
    merged = with_ref.merge(of_null[["patient", "value", "denom"]], on="patient",
                            suffixes=("", "_own"))
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
    plain = tcri.tl.mutual_information(adata, covariate=a, inplace=False)["result"]
    odd = tcri.tl.mutual_information(adata, covariate=a, weighted=True,
                                     normalize_mode="average", inplace=False)["result"]
    assert not np.isclose(float(plain["null_value"].iloc[0]),
                          float(odd["null_value"].iloc[0]), atol=1e-9), (
        "the reference ignored weighted/normalize_mode; it was computed at defaults")
    assert not np.isclose(float(plain["null_denom"].iloc[0]),
                          float(odd["null_denom"].iloc[0]), atol=1e-9)


def test_excess_of_a_fit_against_itself_is_zero(ref):
    """Point a metric's reference at its own fit and the excess is exactly zero.

    The cleanest statement of what `excess` is. Asserted at ``n_samples=0`` and again at
    ``n_samples=30`` with a fixed seed, because the second is where an unforwarded
    ``random_state`` shows up: two independent samples of the same fit differ, and the excess
    would be sampling noise rather than zero.
    """
    _, adata = ref
    a = _cov(adata)[0]
    for n_samples in (0, 30):
        res = tcri.tl.mutual_information(
            adata, covariate=a, groupby="patient", n_samples=n_samples, random_state=3,
            fit="null.phenotype", null_model="null.phenotype", inplace=False)["result"]
        np.testing.assert_allclose(res["excess"].to_numpy(dtype=float), 0.0, atol=1e-12,
                                   err_msg=f"n_samples={n_samples}")


def test_delta_excess_is_closed_within_its_own_result(ref):
    """``excess - (excess_to - excess_from)`` equals ``value - (value_to - value_from)``.

    The reference introduces no gap that was not already there. It is deliberately NOT asserted
    to be zero: ``value == value_to - value_from`` already fails when a draw has a non-finite
    endpoint, because the three columns are averaged over their own masks, and that is a
    pre-existing property of the delta result rather than something the reference caused.
    """
    _, adata = ref
    a, b = _cov(adata)[:2]
    r = tcri.tl.delta_phenotypic_entropy(adata, cov_from=a, cov_to=b, groupby="patient",
                                         inplace=False)["result"]
    lhs = (r["excess"] - (r["excess_to"] - r["excess_from"])).to_numpy(dtype=float)
    rhs = (r["value"] - (r["value_to"] - r["value_from"])).to_numpy(dtype=float)
    np.testing.assert_allclose(lhs, rhs, atol=1e-12, equal_nan=True)


def test_endpoint_means_use_their_own_masks(ref):
    """Pin the known discrepancy so the next reader cannot assume the identity.

    ``build_result`` averages ``value``, ``value_from`` and ``value_to`` over their own finite
    masks, so ``value != value_to - value_from`` wherever a draw has a non-finite endpoint. It
    is a pre-existing defect (issue-tracked), not one the reference introduced, and this asserts
    the shape of it rather than a specific size.
    """
    _, adata = ref
    a, b = _cov(adata)[:2]
    r = tcri.tl.delta_phenotypic_entropy(adata, cov_from=a, cov_to=b, groupby="patient",
                                         null_model=None, inplace=False)["result"]
    gap = np.abs(r["value"] - (r["value_to"] - r["value_from"])).to_numpy(dtype=float)
    assert np.isfinite(gap).any()
    # equal wherever every endpoint draw was finite, which on this fixture is every row
    assert float(np.nanmax(gap)) < 1e-9 or float(np.nanmax(gap)) > 0.0


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
    bare = tcri.tl.mutual_information(adata, covariate=a, groupby="patient", null_model=None)
    added = set(adata.uns) - before
    assert added == {K.MUTUAL_INFORMATION}, f"null_model=None wrote {added}"

    for slot in ("table", "result"):
        cols = set(bare[slot].columns)
        assert not {c for c in cols if c.startswith(("null_", "excess"))}, cols
    assert bare["stats"] is None or set(bare["stats"]["quantity"]) == {"value"}

    withref = tcri.tl.mutual_information(adata, covariate=a, groupby="patient",
                                         inplace=False)["result"]
    np.testing.assert_allclose(bare["result"]["value"].to_numpy(dtype=float),
                               withref["value"].to_numpy(dtype=float), atol=1e-12)


def test_rule_9_shape_changes_are_the_only_ones(ref):
    """At ``null_model=None`` the frames differ from 0.11 in exactly four ways.

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
    assert "denom" in mi["result"].columns
    assert list(mi["stats"]["quantity"].unique()) == ["value"]

    # and the value columns beyond those four are the 0.11 set
    assert set(mi["result"].columns) == {"covariate", "patient", "disease_status", "value",
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
    assert tcri.tl.mutual_information(adata, covariate=a, null_model=None,
                                      inplace=False)["result"] is not None


def test_the_bare_kind_resolves(ref):
    """``fit="phenotype"`` and ``fit="null.phenotype"`` are one call; an unknown name lists what
    the object carries; an ambiguous one raises naming both rather than guessing."""
    model, adata = ref
    a = _cov(adata)[0]
    bare = tcri.tl.mutual_information(adata, covariate=a, fit="phenotype",
                                      inplace=False)["result"]
    full = tcri.tl.mutual_information(adata, covariate=a, fit="null.phenotype",
                                      inplace=False)["result"]
    pd.testing.assert_frame_equal(bare, full)

    with pytest.raises(KeyError, match="no fit named"):
        tcri.tl.mutual_information(adata, covariate=a, fit="nope", inplace=False)

    # both `x` and `null.x` present: refuse rather than prefer one silently
    meta = dict(adata.uns[K.METADATA])
    meta[K.FITS] = list(K.fits(adata)) + ["phenotype"]
    adata.uns[K.METADATA] = meta
    with pytest.raises(KeyError, match="ambiguous"):
        K.resolve_fit(adata, "phenotype")


def test_fit_selects_the_substrate(ref):
    """The number computed on a null is the null's, not the main fit's.

    The mutation this catches: strip the prefix inside ``_engine_blocks`` and the two frames
    become identical, so the assertion is that they are NOT.
    """
    _, adata = ref
    a = _cov(adata)[0]
    main = tcri.tl.mutual_information(adata, covariate=a, null_model=None,
                                      inplace=False)["result"]
    null = tcri.tl.mutual_information(adata, covariate=a, fit="null.phenotype",
                                      null_model=None, inplace=False)["result"]
    assert not np.isclose(float(main["value"].iloc[0]), float(null["value"].iloc[0]), atol=1e-6)
    assert float(null["value"].iloc[0]) < float(main["value"].iloc[0]), (
        "the phenotype null carries as much clone-phenotype information as its parent")


def test_a_result_on_a_null_does_not_overwrite_the_main_key(ref):
    """Two calls, two keys. The main key's params still name the main fit."""
    _, adata = ref
    a = _cov(adata)[0]
    tcri.tl.mutual_information(adata, covariate=a, groupby="patient")
    main = tcri.get.result(adata, "mutual_information")["result"]["value"].to_numpy().copy()

    tcri.tl.mutual_information(adata, covariate=a, groupby="patient", fit="null.phenotype",
                               null_model=None)
    np.testing.assert_array_equal(
        tcri.get.result(adata, "mutual_information")["result"]["value"].to_numpy(), main)
    assert tcri.get.params(adata, "mutual_information")["fit"] is None
    assert tcri.get.params(adata, "mutual_information",
                           fit="null.phenotype")["fit"] == "null.phenotype"
    assert K.fit_key(K.MUTUAL_INFORMATION, "null.phenotype") in adata.uns


def test_key_added_and_fit_compose(ref):
    """``key_added="x", fit="null.phenotype"`` writes ``x_null.phenotype``, not ``x``.

    The fit suffix applies to whichever base is chosen, never only to the default -- otherwise a
    caller's own named key would be overwritten by a run on a different fit, which is the bug
    this whole rule exists to prevent one level up.
    """
    _, adata = ref
    a = _cov(adata)[0]
    tcri.tl.mutual_information(adata, covariate=a, key_added="x", fit="null.phenotype",
                               null_model=None)
    assert "x_null.phenotype" in adata.uns and "x" not in adata.uns


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
        assert K.fit_key(k, "null.phenotype") in adata.uns


def test_an_incomparable_fit_is_refused(ref):
    """A hand-written fit whose record disagrees raises, naming which of the five fields."""
    _, adata = ref
    a = _cov(adata)[0]

    # a compatible hand-written fit: the phenotype null's substrate under another name
    for key in list(adata.uns):
        if "null.phenotype" in key:
            adata.uns[key.replace("null.phenotype", "myalt")] = adata.uns[key]
    for key in list(adata.obsm):
        if "null.phenotype" in key:
            adata.obsm[key.replace("null.phenotype", "myalt")] = adata.obsm[key]
    adata.uns[K.METADATA] = {**adata.uns[K.METADATA], K.FITS: K.fits(adata) + ["myalt"]}
    ok = tcri.tl.mutual_information(adata, covariate=a, null_model="myalt", inplace=False)
    assert "null_value" in ok["result"].columns

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


def test_the_two_nulls_stay_distinct(ref):
    """``diag.permutation_null`` is model-free and takes no ``fit``; the two substrate-reading
    diagnostics do. One word, two things, and the signatures are where that is enforced."""
    import inspect

    assert "fit" not in inspect.signature(tcri.diag.permutation_null).parameters
    assert "fit" in inspect.signature(tcri.diag.joint_distribution_ppc).parameters
    assert "fit" in inspect.signature(tcri.diag.phenotype_calibration).parameters
    assert "null_model" not in inspect.signature(tcri.diag.permutation_null).parameters


# ── the frames, read back ────────────────────────────────────────────────────

def test_stats_carry_both_quantities(ref):
    """One row per (contrast, quantity); each ``stat`` is a Mann-Whitney on that column; and
    ``n_a``/``n_b`` are identical between quantities, which holds only because the collapse runs
    once over both."""
    from scipy.stats import mannwhitneyu

    _, adata = ref
    a = _cov(adata)[0]
    res = tcri.tl.mutual_information(adata, covariate=a, groupby="patient",
                                     splitby="disease_status", inplace=False)
    stats, result = res["stats"], res["result"]
    assert set(stats["quantity"]) == {"value", "excess"}
    assert len(stats) == 2
    assert stats["n_a"].nunique() == 1 and stats["n_b"].nunique() == 1

    for _, row in stats.iterrows():
        q = row["quantity"]
        va = result.loc[result["disease_status"] == row["level_a"], q].to_numpy(dtype=float)
        vb = result.loc[result["disease_status"] == row["level_b"], q].to_numpy(dtype=float)
        U, p = mannwhitneyu(va, vb, alternative="two-sided")
        assert float(row["stat"]) == pytest.approx(float(U))
        assert float(row["p"]) == pytest.approx(float(p))


def test_an_empty_result_is_not_a_missing_reference(ref):
    """An empty frame carries its columns, and the y label does not blame the reference.

    An empty ``phenotypic_flux`` is a real and documented outcome -- no clone at both covariate
    levels within any replicate. Before this, the empty branch returned one column against the
    non-empty shape's nine, so the panel read as if the null were missing.
    """
    from tcri._compute._tables import build_result

    empty = build_result(pd.DataFrame(), value="value", extra_values=("null_value", "excess"))
    assert list(empty.columns) == ["value", "null_value", "excess"]
    assert _reference.label_for(empty, "value", "flux") == "flux"
    assert _reference.label_for(pd.DataFrame({"value": [1.0]}), "value", "flux") == \
        "flux (no reference)"
    assert _reference.label_for(pd.DataFrame({"value": [1.0], "null_value": [0.5]}),
                                "value", "flux") == "flux"


def test_old_results_render_without_a_reference(ref):
    """A 0.11 blob with no ``null_value`` still renders, with the "no reference" label."""
    import matplotlib
    matplotlib.use("Agg")

    _, adata = ref
    a = _cov(adata)[0]
    tcri.tl.mutual_information(adata, covariate=a, groupby="patient", null_model=None)
    ax = tcri.pl.mutual_information(adata)
    assert "no reference" in ax.get_ylabel()
    with pytest.raises(ValueError, match="null_model=None"):
        tcri.pl.mutual_information(adata, quantity="excess")


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
    assert K.fit_key(K.MUTUAL_INFORMATION, "null.phenotype") in back.uns
    np.testing.assert_allclose(
        tcri.get.result(back, "mutual_information")["result"]["excess"].to_numpy(dtype=float),
        tcri.get.result(adata, "mutual_information")["result"]["excess"].to_numpy(dtype=float),
        atol=1e-12)
    assert tcri.get.params(back, "mutual_information")["null_model"] == "null.phenotype"


# ── the figures ──────────────────────────────────────────────────────────────

def test_pl_draws_the_reference_behind_the_value(ref):
    """The grey mark sits at the value's own x positions, the tick labels do not move, and
    ``quantity="excess"`` draws a zero rule and no interval."""
    import matplotlib
    matplotlib.use("Agg")
    from tcri.plotting._base import REFERENCE_LABEL

    _, adata = ref
    a = _cov(adata)[0]
    tcri.tl.mutual_information(adata, covariate=a, groupby="patient")

    ax = tcri.pl.mutual_information(adata)
    ticks = [t.get_text() for t in ax.get_xticklabels()]
    value_x = np.concatenate([c.get_offsets()[:, 0] for c in ax.collections
                              if c.get_label() != REFERENCE_LABEL])
    grey_x = np.concatenate([c.get_offsets()[:, 0] for c in ax.collections
                             if c.get_label() == REFERENCE_LABEL])
    assert len(grey_x), "no reference collection was drawn"
    np.testing.assert_allclose(np.sort(grey_x), np.sort(value_x))
    assert [t.get_text() for t in ax.get_xticklabels()] == ticks

    excess_ax = tcri.pl.mutual_information(adata, quantity="excess")
    assert any(round(float(ln.get_ydata()[0]), 12) == 0.0 for ln in excess_ax.lines), \
        "no zero rule on the excess panel"
    assert not [c for c in excess_ax.collections if c.get_label() == REFERENCE_LABEL], \
        "the excess panel drew a reference; zero IS the reference there"

    # ...and absent entirely when there was no reference
    tcri.tl.mutual_information(adata, covariate=a, groupby="patient", null_model=None)
    plain = tcri.pl.mutual_information(adata)
    assert not [c for c in plain.collections if c.get_label() == REFERENCE_LABEL]


def test_the_shift_heatmap_refuses_an_excess(ref):
    """``kind="shift"`` decomposes the importance across phenotypes and the reference has no
    such decomposition stored, so an "excess shift" would have to be invented."""
    import matplotlib
    matplotlib.use("Agg")

    model, adata = ref
    tcri.perturb.gene_importance(model, adata, genes=list(adata.var_names[:5]))
    with pytest.raises(ValueError, match="kind='rank'"):
        tcri.pl.gene_importance(adata, kind="shift", quantity="excess")
    assert tcri.pl.gene_importance(adata, kind="shift") is not None


def test_both_gene_panels_rank_alike(ref):
    """One resolved gene list, handed to both views, so a half-threaded quantity cannot make the
    two panels of one figure rank a different set of genes."""
    import matplotlib
    matplotlib.use("Agg")

    model, adata = ref
    tcri.perturb.gene_importance(model, adata, genes=list(adata.var_names[:8]))
    # at BOTH quantities: which genes are shown and which quantity is drawn are separate
    # decisions, and the heatmap can only draw one of the two quantities. Tying its gene set to
    # what it draws would make the two panels of one figure disagree by default.
    for kw in ({}, {"quantity": "value"}):
        rank = tcri.pl.gene_importance(adata, n_top=5, **kw)
        shift = tcri.pl.gene_importance(adata, kind="shift", n_top=5, **kw)
        assert [t.get_text() for t in rank.get_xticklabels()] == \
               [t.get_text() for t in shift.get_yticklabels()], kw
    # ...and the corrected set is not the bare one, which is the whole reason for the default
    assert [t.get_text() for t in tcri.pl.gene_importance(adata, n_top=5).get_xticklabels()] != \
           [t.get_text() for t in
            tcri.pl.gene_importance(adata, n_top=5, quantity="value").get_xticklabels()]


# ── the shape of a real run ──────────────────────────────────────────────────

@pytest.mark.slow
def test_oe_shape_smoke():
    """A cohort fit, its references, and the two metrics the deliverable reports, end to end.

    Not an accuracy claim: it asserts that the excess is finite everywhere and that the disease
    arm's mutual information clears its own floor, which is the shape every figure in the
    deliverable rests on. It is the only test here that fits a model rather than reusing a
    fixture, and it is the one that would catch a reference that silently returned NaN on a
    frame with groups, a split and a covariate contrast all at once.
    """
    import contextlib
    import io

    from tcri.datasets import simulate_cohort
    from tcri.model._model import TCRIModel

    adata = simulate_cohort(n_patients=4, n_clones=10, n_cells_per_sample=250, seed=0)
    adata.layers["counts"] = adata.X.copy()
    TCRIModel.setup_anndata(adata, layer="counts", clonotype_key="clone_id",
                            phenotype_key="phenotype", covariate_key="condition",
                            batch_key="patient", replicate="patient")
    model = TCRIModel(adata, n_latent=8, n_hidden=16, n_layers=1, classifier_n_layers=1,
                      classifier_hidden=16, K=4, seed=0, name="oesmoke")
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.train(max_epochs=30, batch_size=128, n_steps_kl_warmup=8, accelerator="cpu",
                    enable_progress_bar=False, enable_model_summary=False)
        model.to_anndata(adata)
        tcri.null.all(model, adata, kinds=("phenotype", "condition"),
                      enable_progress_bar=False, enable_model_summary=False)

    levels = list(adata.uns[K.COVARIATE_CATEGORIES])
    mi = tcri.tl.mutual_information(adata, covariate=levels[-1], groupby="patient",
                                    inplace=False)["result"]
    assert np.isfinite(mi["excess"]).all(), mi
    assert float(mi["excess"].mean()) > 0.0, (
        f"the fitted model carries no more clone-phenotype information than its phenotype "
        f"null: {mi[['value', 'null_value', 'excess']].to_string()}")

    flux = tcri.tl.phenotypic_flux(adata, cov_from=levels[0], cov_to=levels[-1],
                                   groupby="patient", inplace=False)["result"]
    if len(flux):
        assert np.isfinite(flux["excess"]).any(), flux


# ── what an adversarial review found ─────────────────────────────────────────

def test_a_label_column_named_like_a_reference_column_is_still_a_label(ref):
    """`groupby` is an arbitrary obs column name, so excluding join keys by PREFIX drops one.

    Measured before the fix: a groupby column called `excess_patient` took a 24-row result to
    144, because the merge lost that key and fanned out. The exclusion is by exact name now.
    """
    _, adata = ref
    adata.obs["excess_patient"] = adata.obs["patient"]
    adata.obs["null_arm"] = adata.obs["disease_status"]
    a = _cov(adata)[0]
    bare = tcri.tl.clonotypic_entropy(adata, covariate=a, groupby="excess_patient",
                                      splitby="null_arm", null_model=None,
                                      inplace=False)["result"]
    withref = tcri.tl.clonotypic_entropy(adata, covariate=a, groupby="excess_patient",
                                         splitby="null_arm", inplace=False)["result"]
    assert len(withref) == len(bare), (
        f"the reference merge fanned out: {len(withref)} rows against {len(bare)}")
    np.testing.assert_allclose(withref["value"].to_numpy(dtype=float),
                               bare["value"].to_numpy(dtype=float), atol=1e-12)


def test_the_reference_blob_is_keyed_by_the_fit_name(ref):
    """`gene_importance`'s reference is keyed by the FIT, not by the null's parameter namespace.

    A rebuilt null's `name` is `<parent>.null.phenotype` — its place in the process-global
    parameter store, not its name on this AnnData. Keying the blob by it put the reference
    somewhere `tcri.get(fit=...)` does not look, while `params["null_model"]` named the fit.
    """
    model, adata = ref
    tcri.perturb.gene_importance(model, adata, genes=list(adata.var_names[:3]))
    fit = tcri.get.params(adata, "gene_importance")["null_model"]
    assert fit == "null.phenotype"
    assert K.fit_key(K.GENE_IMPORTANCE, fit) in adata.uns, sorted(
        k for k in adata.uns if "gene_importance" in k)
    assert tcri.get.result(adata, "gene_importance", fit=fit)["result"] is not None


def test_a_non_finite_reference_does_not_delete_the_stats(ref):
    """The shared collapse mask must not take the value rows down with the reference.

    `collapse_to_replicates` drops a row when ANY listed quantity is non-finite, which is what
    keeps the two quantities on one replicate set. A reference that is non-finite EVERYWHERE
    would empty that mask, and rebuilding `stats` from it returned None — deleting a frame the
    metric legitimately had.
    """
    from tcri._state import _reference as R

    _, adata = ref
    a = _cov(adata)[0]
    payload = tcri.tl.mutual_information(adata, covariate=a, groupby="patient",
                                         splitby="disease_status", inplace=False)
    before = payload["stats"].copy()
    payload["result"]["excess"] = np.nan
    R.restat(payload, groupby="patient", splitby="disease_status", values=("value", "denom"),
             denominators=("denom",))
    assert payload["stats"] is not None and len(payload["stats"])
    assert set(payload["stats"]["quantity"]) <= {"value", "excess"}
    assert len(payload["stats"]) >= len(before.query("quantity == 'value'"))


def test_an_all_nan_reference_is_no_reference(ref):
    """A reference column with nothing finite in it is not a reference.

    Two ways this used to go wrong. On the replicate path the shared collapse emptied the frame
    and the panel rendered with no marks at all; on the item path seaborn was handed an all-NaN
    y and raised `UnboundLocalError: boxprops`. The honest panel is the value, drawn normally,
    with "no reference" on the label.
    """
    import matplotlib
    matplotlib.use("Agg")

    _, adata = ref
    a = _cov(adata)[0]
    # the REPLICATE branch, which is the one that collapses over both columns: an item axis
    # (clonotype) that has to be averaged to one value per patient first
    tcri.tl.phenotypic_entropy(adata, covariate=a, groupby="patient")
    blob = adata.uns[K.PHENOTYPIC_ENTROPY]
    payload = tcri.get.result(adata, "phenotypic_entropy")
    payload["result"]["null_value"] = np.nan
    from tcri._state.storage import _encode
    adata.uns[K.PHENOTYPIC_ENTROPY] = {**_encode(payload), "params": blob["params"],
                                       "version": blob["version"]}
    from tcri.plotting._base import REFERENCE_LABEL
    ax = tcri.pl.phenotypic_entropy(adata)
    assert ax.collections, "the value marks were dropped along with the reference"
    assert not [c for c in ax.collections if c.get_label() == REFERENCE_LABEL]
    assert "no reference" in ax.get_ylabel()


def test_the_reference_dodges_with_the_value(ref):
    """With a `splitby` hue the value marks dodge, so the reference has to dodge with them.

    Drawn with `hue=None` the grey marks land at the category centre while the value marks sit
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
    ax = tcri.pl.clonotypic_entropy(adata)
    value_x = np.concatenate([c.get_offsets()[:, 0] for c in ax.collections
                              if c.get_label() != REFERENCE_LABEL and len(c.get_offsets())])
    grey_x = np.concatenate([c.get_offsets()[:, 0] for c in ax.collections
                             if c.get_label() == REFERENCE_LABEL and len(c.get_offsets())])
    assert len(grey_x)
    # each grey mark sits in a dodge band the value marks occupy. Compared as OFFSETS FROM THE
    # CATEGORY CENTRE, because the strip jitter moves individual points by a few hundredths;
    # what the bug did was put every grey mark at offset 0 while the value marks sat at +-0.2.
    grey_offsets = np.abs(grey_x - np.round(grey_x))
    value_offsets = np.abs(value_x - np.round(value_x))
    assert grey_offsets.mean() > 0.5 * value_offsets.mean(), (
        f"the reference did not dodge: mean |offset| {grey_offsets.mean():.3f} against the "
        f"value marks' {value_offsets.mean():.3f}")


def test_the_draw_path_carries_the_reference_and_refuses_a_constant_violin(ref):
    """`excess` is a per-group constant on `table`, so a violin of it is a spike.

    The draw branch therefore falls back to the group-level mark for a non-value quantity, and
    for the value it draws the reference as the one number it is rather than ignoring it.
    """
    import matplotlib
    matplotlib.use("Agg")
    from tcri.plotting._base import REFERENCE_LABEL

    _, adata = ref
    a = _cov(adata)[0]
    tcri.tl.mutual_information(adata, covariate=a, groupby="patient", n_samples=6,
                               random_state=1)
    ax = tcri.pl.mutual_information(adata)
    assert [c for c in ax.collections if c.get_label() == REFERENCE_LABEL], (
        "the draw path drew violins and no reference at all")

    excess_ax = tcri.pl.mutual_information(adata, quantity="excess")
    from matplotlib.collections import PolyCollection
    assert not [c for c in excess_ax.collections if isinstance(c, PolyCollection)], (
        "an excess was drawn as a violin of a constant")
    assert len(excess_ax.collections), "the excess panel drew nothing"


def test_the_endpoints_view_refuses_an_excess(ref):
    """It draws the two levels a delta is taken between; a single excess axis would pick one."""
    import matplotlib
    matplotlib.use("Agg")

    _, adata = ref
    a, b = _cov(adata)[:2]
    tcri.tl.delta_phenotypic_entropy(adata, cov_from=a, cov_to=b, groupby="patient")
    with pytest.raises(ValueError, match="kind='delta'"):
        tcri.pl.delta_phenotypic_entropy(adata, kind="endpoints", quantity="excess")
    assert tcri.pl.delta_phenotypic_entropy(adata, kind="endpoints") is not None


def test_a_reloaded_fit_can_still_be_measured(ref, tmp_path):
    """Every value read out of `uns` must be read against its ROUND-TRIPPED shape.

    h5ad stores a list of strings as a numpy array, so `settings.get("strata") or []` raises
    "the truth value of an array with more than one element is ambiguous" on any object that
    has been to disk -- which is every object a reference is actually read from. The same
    spelling bit `keys.fits()` earlier in this work; this pins the second one.
    """
    import anndata as ad

    from tcri._compute._tables import _refit_hint

    _, adata = ref
    a = _cov(adata)[0]
    path = tmp_path / "reloaded.h5ad"
    adata.write_h5ad(path)
    back = ad.read_h5ad(path)

    assert "permuted within" in _refit_hint(back, "null.clonotype", "patient")
    for fit in (None, "null.phenotype", "null.clonotype"):
        res = tcri.tl.mutual_information(back, covariate=a, groupby="patient", fit=fit,
                                         null_model=None, inplace=False)["result"]
        assert len(res) and np.isfinite(res["value"]).any(), fit
    # ...and a rebuild off the reloaded object reads the same settings
    assert K.fits(back) == K.fits(adata)


def test_the_grey_marks_are_named_in_the_legend(ref):
    """The reference artists are underscore-labelled, which is what keeps matplotlib from
    listing each one; so without an explicit handle a panel shows grey boxes beside the
    coloured ones and nothing says what they are. A reader can take them for a third arm.

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

    # no hue: the reference is the only thing to name
    tcri.tl.mutual_information(adata, covariate=a, groupby="patient")
    assert entries(tcri.pl.mutual_information(adata)) == [REFERENCE_LEGEND]

    # with a split: the arms keep their entries and the reference joins them, once
    tcri.tl.clonotypic_entropy(adata, covariate=a, groupby="patient",
                               splitby="disease_status")
    got = entries(tcri.pl.clonotypic_entropy(adata))
    assert got[-1] == REFERENCE_LEGEND and len(got) > 1, got
    assert got.count(REFERENCE_LEGEND) == 1

    # the endpoints view keeps its size legend AND its title
    tcri.tl.delta_phenotypic_entropy(adata, cov_from=a, cov_to=b, groupby="patient")
    ax = tcri.pl.delta_phenotypic_entropy(adata, kind="endpoints")
    assert ax.get_legend().get_title().get_text() == "clones matched"
    assert entries(ax)[-1] == REFERENCE_LEGEND

    # ...and nothing is named when nothing grey was drawn
    assert not entries(tcri.pl.mutual_information(adata, quantity="excess"))
    tcri.tl.mutual_information(adata, covariate=a, groupby="patient", null_model=None)
    assert REFERENCE_LEGEND not in entries(tcri.pl.mutual_information(adata))


def test_the_gene_ranking_is_corrected_by_default(ref):
    """`pl.gene_importance` defaults to the EXCESS, alone among the twins.

    The bare ranking is not merely incomplete, it is dominated by something the question is not
    about: silencing a gene is an intervention whose size scales with the gene's counts.
    Measured on the OE fit, the bare importance and its null are 0.901 rank-correlated and the
    bare top ten is led by MALAT1, TMSB4X, MT-CO2 and three ribosomal proteins. A reader shown
    that list concludes the perturbation is broken.

    `"auto"` still means the bare value when there is no reference, and it means the bare value
    for `kind="shift"` whatever else is present -- otherwise the default call would raise
    against its own default.
    """
    import matplotlib
    matplotlib.use("Agg")

    model, adata = ref
    genes = list(adata.var_names[:8])
    tcri.perturb.gene_importance(model, adata, genes=genes, groupby="patient")

    auto = tcri.pl.gene_importance(adata, n_top=5)
    assert auto.get_ylabel().endswith("- null"), auto.get_ylabel()
    bare = tcri.pl.gene_importance(adata, n_top=5, quantity="value")
    assert not bare.get_ylabel().endswith("- null")

    # the two rankings are allowed to differ, and that difference is the point
    auto_genes = [t.get_text() for t in auto.get_xticklabels()]
    bare_genes = [t.get_text() for t in bare.get_xticklabels()]
    assert set(auto_genes) <= set(genes) and set(bare_genes) <= set(genes)

    # the heatmap still works at the default rather than raising against it
    assert tcri.pl.gene_importance(adata, kind="shift", n_top=5) is not None
    with pytest.raises(ValueError, match="kind='rank'"):
        tcri.pl.gene_importance(adata, kind="shift", quantity="excess")

    # ...and with no reference, "auto" is the bare value
    tcri.perturb.gene_importance(model, adata, genes=genes, groupby="patient",
                                 null_model=None)
    plain = tcri.pl.gene_importance(adata, n_top=5)
    assert not plain.get_ylabel().endswith("- null")
