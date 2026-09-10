"""``pl.gene_importance`` as a cache renderer: the same rules as every twin, plus the two
views it owns.

Every model call goes through the ``cohort`` fixture (see ``tests/test_perturbation.py`` for
why: the Pyro parameter store is process-global).
"""
import inspect

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import tcri
from tcri._state import keys as K
from tests.test_plotting.test_twins import METRIC_ARGS, _n_points, _n_violins

#: What the perturbation could compute on its own, on top of the metric arguments.
PERTURB_ARGS = METRIC_ARGS | {"genes", "use_gate", "batch_size", "model"}


def _without_replicate(adata):
    a = adata.copy()
    a.uns[K.METADATA] = {**a.uns[K.METADATA], K.Config.REPLICATE: None}
    return a


def test_pl_takes_no_perturbation_arguments():
    params = set(inspect.signature(tcri.pl.gene_importance).parameters)
    assert not (params & PERTURB_ARGS), f"pl.gene_importance can compute: {params & PERTURB_ARGS}"
    assert {"key", "kind", "n_top"} <= params


def test_pl_says_which_call_to_run(cohort):
    _, adata = cohort
    a = adata.copy()
    a.uns.pop(K.GENE_IMPORTANCE, None)
    with pytest.raises(KeyError, match=r"tcri\.perturb\.gene_importance\(model, adata"):
        tcri.pl.gene_importance(a)


def test_pl_never_recomputes(cohort, monkeypatch):
    model, adata = cohort
    a = adata.copy()
    tcri.perturb.gene_importance(model, a, genes=[0, 1, 2])

    def _boom(*args, **kwargs):
        raise AssertionError("pl.gene_importance recomputed the perturbation")

    monkeypatch.setattr(tcri.perturb, "gene_importance", _boom)
    monkeypatch.setattr("tcri.perturbation.gene_importance", _boom, raising=False)
    assert tcri.pl.gene_importance(a) is not None
    assert tcri.pl.gene_importance(a, kind="shift") is not None


def test_pl_return_df_is_the_cached_result(cohort):
    model, adata = cohort
    a = adata.copy()
    tcri.perturb.gene_importance(model, a, genes=[0, 1, 2])
    pd.testing.assert_frame_equal(tcri.pl.gene_importance(a, return_df=True),
                                  tcri.get.result(a, "gene_importance")["result"])


def test_kind_is_validated(cohort):
    model, adata = cohort
    a = adata.copy()
    tcri.perturb.gene_importance(model, a, genes=[0])
    with pytest.raises(ValueError, match="kind must be"):
        tcri.pl.gene_importance(a, kind="nonsense")


def test_rank_view_shows_the_top_genes_as_replicate_dots(cohort):
    """The x axis is the ``n_top`` most important genes in descending order, and every dot is a
    patient -- the replicate unit -- never a draw or a cell."""
    model, adata = cohort
    a = adata.copy()
    res = tcri.perturb.gene_importance(model, a)
    ax = tcri.pl.gene_importance(a, n_top=5)

    ranked = (res["result"].groupby("gene", observed=True)["value"].mean()
              .sort_values(ascending=False).index.tolist()[:5])
    assert [t.get_text() for t in ax.get_xticklabels()] == ranked
    assert ax.get_xlabel() == "gene"
    n_patients = a.obs["patient"].nunique()
    assert _n_points(ax) == 5 * n_patients, "the dots are not one per patient per gene"
    assert _n_violins(ax) == 0


def test_rank_view_stars_each_genes_own_contrast(cohort):
    """With a split, the hue is the split and the text above each gene is THAT gene's contrast
    from ``stats`` -- a per-gene test over patients, never a bracket between genes."""
    model, adata = cohort
    a = adata.copy()
    res = tcri.perturb.gene_importance(model, a, splitby="disease_status")
    ax = tcri.pl.gene_importance(a, n_top=6)
    genes = [t.get_text() for t in ax.get_xticklabels()]
    stars = res["stats"].set_index("gene")["stars"]
    assert [t.get_text() for t in ax.texts] == [stars[g] or "ns" for g in genes]
    assert ax.get_legend() is not None, "the split levels need a legend"
    assert not [l for l in ax.lines if len(set(np.round(l.get_xdata(), 6))) > 1
                and np.ptp(l.get_xdata()) >= 1.0], "a line spans two genes"

    tcri.perturb.gene_importance(model, a)
    assert not tcri.pl.gene_importance(a, n_top=6).texts, "stars drawn without a contrast"


def test_rank_view_without_groups_falls_back_to_points_then_violins(cohort):
    model, adata = cohort
    a = _without_replicate(adata)
    tcri.perturb.gene_importance(model, a, genes=[0, 1, 2, 3])
    pts = tcri.pl.gene_importance(a)
    assert _n_points(pts) == 4 and _n_violins(pts) == 0

    tcri.perturb.gene_importance(model, a, genes=[0, 1, 2, 3], n_samples=6, random_state=0)
    vio = tcri.pl.gene_importance(a)
    assert _n_violins(vio) == 4, "with only draws varying, each gene is a violin of its draws"


def test_shift_view_is_a_gene_by_phenotype_heatmap_centred_on_zero(cohort):
    model, adata = cohort
    a = adata.copy()
    res = tcri.perturb.gene_importance(model, a)
    ax = tcri.pl.gene_importance(a, kind="shift", n_top=7)

    assert len(ax.images) == 1
    im = ax.images[0]
    phenotypes = list(a.uns[K.PHENOTYPE_CATEGORIES])
    assert im.get_array().shape == (7, len(phenotypes))
    assert im.norm.vmin == -im.norm.vmax, "the diverging scale is not centred on zero"
    assert [t.get_text() for t in ax.get_xticklabels()] == [str(p) for p in phenotypes]

    genes = [t.get_text() for t in ax.get_yticklabels()]
    ranked = (res["result"].groupby("gene", observed=True)["value"].mean()
              .sort_values(ascending=False).index.tolist()[:7])
    assert genes == ranked
    want = (res["shift"][res["shift"]["gene"] == genes[0]]
            .groupby("phenotype", observed=True)["shift"].mean().reindex(phenotypes))
    np.testing.assert_allclose(np.asarray(im.get_array())[0], want.to_numpy(), atol=1e-12)
    # each row is a zero-sum shift: what one phenotype loses another gains
    assert np.abs(np.asarray(im.get_array()).sum(axis=1)).max() < 1e-5


def test_order_restricts_and_orders_the_genes(cohort):
    model, adata = cohort
    a = adata.copy()
    tcri.perturb.gene_importance(model, a, genes=[0, 1, 2, 3, 4])
    ax = tcri.pl.gene_importance(a, order=["gene_3", "gene_0", "not_a_gene"])
    assert [t.get_text() for t in ax.get_xticklabels()] == ["gene_3", "gene_0"]
    heat = tcri.pl.gene_importance(a, kind="shift", order=["gene_3", "gene_0"])
    assert [t.get_text() for t in heat.get_yticklabels()] == ["gene_3", "gene_0"]
