import contextlib
import io
import logging

import pytest
import numpy as np
import pandas as pd
import warnings

import pyro
import torch
from anndata import AnnData
from scipy import sparse


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False,
        help="run slow statistical-recovery tests (model fits over many configs)",
    )
    parser.addoption(
        "--update-schema-snapshot", action="store_true", default=False,
        help="rewrite tests/snapshots/tl_schemas.json from what the tools store now",
    )


def pytest_configure(config):
    """Deterministic RNG for all tests (Notion T13)."""
    np.random.seed(42)
    torch.manual_seed(42)
    config.addinivalue_line(
        "markers",
        "slow: statistical-recovery test that fits models over several configs; "
        "skipped unless --runslow is passed",
    )


def pytest_collection_modifyitems(config, items):
    """Skip ``@pytest.mark.slow`` unless ``--runslow``.

    Recovery tests fit real models across a grid, so they are minutes-scale and do
    not belong in the per-commit suite — but they are the only tests with an
    accuracy oracle, so they must stay runnable (nightly / pre-release).
    """
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="needs --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


#: Parameters of the session-scoped fitted fixtures, captured when each is built and keyed by
#: the model name it owns. Restored after any test that clears the process-global store.
_FIXTURE_PARAMS: dict[str, dict] = {}


def _remember_fixture_params(name: str) -> None:
    store = pyro.get_param_store()
    _FIXTURE_PARAMS[name] = {
        k: v.detach().clone() for k, v in store.items() if k.startswith(f"{name}.")
    }


@pytest.fixture(autouse=True)
def _keep_the_fixture_params_alive():
    """Put the session fixtures' parameters back after a test that cleared the store.

    Pyro's store is process-global and 22 test files call ``clear_param_store()``; the
    session-scoped ``trained_model`` and ``cohort`` fixtures are built once and read their
    guide concentrations out of that store for the rest of the run, so any of those files
    running in between used to leave them reading another model's parameters or nothing at
    all. The perturbation work hit this twice -- once as an ``IndexError`` when two fixtures
    had different phenotype counts, once silently.

    This is only writable because the fixtures are NAMED (0.12): the keys a fixture owns are
    exactly those under ``f"{name}."``, so they can be restored without touching whatever the
    test under way put in the store. It restores rather than prevents, so a test is still free
    to clear the store for its own purposes.
    """
    yield
    if not _FIXTURE_PARAMS:
        return
    store = pyro.get_param_store()
    live = set(store.keys())
    with torch.no_grad():
        for name, saved in _FIXTURE_PARAMS.items():
            for key, value in saved.items():
                if key not in live:
                    store[key] = value.clone()


def _fit_the_nulls(model, adata):
    """Give a fitted fixture the permutation references every scored metric defaults to.

    Carried by the FIXTURE rather than passed as `null_model=None` at ~200 call sites, so the
    default path is the tested path: a metric called the way a user calls it computes a
    reference, writes a second `uns` key and returns the `excess` columns, and the suite sees
    all of that. The four tests that deliberately measure something else -- device parity,
    recovery accuracy, the delta intersection rule, real-data timing -- pass `null_model=None`
    and say why.

    Three more fits per fixture, on models this small, and they are the only place in the suite
    where a null is exercised through the ordinary reader surface.
    """
    import tcri

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tcri.null.all(model, adata, enable_progress_bar=False, enable_model_summary=False)


def _seed_all(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)


@pytest.fixture
def mock_adata():
    """Create a mock AnnData object with the necessary structure for testing."""
    n_cells = 100
    n_genes = 50
    X = sparse.random(n_cells, n_genes, density=0.1, format='csr')

    obs = pd.DataFrame({
        'unique_clone_id': [f'clone_{i}' for i in range(20)] * 5,
        'phenotype_col': ['A', 'B', 'C'] * 33 + ['A'],
        'timepoint': ['T1', 'T2'] * 50,
        'patient': ['P1', 'P2'] * 50,
    })

    adata = AnnData(X=X, obs=obs)

    adata.uns["tcri_metadata"] = {
        "covariate_col": "timepoint",
        "clone_col": "unique_clone_id",
        "phenotype_col": "phenotype_col",
        "batch_col": "patient"
    }

    adata.uns["tcri_phenotype_categories"] = ['A', 'B', 'C']
    adata.uns["tcri_clone_key"] = "unique_clone_id"
    adata.uns["tcri_phenotype_key"] = "phenotype_col"

    n_clones = 20
    n_phenotypes = 3
    mock_jd = np.random.dirichlet(np.ones(n_phenotypes), size=n_clones)
    mock_jd = pd.DataFrame(
        mock_jd,
        columns=['A', 'B', 'C'],
        index=[f'clone_{i}' for i in range(n_clones)]
    )
    adata.uns["mock_joint_distribution"] = mock_jd

    n_ct_pairs = 40
    adata.uns["tcri_p_ct"] = np.random.dirichlet(
        np.ones(n_phenotypes), size=n_ct_pairs
    ).astype(np.float32)
    adata.uns["tcri_ct_to_cov"] = torch.tensor([0, 1] * 20)
    adata.uns["tcri_ct_to_c"] = torch.tensor([i for i in range(20)] * 2)
    adata.uns["tcri_covariate_categories"] = ['T1', 'T2']
    adata.uns["tcri_clonotype_categories"] = [f'clone_{i}' for i in range(20)]
    adata.uns["tcri_ct_array_for_cells"] = np.random.randint(0, n_ct_pairs, size=n_cells)
    adata.uns["tcri_cov_array_for_cells"] = np.random.randint(0, 2, size=n_cells)
    adata.uns["tcri_local_scale"] = 5.0

    return adata


@pytest.fixture
def mock_joint_distribution():
    """Create a mock joint distribution DataFrame."""
    n_clones = 20
    n_phenotypes = 3
    base_dist = np.random.dirichlet(np.ones(n_phenotypes), size=n_clones)
    temp_dist = base_dist ** (1.0 / 0.5)
    temp_dist = temp_dist / temp_dist.sum(axis=1, keepdims=True)
    return pd.DataFrame(
        temp_dist,
        columns=['A', 'B', 'C'],
        index=[f'clone_{i}' for i in range(n_clones)]
    )


@pytest.fixture(scope="session")
def synthetic_adata():
    """Tiny deterministic AnnData for model fixtures (Notion T1)."""
    _seed_all(0)
    rng = np.random.default_rng(0)

    n_cells, n_genes, n_clones = 200, 50, 20
    phenotypes = ["A", "B", "C"]
    covariates = ["T1", "T2"]
    batches = ["P1", "P2"]

    X = rng.poisson(lam=1.5, size=(n_cells, n_genes)).astype(np.float32)

    patient = rng.choice(batches, size=n_cells)
    base_clone = rng.choice([f"clone_{i}" for i in range(n_clones)], size=n_cells)

    obs = pd.DataFrame({
        # patient-specific clone ids (disjoint across patients), as real `trb_unique` is —
        # so metric groupby='patient' is valid (clones don't span groups).
        "unique_clone_id": [f"{c}_{p}" for c, p in zip(base_clone, patient)],
        "phenotype_col": rng.choice(phenotypes, size=n_cells),
        "timepoint": rng.choice(covariates, size=n_cells),
        "patient": patient,
    })
    for col in obs.columns:
        obs[col] = obs[col].astype("category")

    return AnnData(X=X, obs=obs)


@pytest.fixture(scope="session")
def trained_model(synthetic_adata):
    """TCRIModel fit for 50 epochs on synthetic_adata, with to_anndata applied."""
    _seed_all(0)

    from tcri.model._model import TCRIModel

    adata = synthetic_adata.copy()
    TCRIModel.setup_anndata(
        adata,
        clonotype_key="unique_clone_id",
        phenotype_key="phenotype_col",
        covariate_key="timepoint",
        batch_key="patient",
    )
    # NAMED (0.12): this fixture owns the store keys under "trained." and nothing else, so it
    # coexists with `cohort` and with any model a test builds. It no longer clears the store on
    # the way in -- that is what used to wipe whichever fixture was built first.
    model = TCRIModel(
        adata,
        n_latent=8,
        n_hidden=16,
        n_layers=1,
        classifier_n_layers=1,
        classifier_hidden=16,
        K=3,
        n_pseudo_obs=3,
        name="trained",
    )

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model.train(
            max_epochs=50,
            batch_size=64,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        model.to_anndata(adata)
        _fit_the_nulls(model, adata)
    _remember_fixture_params("trained")
    return model, adata


#: (patient, cohort, omega_concentration). Low omega => sharp clone->phenotype coupling.
#: THREE patients per arm, not one. A contrast needs replicates: with one patient per arm a
#: Mann-Whitney returns p=1.0 whatever the data says, so nothing downstream of `splitby` --
#: the star, the bracket, the n that proves the unit is the patient -- is testable at all.
COHORT = (
    ("P0", "disease", 0.15), ("P1", "disease", 0.20), ("P2", "disease", 0.25),
    ("P3", "control", 1.20), ("P4", "control", 1.50), ("P5", "control", 1.80),
)


@pytest.fixture(scope="session")
def cohort():
    """A fitted 6-patient / 2-arm AnnData: the smallest thing that can carry a contrast.

    Returns ``(model, adata)``. Clone ids are patient-scoped, which is what makes ``groupby``
    legal -- the metric restricts by clone, so a clone spanning patients would let one
    patient's estimate absorb another's cells.
    """
    import anndata as ad
    import pyro

    from tcri.datasets import simulate_tcri
    from tcri.model._model import TCRIModel

    logging.disable(logging.INFO)
    parts = []
    for i, (patient, cohort, omega) in enumerate(COHORT):
        block = simulate_tcri(n_clones=8, n_phenotypes=4, n_genes=25, n_cells=200,
                              n_covariates=2, omega_concentration=omega, seed=i)
        block.obs["clone_id"] = block.obs["clone_id"].astype(str) + "@" + patient
        block.obs["patient"] = patient
        block.obs["disease_status"] = cohort
        block.obs_names = [f"{patient}_{j}" for j in range(block.n_obs)]
        parts.append(block)

    adata = ad.concat(parts, join="outer", label=None)
    for col in ("clone_id", "phenotype", "covariate", "patient", "disease_status"):
        adata.obs[col] = adata.obs[col].astype("category")
    adata.layers["counts"] = adata.X.copy()

    TCRIModel.setup_anndata(adata, layer="counts", clonotype_key="clone_id",
                            phenotype_key="phenotype", covariate_key="covariate",
                            batch_key="patient", replicate="patient")
    model = TCRIModel(adata, n_latent=8, n_hidden=16, n_layers=1, classifier_n_layers=1,
                      classifier_hidden=16, K=4, seed=0, name="cohort")
    with contextlib.redirect_stdout(io.StringIO()):
        model.train(max_epochs=10, batch_size=128, n_steps_kl_warmup=8, accelerator="cpu",
                    enable_progress_bar=False, enable_model_summary=False)
        model.to_anndata(adata)
        _fit_the_nulls(model, adata)
    _remember_fixture_params("cohort")
    logging.disable(logging.NOTSET)
    return model, adata
