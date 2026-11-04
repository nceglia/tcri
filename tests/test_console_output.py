"""What tcri prints, and what it leaves alone.

The rule: tcri never changes the caller's warning or logging state. Its quieting of scvi and
Lightning is scoped to its own calls and restored on exit, and its own output -- warnings, the
notices on the ``tcri`` logger, tables, panels and bars -- each has one switch.

Several checks run in a subprocess, because pytest configures the root logger and the warning
filters of the process it runs in, and the behavior under test is what a bare script sees.
"""
from __future__ import annotations

import contextlib
import inspect
import io
import json
import logging
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import pandas as pd
import pytest

import tcri
from tcri import _output
from tcri._output import _logging, _quiet, _results

ROOT = Path(tcri.__file__).resolve().parent.parent


def _run(code: str, env=None) -> subprocess.CompletedProcess:
    """Run ``code`` in a fresh interpreter with this checkout's tcri; capture both streams."""
    import os

    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True,
                          text=True, cwd=ROOT, timeout=600, env={**os.environ, **(env or {})})


_SMALL_FIT = """
    import tcri
    from tcri.model import TCRIModel
    adata = tcri.datasets.simulate_cohort(n_patients=4, n_cells_per_sample=120, seed=0)
    TCRIModel.setup_anndata(adata, layer="counts", clonotype_key="clone_id",
                            phenotype_key="phenotype", covariate_key="timepoint",
                            batch_key="patient", replicate="patient")
    model = TCRIModel(adata, seed=0)
"""


def _small_model():
    from tcri.model import TCRIModel

    adata = tcri.datasets.simulate_cohort(n_patients=4, n_cells_per_sample=120, seed=0)
    TCRIModel.setup_anndata(adata, layer="counts", clonotype_key="clone_id",
                            phenotype_key="phenotype", covariate_key="timepoint",
                            batch_key="patient", replicate="patient")
    return TCRIModel(adata, seed=0), adata


# ── the caller's state ─────────────────────────────────────────────────────────────────────

def _state():
    names = ("", "tcri", "lightning.pytorch", "lightning.fabric", "scvi",
             "lightning.fabric.utilities.seed", "scvi.data._download",
             "lightning.pytorch.utilities.rank_zero")
    return (list(warnings.filters),
            {n: (logging.getLogger(n).level, list(logging.getLogger(n).filters),
                 list(logging.getLogger(n).handlers), logging.getLogger(n).propagate)
             for n in names})


def test_a_fit_leaves_the_callers_warning_and_logging_state_alone():
    model, _adata = _small_model()
    before = _state()
    model.train(max_epochs=2, print=False, enable_progress_bar=False)
    assert _state() == before


def test_a_fit_that_raises_leaves_the_state_alone_too():
    import lightning.pytorch as pl

    class Boom(pl.Callback):
        def on_train_epoch_end(self, trainer, module):
            raise RuntimeError("boom")

    model, _adata = _small_model()
    before = _state()
    with pytest.raises(RuntimeError, match="boom"):
        model.train(max_epochs=3, print=False, enable_progress_bar=False, callbacks=[Boom()])
    assert _state() == before


def test_a_load_leaves_the_state_alone(tmp_path):
    model, adata = _small_model()
    model.train(max_epochs=1, print=False, enable_progress_bar=False)
    tcri.ut.save_tcri_session(model, adata, str(tmp_path))
    before = _state()
    tcri.ut.load_tcri_session(str(tmp_path), print=False)
    assert _state() == before


def test_a_callers_own_warning_still_shows_after_a_fit():
    done = _run(_SMALL_FIT + """
    import warnings
    model.train(max_epochs=2, print=False, enable_progress_bar=False)
    warnings.warn("the caller's own warning")
    """)
    assert done.returncode == 0, done.stderr
    assert "the caller's own warning" in done.stderr


def test_importing_tcri_changes_nothing_but_its_own_logger():
    """Its dependencies add warning filters when they are imported; tcri itself adds none, and
    leaves the root logger, every other logger and the environment alone."""
    done = _run("""
    import json, logging, os, warnings
    import torch, pyro, lightning.pytorch, scvi, scanpy, anndata, rich, seaborn
    os.environ["SLURM_NTASKS"] = "7"
    filters = list(warnings.filters)
    root = (logging.getLogger().level, list(logging.getLogger().handlers))
    others = {n: (l.level, list(l.handlers), l.propagate)
              for n, l in logging.root.manager.loggerDict.items()
              if isinstance(l, logging.Logger) and not n.startswith("tcri")}
    environ = dict(os.environ)
    import tcri
    print(json.dumps({
        "filters": warnings.filters == filters,
        "root": (logging.getLogger().level, list(logging.getLogger().handlers)) == root,
        "others": all((logging.getLogger(n).level, list(logging.getLogger(n).handlers),
                       logging.getLogger(n).propagate) == v for n, v in others.items()),
        "environ": dict(os.environ) == environ,
    }))
    """)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.strip().splitlines()[-1]) == {
        "filters": True, "root": True, "others": True, "environ": True}


# ── the named third-party lines ────────────────────────────────────────────────────────────

#: The scvi and Lightning lines tcri drops during a fit.
_SILENCED = ("Seed set to", "GPU available:", "TPU available:", "litlogger",
             "does not have many workers", "log_every_n_steps", "LeafSpec",
             "has been automatically set to `cpu`", "GPU available but not used",
             "`Trainer.fit` stopped")


@pytest.mark.parametrize("env", [{}, {"SLURM_PROCID": "0"}], ids=["plain", "slurm-rank"])
def test_a_fit_prints_none_of_the_named_lines(env):
    """If scvi or Lightning rewords a line on the list, it shows up here and this fails. Under
    Slurm, Lightning prefixes its lines with the rank."""
    done = _run(_SMALL_FIT + """
    model.train(max_epochs=2, print=False, enable_progress_bar=False)
    """, env=env)
    assert done.returncode == 0, done.stderr
    out = done.stdout + done.stderr
    shown = [line for line in _SILENCED if line in out]
    assert not shown, out


def test_lines_off_the_list_still_show():
    record = logging.LogRecord("lightning.pytorch.utilities.rank_zero", logging.INFO, "", 0,
                               "Detected KeyboardInterrupt, attempting graceful shutdown ...",
                               None, None)
    assert _quiet._DropNamed().filter(record)
    # the quieting opens inside the capture, as it does inside a caller's own filters, so its
    # ignore filters are the ones checked first
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with _output.quiet_third_party():
            warnings.warn("GPU available but not used. You can set it by doing ...")
            warnings.warn("an unlisted warning")
    assert [str(w.message) for w in caught] == ["an unlisted warning"]


def test_tcris_own_warnings_still_show_and_name_the_callers_line():
    model, adata = _small_model()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.train(max_epochs=2, print=False, enable_progress_bar=False)
    ours = [w for w in caught if "max_epochs" in str(w.message)]
    assert ours, [str(w.message) for w in caught]
    assert all(Path(w.filename).resolve() == Path(__file__).resolve() for w in ours)


def test_a_warning_raised_deep_in_tcri_names_the_callers_line(cohort):
    """The missing-group warning is raised in the group loop, under the ``tl_result``
    decorator; it is a data warning and names this file."""
    _model, adata = cohort
    adata = adata.copy()
    adata.obs["group"] = adata.obs["patient"].astype(str).where(
        adata.obs["patient"] != "P5").astype("category")
    levels = list(adata.obs["covariate"].cat.categories)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tcri.tl.phenotypic_entropy(adata, cov_from=levels[0], cov_to=levels[1], groupby="group",
                                   null_model=None, print=False, inplace=False)
    ours = [w for w in caught if "have no group label" in str(w.message)]
    assert len(ours) == 1, [str(w.message) for w in caught]
    assert issubclass(ours[0].category, tcri.TCRIDataWarning)
    assert Path(ours[0].filename).resolve() == Path(__file__).resolve()


# ── the tcri logger ────────────────────────────────────────────────────────────────────────

def test_notices_print_when_nothing_is_configured():
    done = _run("""
    import logging, tcri
    logging.getLogger("tcri.model._model").info("a notice from a child logger")
    logging.getLogger("tcri.model._model").debug("a debug line")
    """)
    assert done.returncode == 0, done.stderr
    assert "a notice from a child logger" in done.stderr and "INFO" in done.stderr
    assert "a debug line" not in done.stderr


def test_a_configured_root_logger_decides():
    done = _run("""
    import logging, sys, tcri
    logging.basicConfig(level="WARNING", stream=sys.stdout, format="%(levelname)s:%(message)s")
    logging.getLogger("tcri.model._model").info("kept out at WARNING")
    logging.getLogger("tcri.model._model").warning("a warning passes")
    logging.getLogger("tcri").setLevel("INFO")
    logging.getLogger("tcri.model._model").info("let in by the user's level")
    """)
    assert done.returncode == 0, done.stderr
    assert "kept out at WARNING" not in done.stdout + done.stderr
    assert done.stdout.count("WARNING:a warning passes") == 1
    assert done.stdout.count("INFO:let in by the user's level") == 1
    assert "let in" not in done.stderr        # the application's handler, not tcri's


def test_caplog_captures_each_record_once(caplog):
    with caplog.at_level(logging.INFO, logger="tcri"):
        logging.getLogger("tcri.model._model").info("captured once")
    assert [r.getMessage() for r in caplog.records].count("captured once") == 1


def test_the_tcri_logger_has_one_handler_and_does_not_propagate():
    log = logging.getLogger("tcri")
    assert log.propagate is False
    assert sum(isinstance(h, _logging._Router) for h in log.handlers) == 1
    _output.install_handler()
    assert sum(isinstance(h, _logging._Router) for h in log.handlers) == 1


# ── switches ───────────────────────────────────────────────────────────────────────────────

def test_print_and_the_bar_are_display_only(cohort, capsys):
    model, adata = cohort
    adata = adata.copy()                      # the session fixture stays without a stored result
    tcri.tl.mutual_information(adata, covariate=adata.obs["covariate"].cat.categories[0],
                               splitby="disease_status", print=False)
    assert capsys.readouterr().out == ""
    params = tcri.get.params(adata, "mutual_information")
    assert "print" not in params and "enable_progress_bar" not in params
    assert "print" not in model.train_kwargs_ and "enable_progress_bar" not in model.train_kwargs_


def test_a_metric_prints_one_table_and_its_reference_runs_print_nothing(cohort, capsys):
    _model, adata = cohort
    adata = adata.copy()                      # the session fixture stays without a stored result
    level = adata.obs["covariate"].cat.categories[0]
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        tcri.tl.mutual_information(adata, covariate=level, splitby="disease_status", print=True)
    out = capsys.readouterr().out
    assert out.count("mutual_information ·") == 1
    assert "against null.phenotype (2 fits)" in out


def test_reference_runs_raise_no_warnings(cohort):
    """Two fits per reference: without the silence the missing-group warning would show three
    times, once for the caller's call and once for each run on the reference."""
    _model, adata = cohort
    adata = adata.copy()
    adata.obs["group"] = adata.obs["patient"].astype(str).where(
        adata.obs["patient"] != "P5").astype("category")
    levels = list(adata.obs["covariate"].cat.categories)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tcri.tl.phenotypic_entropy(adata, cov_from=levels[0], cov_to=levels[1], groupby="group",
                                   print=False, inplace=False)
    missing = [w for w in caught if "have no group label" in str(w.message)]
    assert len(missing) == 1, [str(w.message) for w in missing]


def test_a_progress_bar_callback_raises():
    from lightning.pytorch.callbacks import TQDMProgressBar

    model, _adata = _small_model()
    with pytest.raises(TypeError, match="draws its own status bar"):
        model.train(max_epochs=1, callbacks=[TQDMProgressBar()])
    with pytest.raises(TypeError, match="simple_progress_bar"):
        model.train(max_epochs=1, simple_progress_bar=False)


# ── the null set ───────────────────────────────────────────────────────────────────────────

def test_null_fits_report_each_condition_once(capsys):
    model, adata = _small_model()
    model.train(max_epochs=2, print=False, enable_progress_bar=False)
    model.to_anndata(adata)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tcri.null.phenotype(model, adata, n_nulls=2, enable_progress_bar=False)
    texts = [str(w.message) for w in caught]
    assert sum("null fits stopped at max_epochs" in t for t in texts) == 1, texts
    assert sum("training stopped because it reached max_epochs" in t for t in texts) == 0
    out = capsys.readouterr().out
    assert "null fits ·" in out and "2 reached max_epochs" in out


def test_a_silent_null_fit_ignores_a_callers_error_filter():
    """The parent's batch fills the whole object, a condition the set does not report when the
    parent ran with it; under an "error" filter a null fit's own warning must not raise."""
    model, adata = _small_model()
    model.train(max_epochs=1, print=False, enable_progress_bar=False)   # batch_size >= n_obs
    model.to_anndata(adata)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="batch_size=")
        tcri.null.phenotype(model, adata, n_nulls=1, print=False, enable_progress_bar=False)
    assert tcri.get.fits(adata) == ["null.phenotype[0]"]


def test_a_null_set_that_fails_partway_reports_what_it_wrote(monkeypatch):
    from tcri.null import _nulls

    model, adata = _small_model()
    model.train(max_epochs=2, print=False, enable_progress_bar=False)
    model.to_anndata(adata)
    real = _nulls._fit_member
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("second fit fails")
        return real(*args, **kwargs)

    monkeypatch.setattr(_nulls, "_fit_member", flaky)
    with warnings.catch_warnings(record=True) as caught, \
            pytest.raises(RuntimeError, match="second fit fails"):
        warnings.simplefilter("always")
        tcri.null.phenotype(model, adata, n_nulls=3, print=False, enable_progress_bar=False)
    assert any("1 of 1 null fits stopped at max_epochs" in str(w.message) for w in caught)


# ── the training record and the panel ──────────────────────────────────────────────────────

def test_the_record_carries_device_duration_reason_and_the_call_seed(tmp_path):
    model, adata = _small_model()
    model.train(max_epochs=2, print=False, enable_progress_bar=False)
    record = model.training_record_
    assert isinstance(record["device"], str) and record["device"]
    assert isinstance(record["duration_s"], float) and record["duration_s"] > 0
    assert record["stop_reason"] == "max_epochs"
    assert record["call_seed"] == 0
    model.train(max_epochs=1, print=False, enable_progress_bar=False)
    assert model.training_record_["call_seed"] == 1 and model.training_record_["seed"] == 0

    model.to_anndata(adata)
    tcri.null.phenotype(model, adata, n_nulls=1, print=False, enable_progress_bar=False)
    path = tmp_path / "a.h5ad"
    adata.write_h5ad(path)
    import anndata as ad

    from tcri._state import keys as K

    uns = ad.read_h5ad(path).uns
    # the main fit's record, as the model holds it, and its loss curves
    main = uns[K.TRAINING]
    expected = {k: v for k, v in model.training_record_.items() if v is not None}
    assert set(main["record"]) == set(expected)
    assert main["record"]["stop_reason"] == "max_epochs" and int(main["record"]["call_seed"]) == 1
    assert isinstance(main["record"]["device"], str)
    assert set(main["history"]) == set(model.history_)
    # each null's, under its own fit
    back = uns[K.fit_key(K.TRAINING, "null.phenotype[0]")]["record"]
    assert isinstance(back["device"], str)
    assert float(back["duration_s"]) > 0
    assert back["stop_reason"] == "max_epochs"
    assert int(back["call_seed"]) == 0


class _RaiseAt:
    """A Lightning callback that raises at ``hook``, from epoch ``epoch`` on."""

    def __new__(cls, hook, epoch=0):
        from lightning.pytorch.callbacks import Callback

        def raise_(self, trainer, *_):
            if trainer.current_epoch >= epoch:
                raise RuntimeError(f"raised at {hook}")

        return type("RaiseAt", (Callback,), {hook: raise_})()


def test_a_call_that_raises_records_what_it_did(tmp_path):
    """A call that raised after its first optimizer step is recorded as an error, so the record
    beside the moved parameters is that call's; a call that raised before it changed nothing."""
    model, adata = _small_model()
    model.train(max_epochs=2, print=False, enable_progress_bar=False)
    first = dict(model.training_record_)

    with pytest.raises(TypeError):                     # an argument error: no runner at all
        model.train(max_epochs=2, simple_progress_bar=True, print=False)
    assert model.training_record_ == first
    with pytest.raises(RuntimeError, match="on_train_start"):    # before the first step
        model.train(max_epochs=2, callbacks=[_RaiseAt("on_train_start")], print=False,
                    enable_progress_bar=False)
    assert model.training_record_ == first

    with pytest.raises(RuntimeError, match="on_train_epoch_start"):
        model.train(max_epochs=5, callbacks=[_RaiseAt("on_train_epoch_start", epoch=1)],
                    print=False, enable_progress_bar=False)
    record = model.training_record_
    assert record["stop_reason"] == "error" and record["epochs_run"] == 1
    assert record["selected_epoch"] is None and record["stopped_early"] is False
    assert record["call_seed"] != first["call_seed"]
    assert "ended by an error" in str(model)
    model.to_anndata(adata)
    assert adata.uns["tcri_training"]["record"]["stop_reason"] == "error"


def test_stop_reason_tells_a_diverged_fit_from_patience():
    from lightning.pytorch.callbacks.early_stopping import EarlyStoppingReason

    from tcri.model._model import _stop_reason

    class Stopper:
        fired = True
        stopping_reason = EarlyStoppingReason.NON_FINITE_METRIC

    assert _stop_reason(Stopper(), 10, 100) == "non-finite objective"
    Stopper.stopping_reason = EarlyStoppingReason.PATIENCE_EXHAUSTED
    assert _stop_reason(Stopper(), 10, 100) == "patience"
    assert _stop_reason(None, 100, 100) == "max_epochs"
    assert _stop_reason(None, 10, 100) == "other limit"


def test_the_end_of_fit_panel_is_the_models_display(capsys):
    model, _adata = _small_model()
    capsys.readouterr()                       # the setup panel
    model.train(max_epochs=2, enable_progress_bar=False)
    printed = capsys.readouterr().out
    assert printed == str(model)
    for row in ("Training", "KL ramp", "Data", "Device", "Time", "reached max_epochs"):
        assert row in printed
    assert repr(model).startswith("TCRIModel(name=")


def test_the_status_line_ends_where_the_stopping_rule_ended(monkeypatch):
    """The bar is on (the default); its last line is the epoch the fit ended at, as the record
    numbers it."""
    from tcri.model._callbacks import RampGatedEarlyStopping

    monkeypatch.setattr(RampGatedEarlyStopping, "step_floor", 0)
    model, _adata = _small_model()
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        model.train(max_epochs=200, n_epochs_kl_warmup=0, early_stopping_patience=2,
                    print=False)
    record = model.training_record_
    assert record["stop_reason"] == "patience"
    last = [line for line in err.getvalue().splitlines() if line.startswith("train  epoch")][-1]
    assert last.startswith(f"train  epoch {record['epochs_run'] - 1} ")
    assert "patience 0 left" in last


# ── tables ─────────────────────────────────────────────────────────────────────────────────

def test_a_table_in_a_file_is_never_cut():
    import pandas as pd

    # six rows, so the table is not transposed: the long label folds inside its cell
    frame = pd.DataFrame({"label": ["an_unbroken_label_far_longer_than_its_column_can_hold"] * 6,
                          "value": [0.123456789] * 6})
    text = _output.plain_text(_output.frame_table(frame, title="t"), width=40)
    assert "…" not in text
    assert text.count("can_hold") == 6          # the label's tail, folded onto its own line


def test_the_interval_label_matches_the_summaries():
    from tcri._compute._tables import summarize

    prob = inspect.signature(summarize).parameters["hdi_prob"].default
    assert _results.INTERVAL == f"{round(prob * 100)}% HDI"


def test_a_bar_over_passes_is_opened_once():
    with _output.pass_progress(False, "x") as outer:
        assert outer is None
    with contextlib.redirect_stderr(io.StringIO()):
        with _output.pass_progress(True, "x") as outer:
            assert _output.current_passes() is outer
            with _output.pass_progress(True, "y") as inner:
                assert inner is outer
    assert _output.current_passes() is None


def test_gene_importance_draws_one_bar_over_its_passes(cohort):
    model, adata = cohort
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        tcri.perturb.gene_importance(model, adata, genes=list(adata.var_names[:2]),
                                     covariate=adata.obs["covariate"].cat.categories[0],
                                     inplace=False, print=False)
    lines = [line for line in err.getvalue().splitlines() if line.startswith("gene_importance")]
    assert len(lines) == 1, err.getvalue()
    assert lines[0].startswith("gene_importance 3/3 · ")    # the fit, then each of 2 nulls


def test_pool_rare_clones_prints_its_table(capsys):
    adata = tcri.datasets.simulate_cohort(n_patients=2, n_cells_per_sample=60, seed=0)
    tcri.pp.pool_rare_clones(adata, clonotype_key="clone_id", groupby="patient", print=True)
    out = capsys.readouterr().out
    assert "pool_rare_clones · clone_id → clone_id_pooled" in out
    assert "clones kept" in out and "P01" in out
    tcri.pp.pool_rare_clones(adata, clonotype_key="clone_id", groupby="patient", print=False)
    assert capsys.readouterr().out == ""


def test_saving_a_session_raises_no_warning_of_tcris_own(tmp_path):
    model, adata = _small_model()
    model.train(max_epochs=1, print=False, enable_progress_bar=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tcri.ut.save_tcri_session(model, adata, str(tmp_path))
    assert not [w for w in caught if "__version__" in str(w.message)]
    assert json.loads((tmp_path / "meta.json").read_text())["versions"]["anndata"] != "unknown"


# ── second round: setup, pp summaries, to_anndata, NullSet, save, get.show, diag ───────────

def test_setup_prints_what_it_registered(capsys):
    from tcri.model import TCRIModel

    adata = tcri.datasets.simulate_cohort(n_patients=2, n_cells_per_sample=60, seed=0)
    TCRIModel.setup_anndata(adata, layer="counts", clonotype_key="clone_id",
                            phenotype_key="phenotype", covariate_key="timepoint",
                            batch_key="patient", replicate="patient")
    out = capsys.readouterr().out
    for row in ("setup_anndata", "Clonotype", "clone_id", "Covariate", "pre, post", "Replicate"):
        assert row in out
    TCRIModel.setup_anndata(adata, layer="counts", clonotype_key="clone_id",
                            phenotype_key="phenotype", covariate_key="timepoint",
                            batch_key="patient", print=False)
    assert capsys.readouterr().out == ""


def test_print_is_not_part_of_the_setup_a_load_replays():
    model, _adata = _small_model()
    args = model.adata_manager.registry["setup_args"]
    assert "clonotype_key" in args and "print" not in args


def test_a_load_prints_one_panel(tmp_path, capsys):
    model, adata = _small_model()
    model.train(max_epochs=1, print=False, enable_progress_bar=False)
    tcri.ut.save_tcri_session(model, adata, str(tmp_path), print=False)
    capsys.readouterr()
    tcri.ut.load_tcri_session(str(tmp_path), print=True)
    out = capsys.readouterr().out
    assert out.count("╭─") == 1 and "session" in out and "setup_anndata" not in out


def _equal(a, b):
    if isinstance(a, dict):
        return set(a) == set(b) and all(a[k].equals(b[k]) for k in a)
    return a.equals(b)


@pytest.mark.parametrize("name, kwargs", [
    ("repertoire_summary", dict(clonotype_key="clone_id", groupby="patient")),
    ("repertoire_summary", dict(clonotype_key="clone_id", groupby="patient",
                                splitby="disease_status")),
    ("clone_persistence", dict(clonotype_key="clone_id", covariate_key="timepoint",
                               groupby="patient")),
    ("clonotype_sharing", dict(clonotype_key="clone_id", groupby="patient")),
    ("clonotype_sharing", dict(clonotype_key="clone_id", groupby="patient",
                               splitby="disease_status")),
])
def test_the_pp_summaries_store_and_print_one_table(name, kwargs, capsys):
    """Stored and ``None``, one table titled by the summary and its groupby (and splitby),
    nothing with ``print=False``; the returned result is the stored one."""
    adata = tcri.datasets.simulate_cohort(n_patients=4, n_cells_per_sample=60, seed=0)
    assert getattr(tcri.pp, name)(adata, **kwargs, print=True) is None
    out = capsys.readouterr().out
    title = " · ".join([name, kwargs["groupby"], *([kwargs["splitby"]] if "splitby" in kwargs
                                                   else [])])
    assert out.count(f"{name} ·") == 1 and title in out, out
    if name == "clonotype_sharing":
        assert f"tcri.get.result(adata, '{name}')['pairs']" in out
        assert f"tcri.get.result(adata, '{name}')['clonotypes']" in out
    returned = getattr(tcri.pp, name)(adata, **kwargs, inplace=False, print=False)
    assert capsys.readouterr().out == ""
    assert _equal(returned, tcri.get.result(adata, name))


def test_an_empty_persistence_prints_one_line(capsys):
    adata = tcri.datasets.simulate_cohort(n_patients=2, n_cells_per_sample=60, seed=0)
    first = adata.obs["timepoint"].cat.categories[0]
    adata = adata[adata.obs["timepoint"] == first].copy()
    tcri.pp.clone_persistence(adata, clonotype_key="clone_id", covariate_key="timepoint",
                              groupby="patient", print=True)
    out = capsys.readouterr().out.strip()
    assert out == ("clone_persistence · patient: no group has clones at two levels of "
                   "timepoint"), out


def test_to_anndata_writes_in_place_and_refuses_a_view():
    model, adata = _small_model()
    model.train(max_epochs=1, print=False, enable_progress_bar=False)
    assert model.to_anndata(adata) is None
    with pytest.raises(ValueError, match="is a view"):
        model.to_anndata(adata[:10])


def test_null_all_returns_a_null_set_that_displays_as_its_panel():
    from tcri.null._nulls import NullSet

    model, adata = _small_model()
    model.train(max_epochs=2, print=False, enable_progress_bar=False)
    model.to_anndata(adata)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nulls = tcri.null.all(model, adata, n_nulls=1, print=False, enable_progress_bar=False)
    assert isinstance(nulls, NullSet) and isinstance(nulls, dict)
    assert set(nulls) == {"phenotype", "clonotype", "covariate"}
    assert repr(nulls) == "NullSet(phenotype=1, clonotype=1, covariate=1)"
    assert "null fits ·" in str(nulls) and "null.phenotype" in str(nulls)
    assert "text/html" in nulls._repr_mimebundle_()
    panel = str(model)
    assert "Nulls" in panel and "per fit" in panel


def test_save_returns_its_manifest_and_prints_on_request(tmp_path, capsys):
    model, adata = _small_model()
    model.train(max_epochs=1, print=False, enable_progress_bar=False)
    capsys.readouterr()                       # setup's panel
    manifest = tcri.ut.save_tcri_session(model, adata, str(tmp_path / "a"))
    assert capsys.readouterr().out == "" and "pyro" in manifest
    assert "pyro" in tcri.ut.save_tcri_session(model, adata, str(tmp_path / "b"), print=True)
    assert "save_tcri_session" in capsys.readouterr().out


def test_get_show_reprints_and_prints_any_level(cohort, capsys):
    _model, adata = cohort
    adata = adata.copy()
    level = adata.obs["covariate"].cat.categories[0]
    tcri.tl.mutual_information(adata, covariate=level, splitby="disease_status", print=True)
    printed = capsys.readouterr().out
    assert tcri.get.show(adata, "mutual_information") is None
    assert capsys.readouterr().out == printed
    tcri.get.show(adata, "mutual_information", level="stats")
    out = capsys.readouterr().out
    assert "field" in out and "delta" in out            # one wide row, printed transposed
    tcri.get.show(adata, "mutual_information", level="draws")
    assert "draw" in capsys.readouterr().out
    with pytest.raises(KeyError, match="level='nope'"):
        tcri.get.show(adata, "mutual_information", level="nope")
    with pytest.raises(KeyError):
        tcri.get.show(adata, "mutual_information", level="item")


_DIAG = {
    "joint_distribution_ppc": lambda m, a, **k: tcri.diag.joint_distribution_ppc(
        a, n_samples=2, n_sims=5, random_state=0, plot=False, **k),
    "phenotype_calibration": lambda m, a, **k: tcri.diag.phenotype_calibration(a, plot=False, **k),
    "reconstruction_ppc": lambda m, a, **k: tcri.diag.reconstruction_ppc(m, a, n_sims=3,
                                                                        plot=False, **k),
    "reconstruction_ppc_seeded": lambda m, a, **k: tcri.diag.reconstruction_ppc(
        m, a, n_sims=3, random_state=0, plot=False, **k),
    "permutation_null": lambda m, a, **k: tcri.diag.permutation_null(a, n_perm=20, plot=False,
                                                                    **k),
}


@pytest.mark.parametrize("check", list(_DIAG))
def test_each_diag_check_prints_returns_and_refuses_doing_nothing(check, cohort, capsys):
    model, adata = cohort
    call = _DIAG[check]
    title = check.removesuffix("_seeded")
    result = call(model, adata)
    assert isinstance(result, dict) and title in capsys.readouterr().out
    assert call(model, adata, return_dict=False) is None
    assert title in capsys.readouterr().out
    assert call(model, adata, print=False) is not None
    assert capsys.readouterr().out == ""
    with pytest.raises(ValueError, match="nothing to do"):
        call(model, adata, print=False, return_dict=False)


def test_a_cut_diag_table_says_how_to_get_the_rest(cohort, capsys):
    _model, adata = cohort
    tcri.diag.phenotype_calibration(adata, n_bins=30, plot=False, return_dict=False)
    assert "return_dict=True" in capsys.readouterr().out
    tcri.diag.phenotype_calibration(adata, n_bins=30, plot=False)
    assert "the returned 'bins'" in capsys.readouterr().out


def test_get_show_reads_the_other_stored_shapes(cohort, capsys):
    _model, adata = cohort
    adata = adata.copy()
    levels = list(adata.obs["covariate"].cat.categories)
    tcri.tl.phenotypic_flux(adata, cov_from=levels[0], cov_to=levels[1], print=False)
    tcri.get.show(adata, "phenotypic_flux")
    out = capsys.readouterr().out
    assert "adjusted vs 0: mean" in out and "signed-rank p" in out
    tcri.get.show(adata, "phenotypic_flux", level="item")
    assert "phenotypic_flux · item" in capsys.readouterr().out
    tcri.get.show(adata, "phenotypic_flux", level="stats")
    assert "covariate" in capsys.readouterr().out
    with pytest.raises(KeyError, match="has the levels"):
        tcri.get.show(adata, "phenotypic_flux", level="nope")
    tcri.tl.joint_distribution(adata, covariate=levels[0], print=False)
    tcri.get.show(adata, "joint_distribution", level="summary")
    assert "clonotype" in capsys.readouterr().out     # the index prints as a column

    tcri.pp.clonotype_sharing(adata, clonotype_key="clone_id", groupby="patient", print=False)
    tcri.get.show(adata, "clonotype_sharing", level="pairs")
    out = capsys.readouterr().out
    assert "clonotype_sharing · pairs" in out and "patient_a" in out
    tcri.pp.repertoire_summary(adata, clonotype_key="clone_id", groupby="patient", print=False)
    tcri.get.show(adata, "repertoire_summary")
    assert "repertoire_summary · patient" in capsys.readouterr().out
    with pytest.raises(KeyError, match="is one frame"):
        tcri.get.show(adata, "repertoire_summary", level="summary")


def test_a_group_column_named_replicate_prints(cohort, capsys):
    _model, adata = cohort
    adata = adata.copy()
    adata.obs["replicate"] = adata.obs["patient"]
    tcri.tl.mutual_information(adata, covariate=adata.obs["covariate"].cat.categories[0],
                               groupby="replicate", null_model=None, print=True)
    out = capsys.readouterr().out
    assert out.count("mutual_information ·") == 1 and "P0" in out


def test_a_change_prints_the_endpoints_it_tests_and_one_line_per_test(cohort, capsys):
    """With a reference: the adjusted endpoints, the paired test per split level, then the
    split contrast; without, the native endpoints."""
    _model, adata = cohort
    adata = adata.copy()
    a, b = list(adata.obs["covariate"].cat.categories)[:2]
    tcri.tl.phenotypic_entropy(adata, cov_from=a, cov_to=b, splitby="disease_status", print=True)
    out = " ".join(capsys.readouterr().out.split())        # a footer line wraps at 80 columns
    assert "adjusted_from" in out and "adjusted_to" in out and "value_from" not in out
    for arm in adata.obs["disease_status"].cat.categories:
        assert out.count(f"{arm}: {a} → {b}: Δ adjusted") == 1, out
    assert out.count("signed-rank p") == 2 and "(n 3 patient, paired)" in out
    assert "disease vs control: Δ adjusted" in out

    tcri.tl.phenotypic_entropy(adata, cov_from=a, cov_to=b, null_model=None, print=True)
    out = " ".join(capsys.readouterr().out.split())
    assert "value_from" in out and "value_to" in out and "adjusted" not in out
    assert out.count(f"{a} → {b}: Δ value") == 1


def test_flux_tests_the_adjusted_value_against_zero(cohort, capsys):
    _model, adata = cohort
    adata = adata.copy()
    a, b = list(adata.obs["covariate"].cat.categories)[:2]
    tcri.tl.phenotypic_flux(adata, cov_from=a, cov_to=b, print=True)
    out = " ".join(capsys.readouterr().out.split())
    assert out.count("adjusted vs 0: mean") == 1 and f"{a} → {b}: Δ" not in out
    tcri.tl.phenotypic_flux(adata, cov_from=a, cov_to=b, null_model=None, print=True)
    out = " ".join(capsys.readouterr().out.split())
    assert "value vs 0: not tested without a reference" in out


def test_a_wide_long_frame_prints_in_blocks_without_wrapping_a_value():
    import pandas as pd

    frame = pd.DataFrame({"group": [f"g{i}" for i in range(8)],
                          **{f"a_rather_long_column_name_{j}": [0.123456789 + i for i in range(8)]
                             for j in range(6)}})
    text = _output.plain_text(_output.frame_table(frame, title="t"), width=80)
    assert text.count("g7") >= 2                      # the label column repeats in each block
    assert all(len(line) <= 80 for line in text.splitlines())
    assert "7.123" in text and "\n.123" not in text  # values stay whole


def test_the_nulls_section_follows_each_fits_early_stopping():
    model, adata = _small_model()
    model.train(max_epochs=2, print=False, enable_progress_bar=False)
    model.to_anndata(adata)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tcri.null.phenotype(model, adata, n_nulls=1, early_stopping=False, key_added="nostop",
                            print=False, enable_progress_bar=False)
    line = [ln for ln in str(model).splitlines() if "null.phenotype.nostop" in ln][0]
    assert "reached max_epochs" not in line


def test_a_wide_frame_prints_whole_in_a_notebook_and_in_labeled_pieces_elsewhere():
    import pandas as pd

    frame = pd.DataFrame({"group": [f"g{i}" for i in range(8)],
                          **{f"a_rather_long_column_name_{j}": [0.5 + i for i in range(8)]
                             for j in range(6)}})
    table = _output.frame_table(frame, title="t")
    assert table.layout(2000, notebook=True) == ("table",)
    assert table.layout(80, notebook=False)[0] == "pieces"
    text = _output.plain_text(table, width=80)
    assert "columns 1–" in text and "with group repeated" in text
    one_row = _output.frame_table(frame.head(1), title="t")
    assert one_row.layout(2000, notebook=True) == ("transposed",)


# ── what a table says about the stored result ──────────────────────────────────────────────

def test_show_counts_the_fits_the_result_was_adjusted_against(cohort, capsys, monkeypatch):
    """The count comes from the stored draws, not from the nulls the object carries now."""
    _model, adata = cohort
    adata = adata.copy()
    tcri.tl.mutual_information(adata, covariate=adata.obs["covariate"].cat.categories[0],
                               print=True)
    assert "against null.phenotype (2 fits)" in capsys.readouterr().out
    monkeypatch.setattr(tcri._state.keys, "members",
                        lambda *_: ["null.phenotype[0]", "null.phenotype[1]", "null.phenotype[2]"])
    tcri.get.show(adata, "mutual_information")
    assert "against null.phenotype (2 fits)" in capsys.readouterr().out


def test_show_reads_a_raw_key_as_its_tool(cohort, capsys):
    _model, adata = cohort
    adata = adata.copy()
    a, b = list(adata.obs["covariate"].cat.categories)[:2]
    tcri.tl.phenotypic_flux(adata, cov_from=a, cov_to=b, print=False)
    tcri.get.show(adata, "tcri_phenotypic_flux")
    out = " ".join(capsys.readouterr().out.split())
    assert "phenotypic_flux · " in out and "adjusted vs 0: mean" in out and "paired" not in out
    tcri.pp.repertoire_summary(adata, clonotype_key="clone_id", groupby="patient", print=False)
    tcri.get.show(adata, "tcri_repertoire_summary")
    assert "repertoire_summary · patient" in capsys.readouterr().out


def test_show_says_why_a_result_has_no_contrast(cohort, capsys):
    _model, adata = cohort
    adata = adata.copy()
    adata.obs["one"] = pd.Categorical(["x"] * adata.n_obs)
    level = adata.obs["covariate"].cat.categories[0]
    tcri.tl.mutual_information(adata, covariate=level, splitby="one", print=False)
    tcri.get.show(adata, "mutual_information", level="stats")
    assert "fewer than two levels of 'one'" in capsys.readouterr().out
    tcri.tl.mutual_information(adata, covariate=level, print=False)
    tcri.get.show(adata, "mutual_information", level="stats")
    assert "splitby was not set" in capsys.readouterr().out


def test_gene_importance_prints_the_contrasts_of_its_top_genes(cohort, capsys):
    model, adata = cohort
    adata = adata.copy()
    genes = list(adata.var_names[:8])
    tcri.perturb.gene_importance(model, adata, genes=genes, splitby="disease_status",
                                 covariate=adata.obs["covariate"].cat.categories[0],
                                 enable_progress_bar=False, print=True)
    out = capsys.readouterr().out
    ranked = tcri.get.replicates(adata, "gene_importance").sort_values(
        "adjusted", ascending=False, kind="stable")
    top = list(dict.fromkeys(ranked["gene"]))[:_results.TOP_GENES]
    lines = [line for line in out.splitlines() if " vs " in line and ": Δ" in line]
    assert [line.split(":")[0] for line in lines] == top, lines
    assert "contrasts of 3 more genes: tcri.get.stats(adata, 'gene_importance')" in out


def test_a_cut_sharing_table_points_at_its_groups(capsys):
    adata = tcri.datasets.simulate_cohort(n_patients=24, n_cells_per_sample=20, seed=0)
    tcri.pp.clonotype_sharing(adata, clonotype_key="clone_id", groupby="patient", print=True)
    out = capsys.readouterr().out
    assert "4 more rows: tcri.get.result(adata, 'clonotype_sharing')['groups']" in out, out


def test_a_text_value_column_is_not_repeated_as_a_label():
    frame = pd.DataFrame({"group": [f"g{i}" for i in range(8)],
                          **{f"a_rather_long_column_name_{j}": [0.5 + i for i in range(8)]
                             for j in range(4)},
                          "adjusted 94% HDI": [f"{i}.1 – {i}.9" for i in range(8)]})
    text = _output.plain_text(_output.frame_table(frame, title="t"), width=80)
    assert "with group repeated" in text and "HDI repeated" not in text, text
    assert text.count("adjusted 94% HDI") == 1


def test_several_references_read_back_from_h5ad_print_as_names():
    import numpy as np

    names = np.array(["null.phenotype[0]", "null.clonotype[1]"], dtype=object)
    assert (_results._reference(names, 2)
            == "against null.phenotype[0], null.clonotype[1] (2 fits)")


def test_print_is_on_by_default_only_for_setup_train_and_the_null_fits():
    """Setup, a fit and the null fits report by default; the metrics, the summaries and the
    session functions print only when asked."""
    from tcri.model import TCRIModel

    def default(fn):
        return inspect.signature(fn).parameters["print"].default

    on = [TCRIModel.setup_anndata, TCRIModel.train, tcri.null.phenotype, tcri.null.clonotype,
          tcri.null.covariate, tcri.null.all]
    off = [getattr(tcri.tl, n) for n in tcri.tl.__all__]
    off += [tcri.perturb.gene_importance, tcri.pp.pool_rare_clones, tcri.pp.repertoire_summary,
            tcri.pp.clone_persistence, tcri.pp.clonotype_sharing, tcri.ut.save_tcri_session,
            tcri.ut.load_tcri_session]
    assert [f.__name__ for f in on if default(f) is not True] == []
    assert [f.__name__ for f in off if default(f) is not False] == []
