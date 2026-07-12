#!/usr/bin/env python3
"""Build the TCRI dependency map for the *target* API.

Two graphs over the same nodes:
  (1) call graph      — function -> function (who calls whom)
  (2) dataflow graph  — function -> adata-state key -> function (producers/consumers)

Renders: producer/consumer + call-adjacency tables (Markdown + HTML), a Mermaid
DAG for each graph (renders on GitHub / in the HTML via CDN), and a Graphviz DOT
combining both (render offline: `dot -Tsvg tcri_dependency_map.dot -o dep.svg`).

This is curated from the target contract (build_tcri_contract.py), so it stays a
*contract* we refactor toward — not an after-the-fact extraction. See the
"How this is built / standard tooling" section for the auto-extraction path.
"""
import html as _html
import os
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_MD = os.path.join(_HERE, "tcri_dependency_map.md")
OUT_HTML = os.path.join(_HERE, "tcri_dependency_map.html")
OUT_DOT = os.path.join(_HERE, "tcri_dependency_map.dot")

NS_COLOR = {
    "ml": "#dc2626", "pp": "#0f766e", "tl": "#7c3aed",
    "pl": "#2563eb", "ut": "#475569", "core": "#b45309", "state": "#334155",
}
NS_FILL = {
    "ml": "#fde2e2", "pp": "#d6f0ec", "tl": "#ece3fb",
    "pl": "#dbe8fd", "ut": "#e5e9ef", "core": "#fef3c7", "state": "#ffffff",
}

# ---- function nodes (target API + shared prims) -----------------------------
FUNCS = [
    # ml
    ("ml", "setup_anndata"), ("ml", "train"), ("ml", "get_latent_representation"),
    ("ml", "get_cell_phenotype_probs"), ("ml", "get_p_ct"), ("ml", "boost_phenotype_prior"),
    # pp
    ("pp", "register_model"), ("pp", "joint_distribution"), ("pp", "group_singletons"),
    ("pp", "clone_size"), ("pp", "filter_genes"),
    # tl
    ("tl", "clonotypic_entropy"), ("tl", "phenotypic_entropy"), ("tl", "mutual_information"),
    ("tl", "clonality"), ("tl", "flux"), ("tl", "delta_clonotypic_entropy"),
    ("tl", "mi_compare"), ("tl", "delta_entropy_table"), ("tl", "flux_table"),
    # pl
    ("pl", "mutual_information"), ("pl", "clonotypic_entropy"), ("pl", "phenotypic_entropy"),
    ("pl", "clonality"), ("pl", "flux"), ("pl", "mi_compare"),
    ("pl", "bayesian_mutual_information"), ("pl", "ridge_delta_entropy"),
    ("pl", "phenotypic_flux"), ("pl", "polar_plot"), ("pl", "clone_size_umap"),
    ("pl", "top_clone_umap"), ("pl", "phenotype_probabilities_umap"),
    ("pl", "model_loss"), ("pl", "archetypes"), ("pl", "model_pgm"),
    # ut
    ("ut", "save_tcri_session"), ("ut", "load_tcri_session"), ("ut", "write_adata_safely"),
    # core (shared primitives that participate in call edges)
    ("core", "_joint_to_mi"), ("core", "_stats.distance"), ("core", "_stats.auc_perm"),
    ("core", "_stats.bootstrap_auc"), ("core", "_group_table"), ("core", "_metric_boxplot"),
    ("core", "_build_sankey"),
]
NS = {f"{ns}.{nm}": ns for ns, nm in FUNCS}

# ---- call edges: caller -> callee ------------------------------------------
CALLS = [
    # pl -> tl / pp / core
    ("pl.mutual_information", "tl.mutual_information"),
    ("pl.clonotypic_entropy", "tl.clonotypic_entropy"),
    ("pl.phenotypic_entropy", "tl.phenotypic_entropy"),
    ("pl.clonality", "tl.clonality"),
    ("pl.clonality", "core._metric_boxplot"),
    ("pl.flux", "tl.flux"),
    ("pl.mi_compare", "tl.mi_compare"),
    ("pl.mi_compare", "core._stats.auc_perm"),
    ("pl.mi_compare", "core._stats.bootstrap_auc"),
    ("pl.bayesian_mutual_information", "tl.mutual_information"),
    ("pl.ridge_delta_entropy", "tl.delta_entropy_table"),
    ("pl.phenotypic_flux", "pp.joint_distribution"),
    ("pl.phenotypic_flux", "core._build_sankey"),
    ("pl.polar_plot", "pp.joint_distribution"),
    ("pl.polar_plot", "tl.clonotypic_entropy"),
    ("pl.clone_size_umap", "pp.clone_size"),
    ("pl.model_loss", "ml.train"),
    ("pl.archetypes", "ml.train"),
    # tl -> pp / tl / core
    ("tl.clonotypic_entropy", "pp.joint_distribution"),
    ("tl.phenotypic_entropy", "pp.joint_distribution"),
    ("tl.mutual_information", "pp.joint_distribution"),
    ("tl.mutual_information", "core._joint_to_mi"),
    ("tl.flux", "pp.joint_distribution"),
    ("tl.flux", "core._stats.distance"),
    ("tl.delta_clonotypic_entropy", "tl.clonotypic_entropy"),
    ("tl.mi_compare", "tl.mutual_information"),
    ("tl.mi_compare", "core._group_table"),
    ("tl.delta_entropy_table", "tl.delta_clonotypic_entropy"),
    ("tl.delta_entropy_table", "core._group_table"),
    ("tl.flux_table", "tl.flux"),
    ("tl.flux_table", "core._group_table"),
    # pp -> ml
    ("pp.register_model", "ml.get_cell_phenotype_probs"),
    ("pp.register_model", "ml.get_latent_representation"),
    ("pp.register_model", "ml.get_p_ct"),
    # ml internal
    ("ml.get_cell_phenotype_probs", "ml.get_p_ct"),
    # ut
    ("ut.save_tcri_session", "ut.write_adata_safely"),
    ("ut.load_tcri_session", "ml.setup_anndata"),
]

# ---- dataflow: adata-state keys with producer + concrete consumers ----------
# (consumers listed are DIRECT readers; metrics that read only via
#  pp.joint_distribution are attributed to joint_distribution, not re-listed)
STATE = [
    ("uns[tcri_metadata]", "pp.register_model",
     ["pp.joint_distribution", "tl.clonality", "tl.mi_compare",
      "tl.delta_entropy_table", "tl.flux_table", "pp.clone_size"]),
    ("uns[tcri_p_ct]", "pp.register_model", ["pp.joint_distribution"]),
    ("uns[tcri_local_scale]", "pp.register_model", ["pp.joint_distribution"]),
    ("uns[tcri_*_categories]", "pp.register_model",
     ["pp.joint_distribution", "tl.clonotypic_entropy", "tl.phenotypic_entropy"]),
    ("uns[tcri_ct_to_cov/ct_to_c]", "pp.register_model", ["pp.joint_distribution"]),
    ("uns[tcri_{ct,cov}_array_for_cells]", "pp.register_model", ["pp.joint_distribution"]),
    ("obsm[X_tcri_logits]", "pp.register_model", ["pp.joint_distribution"]),
    ("obsm[X_tcri_probabilities]", "pp.register_model", ["pl.phenotype_probabilities_umap"]),
    ("obsm[X_tcri]", "pp.register_model", ["pl.clone_size_umap", "pl.top_clone_umap"]),
    ("obs[tcri_phenotype]", "pp.register_model", ["tl.clonality", "pl.top_clone_umap"]),
    ("obs[clone_size]", "pp.clone_size", ["pl.clone_size_umap"]),
    ("obs[trb_unique]", "pp.group_singletons", ["ml.setup_anndata"]),
    ("pyro_param_store", "ml.train",
     ["ml.get_p_ct", "ut.save_tcri_session", "ut.load_tcri_session"]),
]

# =============================================================================
# derive adjacency
# =============================================================================
calls_out = defaultdict(list)
calls_in = defaultdict(list)
for a, b in CALLS:
    calls_out[a].append(b)
    calls_in[b].append(a)

ALL = [f"{ns}.{nm}" for ns, nm in FUNCS]
entry_points = [f for f in ALL if f not in calls_in and NS.get(f) in ("pl", "ut", "pp")]
leaves = [f for f in ALL if f not in calls_out and NS.get(f) in ("ml", "core", "pp")]


def sid(s):
    out = []
    for ch in s:
        out.append(ch if ch.isalnum() else "_")
    return "n_" + "".join(out)


# ---- Mermaid ----------------------------------------------------------------
def mermaid_calls():
    L = ["flowchart TB"]
    for ns in ("pl", "tl", "pp", "ml", "core", "ut"):
        members = [f for f in ALL if NS[f] == ns and (f in calls_out or f in calls_in)]
        if not members:
            continue
        L.append(f'  subgraph {ns.upper()}')
        for f in members:
            L.append(f'    {sid(f)}["{f}"]')
        L.append("  end")
    for a, b in CALLS:
        L.append(f"  {sid(a)} --> {sid(b)}")
    for ns, color in NS_FILL.items():
        L.append(f"  classDef {ns} fill:{color},stroke:{NS_COLOR[ns]},color:#111;")
    for f in ALL:
        if f in calls_out or f in calls_in:
            L.append(f"  class {sid(f)} {NS[f]};")
    return "\n".join(L)


def mermaid_dataflow():
    L = ["flowchart LR"]
    seen = set()
    for key, prod, cons in STATE:
        kid = sid("KEY_" + key)
        L.append(f'  {kid}[("{key}")]')
        L.append(f'  class {kid} state;')
        if prod not in seen:
            L.append(f'  {sid(prod)}["{prod}"]'); L.append(f'  class {sid(prod)} {NS[prod]};'); seen.add(prod)
        L.append(f"  {sid(prod)} ==> {kid}")
        for c in cons:
            if c not in seen:
                L.append(f'  {sid(c)}["{c}"]'); L.append(f'  class {sid(c)} {NS[c]};'); seen.add(c)
            L.append(f"  {kid} -.-> {sid(c)}")
    for ns, color in NS_FILL.items():
        L.append(f"  classDef {ns} fill:{color},stroke:{NS_COLOR[ns]},color:#111;")
    return "\n".join(L)


# ---- Graphviz DOT (combined) ------------------------------------------------
def dot():
    L = ['digraph tcri {', '  rankdir=TB; node [fontname="Helvetica",fontsize=10];',
         '  edge [fontname="Helvetica",fontsize=8];']
    for ns in ("pl", "tl", "pp", "ml", "ut", "core"):
        L.append(f'  subgraph cluster_{ns} {{ label="{ns}"; style=rounded; color="{NS_COLOR[ns]}";')
        for f in ALL:
            if NS[f] == ns:
                L.append(f'    "{f}" [shape=box,style=filled,fillcolor="{NS_FILL[ns]}",'
                         f'color="{NS_COLOR[ns]}"];')
        L.append("  }")
    L.append("  // call edges (solid)")
    for a, b in CALLS:
        L.append(f'  "{a}" -> "{b}" [color="#475569"];')
    L.append("  // state keys + dataflow (write=bold, read=dashed)")
    for key, prod, cons in STATE:
        L.append(f'  "{key}" [shape=cylinder,style=filled,fillcolor="#f8fafc",color="#334155"];')
        L.append(f'  "{prod}" -> "{key}" [color="#0f766e",penwidth=1.6];')
        for c in cons:
            L.append(f'  "{key}" -> "{c}" [style=dashed,color="#94a3b8"];')
    L.append("}")
    return "\n".join(L)


# =============================================================================
# render markdown + html
# =============================================================================
STANDARD = [
    ("Call graph (who-calls-whom)",
     "The de-facto standard is a directed graph rendered with <b>Graphviz/DOT</b>. "
     "Auto-extract from source with the stdlib <code>ast</code> module, or tools like "
     "<code>pyan3</code>, <code>code2flow</code>, <code>pydeps</code> (module-level), or "
     "<code>griffe</code> (the engine behind mkdocstrings). For Markdown-native rendering use "
     "<b>Mermaid</b> <code>flowchart</code> (GitHub renders it inline); for interactive web use "
     "<code>cytoscape.js</code> or <code>d3</code>."),
    ("Dataflow / producer-consumer",
     "Because TCRI couples through <code>adata.uns/obsm/obs</code> keys (not just direct calls), "
     "the precise model is a <b>bipartite graph</b> of functions ↔ state keys. This is exactly a "
     "<b>build-system DAG</b>: state keys are the artifacts/targets, functions are the rules. The "
     "standard tools for that shape are <b>Make</b>, <b>Snakemake</b>, or <b>dbt</b> "
     "(<code>dbt docs</code> renders an interactive lineage graph) — and the same idea is what "
     "scverse calls a <i>data-flow</i>/provenance graph. Here it is curated by hand from the "
     "contract so it can lead the refactor rather than trail it."),
    ("Preventing drift after the refactor",
     "Once the target lands, point an <code>ast</code> walker at <code>tcri/</code> to extract the "
     "<i>actual</i> call edges + <code>uns/obsm/obs</code> read/writes and diff them against this "
     "curated graph in CI. Drift = a function that calls or reads/writes something the contract "
     "doesn't list."),
]


def md():
    L = ["# TCRI — Dependency Map (target API)",
         "",
         "_Call graph + `adata`-state producer/consumer links for the post-refactor API. "
         "Curated from `build_tcri_contract.py`; regenerate with `build_tcri_depgraph.py`._",
         "",
         "**Two graphs, same nodes.** (1) **Call graph** — `pl` → `tl` → `pp` → `ml`, bottoming "
         "out at the model + shared `core` primitives. (2) **Dataflow** — everything funnels "
         "through the `tcri_*` state that `pp.register_model` writes; `pp.joint_distribution` is "
         "the universal consumer hub.",
         "",
         "## Call graph", "",
         "```mermaid", mermaid_calls(), "```", "",
         "### Call adjacency", "",
         "| function | calls | called by |", "|---|---|---|"]
    for f in ALL:
        if f in calls_out or f in calls_in:
            out = ", ".join(f"`{x}`" for x in calls_out.get(f, [])) or "—"
            inn = ", ".join(f"`{x}`" for x in calls_in.get(f, [])) or "—"
            L.append(f"| `{f}` | {out} | {inn} |")
    L += ["", f"**Entry points** (no caller): " + ", ".join(f"`{f}`" for f in entry_points),
          "", f"**Leaves** (call nothing in-package): " + ", ".join(f"`{f}`" for f in leaves),
          "",
          "## Dataflow (producers / consumers)", "",
          "```mermaid", mermaid_dataflow(), "```", "",
          "### State producer/consumer table", "",
          "| adata key | produced by | consumed by |", "|---|---|---|"]
    for key, prod, cons in STATE:
        L.append(f"| `{key}` | `{prod}` | " + ", ".join(f"`{c}`" for c in cons) + " |")
    L += ["",
          "> Metrics that read only `uns[tcri_metadata]` + categories **via** "
          "`pp.joint_distribution` are attributed to that hub, not re-listed per key.",
          "",
          "## How this is built / standard tooling", ""]
    for title, body in STANDARD:
        b = body.replace("<b>", "**").replace("</b>", "**").replace("<i>", "_").replace("</i>", "_")
        b = b.replace("<code>", "`").replace("</code>", "`")
        L.append(f"**{title}.** {b}")
        L.append("")
    L += ["## Graphviz", "",
          "A combined DOT file is emitted to `tcri_dependency_map.dot` "
          "(call edges solid, writes bold-teal, reads dashed). Render:",
          "", "```bash", "dot -Tsvg tcri_dependency_map.dot -o tcri_dependency_map.svg", "```"]
    return "\n".join(L)


def esc(s):
    return _html.escape(str(s))


def render_html():
    css = """
    body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#fafafa;color:#1a1a1a}
    .wrap{max-width:1180px;margin:0 auto;padding:28px}
    h1{font-size:26px;margin:0 0 4px} h2{font-size:20px;margin:34px 0 12px;border-bottom:2px solid #eee;padding-bottom:6px}
    h3{font-size:15px;margin:18px 0 8px}
    .sub{color:#555;margin:0 0 14px}
    .conv{background:#f1f5f9;border-left:4px solid #64748b;padding:10px 14px;border-radius:6px;margin:14px 0}
    table{border-collapse:collapse;width:100%;font-size:12.5px;margin:8px 0 20px}
    th,td{border:1px solid #e5e7eb;padding:6px 9px;text-align:left;vertical-align:top}
    th{background:#f3f4f6;position:sticky;top:0}
    code{background:#eef2ff;padding:1px 5px;border-radius:4px;font-size:92%}
    .mermaid{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:14px;margin:10px 0;overflow-x:auto}
    .note{background:#fffbeb;border-left:4px solid #d97706;padding:8px 12px;border-radius:5px;margin:10px 0}
    .legend span{display:inline-block;margin:2px 8px 2px 0;padding:3px 10px;border-radius:12px;color:#fff;font-weight:600;font-size:12px}
    """
    legend = "".join(
        f'<span style="background:{NS_COLOR[ns]}">{ns}</span>'
        for ns in ("ml", "pp", "tl", "pl", "ut", "core"))
    H = ['<!doctype html><meta charset="utf-8"><title>TCRI dependency map</title>',
         f'<style>{css}</style>',
         '<script type="module">import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";'
         'mermaid.initialize({startOnLoad:true, flowchart:{useMaxWidth:true}});</script>',
         '<div class="wrap">']
    H.append("<h1>TCRI — Dependency Map (target API)</h1>")
    H.append('<p class="sub">Call graph + <code>adata</code>-state producer/consumer links · '
             'curated from the target contract.</p>')
    H.append(f'<div class="legend">{legend}</div>')
    H.append('<div class="conv"><b>Two graphs, same nodes.</b> (1) <b>Call graph</b>: '
             '<code>pl → tl → pp → ml</code>, bottoming out at the model + shared <code>core</code> '
             'primitives. (2) <b>Dataflow</b>: everything funnels through the <code>tcri_*</code> '
             'state that <code>pp.register_model</code> writes; <code>pp.joint_distribution</code> is '
             'the universal consumer hub.</div>')

    H.append("<h2>Call graph</h2>")
    H.append(f'<div class="mermaid">{esc(mermaid_calls())}</div>')
    H.append("<h3>Call adjacency</h3>")
    H.append("<table><tr><th>function</th><th>calls</th><th>called by</th></tr>")
    for f in ALL:
        if f in calls_out or f in calls_in:
            out = ", ".join(f"<code>{esc(x)}</code>" for x in calls_out.get(f, [])) or "—"
            inn = ", ".join(f"<code>{esc(x)}</code>" for x in calls_in.get(f, [])) or "—"
            color = NS_COLOR[NS[f]]
            H.append(f'<tr><td style="border-left:4px solid {color}"><code>{esc(f)}</code></td>'
                     f"<td>{out}</td><td>{inn}</td></tr>")
    H.append("</table>")
    H.append('<div class="note"><b>Entry points</b> (no caller): '
             + ", ".join(f"<code>{esc(f)}</code>" for f in entry_points)
             + "<br><b>Leaves</b> (call nothing in-package): "
             + ", ".join(f"<code>{esc(f)}</code>" for f in leaves) + "</div>")

    H.append("<h2>Dataflow (producers / consumers)</h2>")
    H.append(f'<div class="mermaid">{esc(mermaid_dataflow())}</div>')
    H.append("<table><tr><th>adata key</th><th>produced by</th><th>consumed by</th></tr>")
    for key, prod, cons in STATE:
        H.append(f"<tr><td><code>{esc(key)}</code></td><td><code>{esc(prod)}</code></td>"
                 f"<td>" + ", ".join(f"<code>{esc(c)}</code>" for c in cons) + "</td></tr>")
    H.append("</table>")
    H.append('<div class="note">Metrics that read only <code>uns[tcri_metadata]</code> + categories '
             '<b>via</b> <code>pp.joint_distribution</code> are attributed to that hub, not re-listed '
             'per key.</div>')

    H.append("<h2>How this is built / standard tooling</h2>")
    for title, body in STANDARD:
        H.append(f"<p><b>{esc(title)}.</b> {body}</p>")
    H.append("<h2>Graphviz</h2>")
    H.append('<p>Combined DOT emitted to <code>tcri_dependency_map.dot</code> '
             '(call edges solid, writes bold-teal, reads dashed). Render: '
             '<code>dot -Tsvg tcri_dependency_map.dot -o tcri_dependency_map.svg</code>.</p>')
    H.append("</div>")
    return "".join(H)


with open(OUT_MD, "w") as f:
    f.write(md())
with open(OUT_HTML, "w") as f:
    f.write(render_html())
with open(OUT_DOT, "w") as f:
    f.write(dot())
print("wrote", OUT_MD)
print("wrote", OUT_HTML)
print("wrote", OUT_DOT)
print(f"nodes: {len(ALL)} | call edges: {len(CALLS)} | state keys: {len(STATE)} | "
      f"entry points: {len(entry_points)} | leaves: {len(leaves)}")
