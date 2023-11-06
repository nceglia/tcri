import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statannot import add_stat_annotation
from gseapy import dotplot

import collections
import operator
import itertools

from ..metrics._metrics import clonotypic_entropy as centropy
from ..metrics._metrics import transcriptional_entropy as pentropy
from ..metrics._metrics import mutual_information as mutual_info
from ..metrics._metrics import rank_clones_by_transcriptional_entropy, rank_genes_by_clonotypic_entropy, flux_l1, flux_dkl
from ..metrics._metrics import marker_enrichment as menrich
from ..preprocessing._preprocessing import transcriptional_joint_distribution, phenotypic_joint_distribution
from ..preprocessing._preprocessing import clone_size as cs

import warnings
warnings.filterwarnings('ignore')

tcri_colors = [
    "#272822",  # Background
    "#F92672",  # Red
    "#FD971F",  # Orange
    "#E6DB74",  # Yellow
    "#A6E22E",  # Green
    "#66D9EF",  # Blue
    "#AE81FF",  # Purple
    "#75715E",  # Brown
    "#F92659",  # Pink
    "#D65F0E",  # Abricos
    "#1E1E1E",   # Black
    "#004d47",  # Darker Teal
    "#D291BC",  # Soft Pink
    "#3A506B",  # Dark Slate Blue
    "#5D8A5E",  # Sage Green
    "#A6A1E2",  # Dull Lavender
    "#E97451",  # Burnt Sienna
    "#6C8D67",  # Muted Lime Green
    "#832232",  # Dim Maroon
    "#669999",  # Desaturated Cyan
    "#C08497",  # Dusty Rose
    "#587B7F",  # Ocean Blue
    "#9A8C98",  # Muted Purple
    "#F28E7F",  # Salmon
    "#F3B61F",  # Goldenrod
    "#6A6E75",  # Iron Gray
    "#FFD8B1",  # Light Peach
    "#88AB75",  # Moss Green
    "#C38D94",  # Muted Rose
    "#6D6A75",  # Purple Gray
]

sns.set_palette(sns.color_palette(tcri_colors))

def tcr_umap(adata, reduction="umap", top_n=10, size=25):
    df = adata.obs
    seq_column = adata.uns["tcri_clone_key"]
    plt.figure(figsize = (10, 8))
    clonotype_counts = collections.defaultdict(int)
    for clonotype in df[seq_column]:
        clonotype_counts[clonotype] += 1
    top_clonotypes = sorted(clonotype_counts.items(), key=operator.itemgetter(1),reverse=True)
    top_clonotypes = [x[0] for x in top_clonotypes[:top_n]]
    ax1 = plt.subplot(1,1,1)
    sizes = []
    clonotype_labels = []
    for clonotype in df[seq_column]:
        clonotype = str(clonotype)
        if clonotype not in top_clonotypes or clonotype == "None" or clonotype == "nan":
            sizes.append(2)
            clonotype_labels.append("_Other")
        else:
            sizes.append(size)
            clonotype_labels.append(str(clonotype) + " {}".format(clonotype_counts[clonotype]))
    df["Size"] = sizes
    df["TCR Sequence"] = clonotype_labels
    colors = tcri_colors
    order = []
    for c in set(clonotype_labels):
        if c != "_Other":
            order.append(c)
    order.append("_Other")
    colors = colors[:len(set(clonotype_labels))-1] + ["#75715E"]
    x = [x[0] for x in adata.obsm["X_{}".format(reduction)]]
    y = [x[1] for x in adata.obsm["X_{}".format(reduction)]]
    customPalette = sns.set_palette(sns.color_palette(colors))
    sns.scatterplot(data=df,x=x, y=y, hue='TCR Sequence',hue_order=order, ax=ax1, alpha=0.7,linewidth=0.05,size=sizes,palette=customPalette)
    ax1.set_xlabel('UMAP-1')
    ax1.set_ylabel('UMAP-2')
    ax1.xaxis.set_ticklabels([])
    ax1.yaxis.set_ticklabels([])
    ax1.xaxis.set_ticks([])
    ax1.yaxis.set_ticks([])
    ax1.set_title("Top 10 TCR Clone by Size")
    h,l = ax1.get_legend_handles_labels()
    ax1.legend(h[:top_n-1], l[:top_n-1], borderaxespad=2.,fontsize='9',bbox_to_anchor=(0, 1), loc='best')
    plt.tight_layout()

def transcriptional_entropy(adata, phenotype_key, groupby, splitby=None, genes=None, figsize=(12,4)):
    if splitby == None:
        adata.obs["splitby"] = groupby
        splitby = "splitby"
    fig,ax = plt.subplots(1,1,figsize=figsize)
    ents = []
    group = []
    phs = []
    hue = []
    for g in set(adata.obs[groupby]):
        sdata = adata[adata.obs[groupby]==g]
        for ph in set(adata.obs[phenotype_key]):
            pdata = sdata[sdata.obs[phenotype_key] == ph]
            for s in set(pdata.obs[splitby]):
                zdata = pdata[pdata.obs[splitby] == s]
                joint_distribution(zdata)
                ent = pentropy(zdata,genes=genes)
                ents.append(ent)
                phs.append(ph)
                group.append(g)
                hue.append(s)
    df = pd.DataFrame.from_dict({"Phenotype":phs,"Entropy":ents,"Group":group, splitby:hue})
    ph_order = df.groupby("Phenotype").median().sort_values("Entropy").index.tolist()
    if splitby == "splitby":
        hue = None
        hue_order = None
    else:
        hue_order = list(set(hue))
    sns.boxplot(data=df,x="Phenotype",y="Entropy",order=ph_order, hue=hue,hue_order=hue_order)
    if hue_order == None:
        sns.swarmplot(data=df,x="Phenotype",y="Entropy",order=ph_order,hue=hue,hue_order=hue_order)
    else:
        pairs = []
        combos = list(itertools.combinations(list(set(hue)),2))
        for ph in list(set(phs)):
            for c in combos:
                pairs.append(((ph, c[0]),(ph,c[1])))
        ax, test_results = add_stat_annotation(ax, 
                                               data=df,
                                               x="Phenotype",
                                               y="Entropy", 
                                               order=ph_order,
                                               hue=hue,
                                               hue_order=hue_order, 
                                               box_pairs=pairs, 
                                               test='Mann-Whitney', 
                                               text_format='star', 
                                               loc='outside', 
                                               verbose=2)
        fig.legend(loc='upper left', bbox_to_anchor=(1.03, 1))
    return ax

def marker_enrichment_score(adata,markers):
    pre_res = menrich(adata,markers)
    terms = pre_res.res2d.Term
    return pre_res.plot(terms, show_ranking=False, legend_kws={'loc': (1.05, 0)})

def marker_enrichment_dotplot(adata,markers,figsize=(12,7)):
    pre_res = menrich(adata,markers)
    ax = dotplot(pre_res.res2d,
                column="FDR q-val",
                title='Marker Enrichment by Lower Entropy Genes',
                cmap=plt.cm.viridis,
                size=4, # adjust dot size
                figsize=figsize, cutoff=1, show_ring=True)
    return ax

def rank_markers(adata, markers, groupby, group_order=None, figsize=(12,4)):
    grp = []
    rnk = []
    phs = []
    for group in set(adata.obs[groupby]):
        sdata = adata[adata.obs[groupby]==group]
        joint_distribution(sdata)
        df = rank_genes_by_clonotypic_entropy(sdata)
        sorted_ce = df["Gene"].tolist()
        for ph, genes in markers.items():
            for g in genes:
                if g not in sorted_ce: continue
                rnk.append(sorted_ce.index(g))
                phs.append(ph)
                grp.append(group)
    df = pd.DataFrame.from_dict({"Phenotype":phs,"Rank":rnk,"Group":grp})
    order = df.groupby("Phenotype").median().sort_values("Rank").index.tolist()
    if group_order == None:
        group_order=list(set(grp))
    _,ax= plt.subplots(1,1,figsize=figsize)
    sns.boxplot(data=df,x="Phenotype",y="Rank",hue="Group",order=order, hue_order=group_order,ax=ax)
    pairs = []
    combos = list(itertools.combinations(list(set(grp)),2))
    for ph in list(set(phs)):
        for c in combos:
            pairs.append(((ph, c[0]),(ph,c[1])))
    ax, test_results = add_stat_annotation(ax, 
                                            data=df,
                                            x="Phenotype",
                                            y="Rank", 
                                            order=order,
                                            hue="Group",
                                            hue_order=group_order,
                                            box_pairs=pairs, 
                                            test='Mann-Whitney', 
                                            text_format='star', 
                                            loc='outside', 
                                            verbose=2)
    return ax

def clonotypic_entropy(adata, groupby, splitby,  clones=None, figsize=(12,4)):
    tcr_key = adata.uns["tcri_clone_key"]
    fig,ax = plt.subplots(1,1,figsize=figsize)
    ents = []
    group = []
    phs = []
    hue = []
    for g in set(adata.obs[groupby]):
        pdata = adata[adata.obs[groupby]==g]
        for s in set(pdata.obs[splitby]):
            zdata = pdata[pdata.obs[splitby] == s]
            joint_distribution(zdata)
            ent = centropy(zdata,clones=clones)
            ents.append(ent)
            group.append(g)
            hue.append(s)
    df = pd.DataFrame.from_dict({"Entropy":ents,"Group":group, splitby:hue})
    order = df.groupby(splitby).median().sort_values("Entropy").index.tolist()
    hue_order = list(set(hue))
    sns.boxplot(data=df,x=splitby,y="Entropy",order=order)
    sns.swarmplot(data=df,x=splitby,y="Entropy",order=order)
    pairs = list(itertools.combinations(list(set(hue)),2))
    ax, test_results = add_stat_annotation(ax, 
                                            data=df,
                                            x=splitby,
                                            y="Entropy", 
                                            order=order,
                                            box_pairs=pairs, 
                                            test='Mann-Whitney', 
                                            text_format='star', 
                                            loc='outside', 
                                            verbose=2)
    fig.legend(loc='upper left', bbox_to_anchor=(1.03, 1))
    return ax

def flux(adata, key, from_this, to_that, groupby, splitby, order=None, method="L1", figsize=(12,3)):
    assert method == "L1" or method == "dkl", "Method must be 'L1' or 'dkl'."
    if method == "L1":
        flux_fxn = flux_l1
    else:
        flux_fxn = flux_dkl
    flux = []
    label = []
    group = []
    for s in set(adata.obs[splitby]):
        sdata = adata[adata.obs[splitby] == s]
        for p in set(sdata.obs['patient']):
            resp = sdata[sdata.obs['patient'] == p]
            val = flux_fxn(resp,key=key,from_this=from_this,to_that=to_that)
            flux.append(val)
            label.append(s)
            group.append(p)
    df = pd.DataFrame.from_dict({"Label":label,"Flux {}".format(method):flux, "Group":group})
    if order == None:
        order = df.groupby("Label").median().sort_values("Flux {}".format(method)).index.tolist()
    pairs = list(itertools.combinations(order,2))
    fig, ax = plt.subplots(1,1,figsize=figsize)
    sns.boxplot(data=df,x="Label",y='Flux {}'.format(method),order=order)
    sns.swarmplot(data=df, x="Label", y="Flux {}".format(method),order=order,color="#999999",s=10)
    ax, test_results = add_stat_annotation(ax, 
                                            data=df,
                                            x="Label",
                                            y="Flux {}".format(method), 
                                            order=order,
                                            box_pairs=pairs,
                                            test='t-test_ind', 
                                            text_format='star', 
                                            loc='outside', 
                                            verbose=2)
    return ax

def clone_size(adata, reduction="umap",figsize=(10,8),scale=1,alpha=0.7,palette="coolwarm"):
    cs(adata)
    df = adata.obs
    reduction="umap"
    scale_factor=3.
    sizes = np.log10(adata.obs["clone_size"].to_numpy())
    df["UMAP1"] = [x[0] for x in adata.obsm["X_{}".format(reduction)]]
    df["UMAP2"] = [x[1] for x in adata.obsm["X_{}".format(reduction)]]
    df["log(Clone Size)"] = sizes
    fig,ax=plt.subplots(1,1,figsize=(10,8))
    sns.scatterplot(data=df,x="UMAP1", y="UMAP2", hue="log(Clone Size)",palette=palette, ax=ax, alpha=alpha,linewidth=0.)
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    fig.tight_layout()
    return ax

def mutual_information(adata, groupby, splitby, order=None, figsize=(2,4)):
    mis = []
    label = []
    group = []
    for s in set(adata.obs[splitby]):
        sdata = adata[adata.obs[splitby] == s]
        for p in set(sdata.obs['patient']):
            xdata = sdata[sdata.obs['patient'] == p]
            joint_distribution(xdata)
            mi = mutual_info(xdata)
            mis.append(mi)
            label.append(s)
            group.append(p)
    df = pd.DataFrame.from_dict({"Label":label,"MI": mis, "Group":group})
    if order == None:
        order = df.groupby("Label").median().sort_values("MI").index.tolist()
    pairs = list(itertools.combinations(order,2))
    fig, ax = plt.subplots(1,1,figsize=figsize)
    sns.boxplot(data=df, x="Label", y='MI', order=order)
    sns.swarmplot(data=df, x="Label", y="MI",color="#999999",s=10,order=order)
    ax, test_results = add_stat_annotation(ax, 
                                            data=df,
                                            x="Label",
                                            y="MI", 
                                            order=order,
                                            box_pairs=pairs,
                                            test='t-test_ind', 
                                            text_format='star', 
                                            loc='outside', 
                                            verbose=2)
    return ax
