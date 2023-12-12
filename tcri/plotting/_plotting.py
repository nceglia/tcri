import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statannot import add_stat_annotation
import matplotlib.patches as mpatches

from gseapy import dotplot
import tqdm

import collections
import operator
import itertools

from ..metrics._metrics import clonotypic_entropies as centropies
from ..metrics._metrics import clonality as clonality_tl
from ..metrics._metrics import flux as flux_tl
from ..metrics._metrics import mutual_information as mutual_information_tl
from ..metrics._metrics import clone_fraction as clone_fraction_tl
from ..utils._utils     import Phenotypes, CellRepertoire, Tcell, plot_pheno_sankey, plot_pheno_ternary_change_plots, draw_clone_bars
from ..preprocessing._preprocessing import clone_size, joint_distribution

#
import warnings
warnings.filterwarnings('ignore')

tcri_colors = [
    "#F28E7F",  # Salmon
    "#6A6E75",  # Iron Gray
    "#AE81FF",  # Purple
    "#FD971F",  # Orange
    "#E6DB74",  # Yellow
    "#A6E22E",  # Green
    "#66D9EF",  # Blue
    "#75715E",  # Brown
    "#F92659",  # Pink
    "#D65F0E",  # Abricos
    "#F92672",  # Red
    "#1E1E1E",   # Black
    "#004d47",  # Darker Teal
    "#272822",  # Background
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
    "#F3B61F",  # Goldenrod
    "#FFD8B1",  # Light Peach
    "#88AB75",  # Moss Green
    "#C38D94",  # Muted Rose
    "#6D6A75",  # Purple Gray
]

sns.set_palette(sns.color_palette(tcri_colors))

def phenotypic_flux(adata, splitby, order, clones=None, normalize=True, nt=True):
    phenotypes = Phenotypes(adata.obs[adata.uns["tcri_phenotype_key"]].unique())
    repertoires = dict()
    times = list(range(len(order)))
    for s in order:
        repertoires[s] = CellRepertoire(clones_and_phenos = {}, 
                                        phenotypes = phenotypes, 
                                        use_genes = False, 
                                        use_chain = False,
                                        seq_type = 'ntseq',
                                        chains_to_use = ['TRB'],
                                        name = s)
    for condition, seq, phenotype in zip(adata.obs[splitby],
                                         adata.obs[adata.uns["tcri_clone_key"]],
                                         adata.obs[adata.uns["tcri_phenotype_key"]]):
        if str(seq) != "nan" and condition in repertoires:
            if nt:
                t = Tcell(phenotypes = phenotypes, phenotypes_and_counts = {phenotype: 1}, 
                                                          TRB = dict(ntseq = seq), 
                                                          use_genes = False)
            else:
                t = Tcell(phenotypes = phenotypes, phenotypes_and_counts = {phenotype: 1}, 
                                                          TRB = dict(aaseq = seq), 
                                                          use_genes = False)
            repertoires[condition].cell_list.append(t)
    for condition, rep in repertoires.items():
        rep._set_consistency()
    fig, ax = plot_pheno_sankey(phenotypes = phenotypes, 
                                cell_repertoires = [repertoires[condition] for condition in order], 
                                clones = clones,
                                times = times,
                                xlim = [min(times), max(times)],
                                time_rescale = 1,
                                normalize=normalize,
                                xlabel = splitby,
                                return_axes = True, 
                                show_legend = True)
    ax.set_xticks(times)
    ax.set_xticklabels(order)
    fig.savefig("test.png")

def freq_to_size_scaling(freq):
    return 10*(freq**(1/2))

def freq_to_size_legend(ax, min_freq = 1e-6, max_freq = 1, loc = [0.85, 0.92], size = 0.25, x_offset = 0.1):
    freq_to_y_pos = lambda f: loc[1]-np.log(max_freq/f)*size/np.log(max_freq/min_freq)
    legend_pnts = np.exp(np.arange(np.log(min_freq), np.log(max_freq), 0.001))
    ax.scatter(loc[0]*np.ones(len(legend_pnts)), [freq_to_y_pos(y) for y in legend_pnts], s = [freq_to_size_scaling(x)**2 for x in legend_pnts], marker = '_', c = 'k')
    ax.plot(loc[0]*np.ones(2)+ x_offset, [freq_to_y_pos(min_freq), freq_to_y_pos(max_freq)], 'k', lw = 1)
    major_ticks = [10**x for x in  range(int(np.ceil(np.log10(min_freq))), int(np.floor(np.log10(max_freq))+1))]
    minor_ticks = sum([[x*major_tick for x in range(1, 10)] for major_tick in major_ticks[:-1]], [])
    for minor_tick in minor_ticks:
        ax.plot([loc[0]*np.ones(2)+ x_offset, loc[0]*np.ones(2)+ 1.2*x_offset], [freq_to_y_pos(minor_tick), freq_to_y_pos(minor_tick)], 'k', lw = 0.5)
    for major_tick in major_ticks:
        ax.plot([loc[0]*np.ones(2)+ x_offset, loc[0]*np.ones(2)+ 1.5*x_offset], [freq_to_y_pos(major_tick), freq_to_y_pos(major_tick)], 'k', lw = 0.75)
        if major_tick < 0.1:
            ax.text(loc[0]+ 1.7*x_offset, freq_to_y_pos(major_tick), '%.0e'%(major_tick), ha = 'left', va = 'center', rotation = 0, fontsize = 8)
        else:
            ax.text(loc[0]+ 1.7*x_offset, freq_to_y_pos(major_tick), str(major_tick), ha = 'left', va = 'center', rotation = 0, fontsize = 8)

def ternary_plot(adata, phenotype_names, splitby, condition, nt=False):
    phenotypes = Phenotypes(phenotype_names)
    repertoires = dict()

    for s in set(adata.obs[splitby]):
        repertoires[s] = CellRepertoire(clones_and_phenos = {}, 
                                        phenotypes = phenotypes, 
                                        use_genes = False, 
                                        use_chain = False,
                                        seq_type = 'ntseq',
                                        chains_to_use = ['TRB'],
                                        name = s)
    for condition, seq, phenotype in zip(adata.obs[splitby],
                                         adata.obs[adata.uns["tcri_clone_key"]],
                                         adata.obs[adata.uns["tcri_phenotype_key"]]):
        if str(seq) != "nan" and condition in repertoires:
            if nt:
                t = Tcell(phenotypes = phenotypes, phenotypes_and_counts = {phenotype: 1}, 
                                                          TRB = dict(ntseq = seq), 
                                                          use_genes = False)
            else:
                t = Tcell(phenotypes = phenotypes, phenotypes_and_counts = {phenotype: 1}, 
                                                          TRB = dict(aaseq = seq), 
                                                          use_genes = False)
            repertoires[condition].cell_list.append(t)
    for condition, rep in repertoires.items():
        rep._set_consistency()

    phenotype_names_dict = {p: p for p in phenotype_names}
    start_clones_and_phenos = repertoires[condition]
    # end_clones_and_phenos = repertoires[end_sample]
    s_dict = {c: freq_to_size_scaling(sum(start_clones_and_phenos[c].values())/start_clones_and_phenos.norm) for c in start_clones_and_phenos.clones}

    c_clones = sorted(start_clones_and_phenos.clones, key = s_dict.get, reverse = True)[:10]

    fig, ax = plot_pheno_ternary_change_plots(start_clones_and_phenotypes = start_clones_and_phenos,
                                            #end_clones_and_phenotypes = end_clones_and_phenos,
                                            phenotypes = phenotype_names, 
                                            phenotype_names = phenotype_names_dict,
                                            clones = c_clones,
                                            line_type = 'arrows', 
                                            s_dict = s_dict,
                                            return_axes  = True)
    freq_to_size_legend(ax)

def clone_umap(adata, reduction="umap", top_n=10, size=25, bg_size=0.1, figsize=(12,5)):
    df = adata.obs
    seq_column = adata.uns["tcri_clone_key"]
    plt.figure(figsize = figsize)
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
            sizes.append(bg_size)
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
    sns.scatterplot(x=x, y=y, hue=clonotype_labels,hue_order=order, ax=ax1, alpha=0.7,size=sizes, linewidth=0.0,palette=customPalette)
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


def tcri_boxplot(adata, function, groupby=None,ylabel="", splitby=None,figsize=(8,4),s=20):
    if groupby == None and splitby == None:
        data = function(adata)
        df = pd.DataFrame(list(data.items()), columns=['Phenotype', 'Clonotypic Entropy'])
        fig,ax=plt.subplots(1,1,figsize=figsize)
        sns.stripplot(data=df,x="Phenotype",y=ylabel,s=s,ax=ax, palette=tcri_colors)
        ax.set_ylim(0,max(df[ylabel] + 0.1))
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        fig.tight_layout()
    elif groupby != None and splitby == None:
        groups = adata.obs[groupby].unique()
        dfs = []
        for group in groups:
            data = function(adata[adata.obs[groupby]==group])
            df = pd.DataFrame(list(data.items()), columns=['Phenotype',ylabel])
            df[groupby] = group
            dfs.append(df)
        df = pd.concat(dfs)
        order = df.groupby(["Phenotype"]).median(ylabel).sort_values(ylabel).index.tolist()
        fig,ax=plt.subplots(1,1,figsize=figsize)
        sns.stripplot(data=df,x="Phenotype",y=ylabel,s=s,hue=groupby,ax=ax, palette=tcri_colors,order=order)
        sns.boxplot(data=df,x="Phenotype",y=ylabel,ax=ax, color="#999999",order=order)
        ax.set_ylim(0,max(df[ylabel] + 0.1))
        ax.set_title(ylabel)
        ax.set_ylabel(ylabel)
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        fig.tight_layout()
    elif groupby != None and splitby != None:
        groups = adata.obs[groupby].unique()
        dfs = []
        for group in groups:
            sub = adata[adata.obs[groupby]==group]
            splits = sub.obs[splitby].unique()
            for split in splits:
                data = function(sub[sub.obs[splitby]==split])
                df = pd.DataFrame(list(data.items()), columns=['Phenotype', ylabel])
                df[groupby] = group
                df[splitby] = split
                dfs.append(df)
        df = pd.concat(dfs)
        fig,ax=plt.subplots(1,1,figsize=figsize)
        order = df.groupby(["Phenotype"]).median(ylabel).sort_values(ylabel).index.tolist()
        sns.boxplot(data=df,x="Phenotype",y=ylabel,ax=ax, hue=splitby,palette=tcri_colors,order=order)
        ax.set_ylim(0,max(df[ylabel] + 0.1))
        ax.set_title(ylabel)
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        ax.set_ylabel(ylabel)
        fig.tight_layout()
    else:
        raise ValueError("'groupby' must be set to use 'splitby'.")
    return ax


def clonality(adata, groupby = None, splitby=None, s=10, figsize=(12,5)):
    return tcri_boxplot(adata,clonality_tl, ylabel="Clonality", groupby=groupby, splitby=splitby, s=s, figsize=figsize)

def clonotypic_entropies(adata, method="probabalistic", normalized=True, groupby=None, splitby=None, s=10, figsize=(12,5)):
    func = lambda x : centropies(x, normalized=normalized, method=method)
    return tcri_boxplot(adata, func, groupby=groupby, ylabel="Clonotypic Entropy", splitby=splitby, s=s, figsize=figsize)

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

def clone_size_umap(adata, reduction="umap",figsize=(10,8),scale=1,alpha=0.7,palette="coolwarm"):
    clone_size(adata)
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

def clone_fraction(adata, groupby):
    fractions = clone_fraction_tl(adata,groupby=groupby)
    draw_clone_bars(fractions, title=groupby)

def flux(adata,key, from_this, to_that, groupby, method="probabilistic", paint=None, distance_metric="l1", figsize=(12,5)):
    dfs = []
    if paint != None:
        palette = []
        legend_handles = [] 
        paint_categories = adata.obs[paint].unique()
        pcolors = dict(zip(paint_categories, tcri_colors))
        for category in paint_categories:
            handle = mpatches.Patch(color=pcolors[category], label=category)
            legend_handles.append(handle)
    else:
        palette = tcri_colors
    for x in tqdm.tqdm(list(set(adata.obs[groupby]))):
        sdata = adata[adata.obs[groupby]==x]
        l1_distances = flux_tl(sdata,key=key,from_this=from_this,to_that=to_that,distance_metric=distance_metric)
        df = pd.DataFrame(list(l1_distances.items()), columns=['Clone', distance_metric])
        df[groupby] = x
        if paint!=None:
            pcat = sdata.obs[paint].unique().tolist()[0]
            palette.append(pcolors[pcat])
        dfs.append(df)
    df = pd.concat(dfs)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    order = df.groupby(groupby).median(distance_metric).sort_values(distance_metric).index.tolist()
    fig,ax=plt.subplots(1,1,figsize=figsize)
    sns.boxplot(data=df,x=groupby,y=distance_metric,order=order,palette=palette,ax=ax)
    # If paint is not None, add the legend to the plot
    if paint != None:
        ax.legend(handles=legend_handles, title=paint)
    fig.tight_layout()
    return ax

def mutual_information(adata, groupby, splitby=None, method="probabilistic", figsize=(6,5)):
    mis = []
    groups = []
    splits = []
    for group in set(adata.obs[groupby]):
        gdata = adata[adata.obs[groupby] == group]
        if splitby != None:
            for split in set(gdata.obs[splitby]):
                sdata = gdata[gdata.obs[splitby] == split]
                mi = mutual_information_tl(sdata, method=method)
                mis.append(mi)
                groups.append(group)
                splits.append(split)
        else:
            joint_distribution(gdata, )
            mi = mutual_information_tl(gdata,method=method)
            mis.append(mi)
            groups.append(group)
    df = pd.DataFrame.from_dict({"MI":mis, groupby: groups})
    if splitby != None:
        df[splitby] = splits
    fig, ax = plt.subplots(1,1,figsize=figsize)
    sns.boxplot(data=df,x=splitby,y="MI",ax=ax)
    fig.tight_layout()
    return ax