import collections
import operator
import seaborn as sns
import matplotlib.pyplot as plt

def tcr_umap(adata, reduction="umap", top_n=10, filename="tcr_plot.png", seq_column="IR_VDJ_1_junction_aa"):
    df = adata.obs
    dft = df[df[seq_column].notnull()]
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
            sizes.append(50)
            clonotype_labels.append(str(clonotype) + " {}".format(clonotype_counts[clonotype]))
    df["Size"] = sizes
    df["TCR Sequence"] = clonotype_labels
    colors = ["#fd971e","#66D9EF","#E6DB74","#7FFFD4","#AE81FF","#000000","#f92572","#A6E22E","#f47d7c","#0a012d","#3aaea3"]
    order = []
    for c in set(clonotype_labels):
        if c != "_Other":
            order.append(c)
    order.append("_Other")
    colors = colors[:len(set(clonotype_labels))-1] + ["#75715E"]
    x = [x[0] for x in adata.obsm["X_{}".format(reduction)]]
    y = [x[1] for x in adata.obsm["X_{}".format(reduction)]]
    customPalette = sns.set_palette(sns.color_palette(colors))
    sns.scatterplot(data=df,x=x, y=y, hue='TCR Sequence',hue_order=order, ax=ax1, alpha=0.7,linewidth=0.05,s=sizes,palette=customPalette)
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
    return [x for x in top_clonotypes if str(x)!="None"]
