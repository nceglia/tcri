import numpy
import pandas 
import matplotlib.pyplot as plt
import seaborn as sns
from ..metrics._metrics import phenotypic_entropy as centropy

def clonotypic_entropy(adata, title=""):
    entropies = []
    phenotypes = []
    conditions = []
    samples = []
    for c in set(adata.obs[adata.uns["condition_column"]]): 
        sub = adata[adata.obs[adata.uns["condition_column"]]==c]
        for s in set(sub.obs[sub.uns["sample_column"]]):
            cdata = sub[sub.obs[sub.uns["sample_column"]]==s]
            for ph in set(cdata.obs[sub.uns["phenotype_column"]]):
                ent = entropy(cdata, ph)
                entropies.append(ent)
                phenotypes.append(ph)
                conditions.append(c)
                samples.append(s)
    df = pandas.DataFrame.from_dict({"Sample":samples,
                                     "Condition":conditions,
                                     "Phenotype":phenotypes,
                                     "Normalized Entropy":entropies})
    df = df.sort_values("Normalized Entropy")
    fig, ax = plt.subplots(1,1,figsize=(10,5))
    sns.boxplot(data=df,x="Phenotype",y="Normalized Entropy",hue="Condition",ax=ax)
    plt.tight_layout()
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)