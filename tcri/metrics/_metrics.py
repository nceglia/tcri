import numpy
import pandas 
import collections
import tqdm

def calc_entropy(P, log_units = 2):
    P = P[P>0].flatten()
    return numpy.dot(P, -numpy.log(P))/numpy.log(log_units)

def clonotypic_entropy(adata, phenotype):
    clonotype_dist = collections.defaultdict(list)
    for ct, clonotype in zip(adata.obs[adata.uns["phenotype_column"]], adata.obs["IR_VDJ_1_junction_aa"]):
        if ct == phenotype:
            prob = 1.0
        else:
            prob = 0.0
        clonotype_dist[clonotype].append(prob)
    clonotype_expected_cells = dict()
    total_sum = 0
    for clone, ps in clonotype_dist.items():
        clonotype_expected_cells[clone] = sum(ps)
        total_sum += sum(ps)
    phenotype_entropy = []
    for clone, psum in clonotype_expected_cells.items():
        phenotype_entropy.append(psum/total_sum)
    return calc_entropy(numpy.array(phenotype_entropy))  / numpy.log2(total_sum)

def phenotypic_entropy(adata):
    prob_cols = []
    for x in adata.obs.columns:
        if "Pseudo-probability" in x:
            prob_cols.append(x)
    df = adata.obs
    clonotype_dist = collections.defaultdict(list)
    for c in set(df[adata.uns["clonotype_column"]]):
        dfx = df[df[adata.uns["clonotype_column"]]==c]
        X = dfx[prob_cols]
        X /= X.sum(axis=0)
        dist = X.to_numpy()[0]
        clonotype_dist[c] = dist
    clonotype_expected_cells = dict()
    total_sum = 0
    for clone, ps in clonotype_dist.items():
        clonotype_expected_cells[clone] = sum(ps)
        total_sum += sum(ps)
    phenotype_entropy = []
    for clone, psum in clonotype_expected_cells.items():
        phenotype_entropy.append(psum/total_sum)
    return calc_entropy(numpy.array(phenotype_entropy))  / numpy.log2(total_sum)


def phenotypic_flux(adata, from_this="Pre", to_this="Post"):
    precounts = collections.defaultdict(lambda : collections.defaultdict(int))
    postcounts = collections.defaultdict(lambda : collections.defaultdict(int))
    for clone in tqdm.tqdm(list(set(adata.obs[adata.uns["clonotype_column"]]))):
        if str(clone) == "nan": continue
        sub = adata[adata.obs[adata.uns["clonotype_column"]]==clone].copy()
        if len(sub.obs.index) < 5: continue
        pre = sub[sub.obs[adata.uns["condition_column"]] == from_this]
        post = sub[sub.obs[adata.uns["condition_column"]] == to_this]
        for ph_pre in set(pre.obs["genevector"]):
            for ph_post in set(post.obs["genevector"]):
                precount = len(pre[pre.obs["genevector"]==ph_pre].obs.index)
                postcount = len(post[post.obs["genevector"]==ph_post].obs.index)
                precounts[ph_pre][ph_post] += precount
                postcounts[ph_pre][ph_post] += postcount
    table = dict()
    table["Pre"] = []
    table["Post"] = []
    table["Phenotype A"] = []
    table["Phenotype B"] = []
    for ph_pre, posts in precounts.items():
        for ph_post, values in posts.items():
            table["Pre"].append(values)
            table["Post"].append(postcounts[ph_pre][ph_post])
            table["Phenotype A"].append(ph_pre)
            table["Phenotype B"].append(ph_post)
    table = pandas.DataFrame.from_dict(table)
    table["Pre"] = table["Pre"] / table["Pre"].sum()
    table["Post"] = table["Post"] / table["Post"].sum()
    return table