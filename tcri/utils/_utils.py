# -*- coding: utf-8 -*-
"""
Copyright (C) 2022 Zachary Sethna
"""
from __future__ import print_function, division
import os
import sys
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import mpltern
import numpy as np
from scipy.stats import fisher_exact, binom_test
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors


def probabilities(adata):
    matrix = adata.obs[adata.uns["probability_columns"]]
    barcodes = matrix.index.tolist()
    cells = np.nan_to_num(matrix.to_numpy())
    index = adata.uns["joint_distribution"].index
    probabs = dict()
    for bc, cell in zip(barcodes, cells):
        probabs[bc] = dict(zip(index, cell))
    return probabs

class Error(Exception):
    pass

tcri_colors = [
    "#272822",  # Background
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

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2022 Zachary Sethna
"""

def gene_to_num_str(gene_name, gene_type):
    """Strips excess gene name info to number string.

    Parameters
    ----------
    gene_name : str
        Gene or allele name
    gene_type : char
        Genomic cassette type. (i.e. V, D, or J)

    Returns
    -------
    num_str : str
        Reduced gene or allele name with leading zeros and excess
        characters removed.

    """

    num_str = gene_name.lower().split(gene_type.lower())[-1]
    num_str = '-'.join([g.lstrip('0') for g in num_str.split('-')])
    num_str = '*'.join([g.lstrip('0') for g in num_str.split('*')])

    return gene_type.lower() + num_str

def nt2aa(ntseq):
    """Translate a nucleotide sequence into an amino acid sequence.

    Parameters
    ----------
    ntseq : str
        Nucleotide sequence composed of A, C, G, or T (uppercase or lowercase)

    Returns
    -------
    aaseq : str
        Amino acid sequence

    Example
    --------
    >>> nt2aa('TGTGCCTGGAGTGTAGCTCCGGACAGGGGTGGCTACACCTTC')
    'CAWSVAPDRGGYTF'

    """
    nt2num = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}
    aa_dict ='KQE*TPASRRG*ILVLNHDYTPASSRGCILVFKQE*TPASRRGWMLVLNHDYTPASSRGCILVF'

    return ''.join([aa_dict[nt2num[ntseq[i]] + 4*nt2num[ntseq[i+1]] + 16*nt2num[ntseq[i+2]]] for i in range(0, len(ntseq), 3) if i+2 < len(ntseq)])

def is_ntseq(seq):
    return all([c in 'ACGT' for c in seq.upper()])

def is_aaseq(seq):
    return all([c in 'ACDEFGHIKLMNPQRSTVWY*' for c in seq.upper()])


class ClonesAndData(object):
    """Base class for data associated with clones.

    This class will behave very similarly to defaultdict from collections. The
    standard dictionary methods (items, keys, values, get) function
    identically, as do the magic methods of __len__, __setitem__, __iter__,
    __delitem__, and __eq__. Accessing items, __getitem__, behaves similarly to
    that of a default dictionary in that it returns the class default_value if
    the key isn't in the class, however unlike a defaultdict it will not add
    the key and default_value to the class.

    Class Attributes
    ----------------
    default_value : None
        Default value returned if a clone not in the class is accessed.

    Attributes
    ----------
    clones_and_data : dict
        Dictionary keyed by clones, values are the data associated with the
        clone.

    Methods
    -------
    clones()
        Returns list of clones in the class (very similar to self.keys())
    clone_intersection(clones)
        Returns the intersection of self.clones() and clones
    clone_union(clones)
        Returns the union of self.clones() and clones

    """

    default_value = None

    def __init__(self, clones_and_data = {}):
        self.clones_and_data = clones_and_data.copy()

    def __len__(self):
        return len(self.clones_and_data)

    def __getitem__(self, clone):
        return self.clones_and_data.get(clone, self.default_value)

    def __setitem__(self, clone, datum):
        self.clones_and_data[clone] = datum

    def __delitem__(self, clone):
        del self.clones_and_data[clone]

    def __iter__(self):
        return iter(self.clones_and_data)

    def __eq__(self, clones_and_data):
        if type(clones_and_data) != type(self): return False
        return self.clones_and_data == clones_and_data.clones_and_data

    def items(self):
        return self.clones_and_data.items()

    def keys(self):
        return self.clones_and_data.keys()

    def values(self):
        return self.clones_and_data.values()

    def get(self, clone, default = default_value):
        return self.clones_and_data.get(clone, default)

    def clones(self):
        return list(self.clones_and_data.keys())

    def clone_intersection(self, clones):
        try:
            clones = set(clones.keys())
        except AttributeError:
            clones = set(clones)

        return list(clones.intersection(self.clones_and_data))

    def clone_union(self, clones):
        try:
            clones = set(clones.keys())
        except AttributeError:
            clones = set(clones)

        return list(clones.union(self.clones_and_data))
    
    def __repr__(self):
        head_str = ', '.join(['%s: %s'%(clone, datum) for clone, datum in list(self.clones_and_data.items())[:10]])
        if len(self.clones_and_data) > 10:
            head_str += ', ...'
        return "ClonesAndData(%s)"%('Number of clones: %s, Data: {%s}'%(len(self.clones_and_data), head_str))

class ClonesAndCounts(ClonesAndData):
    """Class for cell/read counts associated with a clone.

    Inherits from ClonesAndData and thus will behave similarly to a defaultdict
    with default_value of 0. However, it will remove all clones with counts<=0
    and has support added for the magic method __add__ which can add two
    ClonesAndCount classes.

    Class Attributes
    ----------------
    default_value : 0
        Default value returned if a clone not in the class is accessed.

    Attributes
    ----------
    clones_and_data : dict
        Dictionary keyed by clones, values are counts for each clone.
    norm : int or float
        Normalization of the counts. Generally this is sum of the counts of all
        clones in the class, however it can be set independently if the
        normalization needs to be different. This value will be automatically
        updated if items are added/deleted or if the counts are updated for
        a given clone.

    Methods
    -------
    get_frequency(clone, add_pseudo_count = False)
        Returns the frequency of the clone in a repertoire defined by the
        clones/counts in the class (self[clone]/self.norm). If pseudocounts are
        added, it will change the default count from 0 to 1/3. (So a the
        frequency returned from a clone not in the class will be
        1/(3*self.norm)).
    set_norm(norm = None)
        Sets self.norm. If no norm is provided it will set the norm to the sum
        of the counts over all clones (i.e. self.norm = sum(self.values()))

    """


    default_value = 0

    def __init__(self, clones_and_counts = {}):
        ClonesAndData.__init__(self)
        self.clones_and_data = clones_and_counts.copy()
        self.set_norm()

    def __setitem__(self, clone, count):
        if count > 0:
            self.norm += count - self.__getitem__(clone)
            self.clones_and_data[clone] = count
        else:
            self.__delitem__(clone)

    def __delitem__(self, clone):
        self.norm -= self.__getitem__(clone)
        del self.clones_and_data[clone]

    def __add__(self, clones_and_counts):
        tot_clones_and_counts = ClonesAndCounts(self.clones_and_data)
        for clone, count in clones_and_counts.clones_and_data.items():
            tot_clones_and_counts[clone] += count
        return tot_clones_and_counts

    def get(self, clone, default = default_value):
        return self.clones_and_data.get(clone, default)

    def set_norm(self, norm = None):
        """Set normalization constant.

        Sets the attribute norm.

        Parameters
        ----------
        norm : float or None (default is None)
            Either sets the norm to the provided value or to the sum of the
            counts of all clones in the class (i.e. sum(self.values())).
        """
        if norm is None:
            self.norm = sum(self.clones_and_data.values())
        else:
            self.norm = norm

    def get_frequency(self, clone, add_pseudo_count = False):
        """Get a clone's frequency.

        Parameters
        ----------
        clone : str
            Clone whose frequency is to be returned
        add_pseudo_count : bool (default is False)
            If True, adds a pseudocount of 1/3 when computing the frequency of
            a clone not in the class.

        Returns
        -------
        freq : float
            Frequency of the clone in the class.

        """
        return self.get(clone, add_pseudo_count/3)/self.norm
    
    def __repr__(self):
        head_str = ', '.join(['%s: %s'%(clone, datum) for clone, datum in list(self.clones_and_data.items())[:10]])
        if len(self.clones_and_data) > 10:
            head_str += ', ...'
        return "ClonesAndCounts(%s)"%('Number of clones: %s, Norm: %s, Data: {%s}'%(len(self.clones_and_data), self.norm, head_str))


class ClonesAndPvals(ClonesAndData):
    """Class for pvalues associated with a clone.

    Inherits from ClonesAndData and thus will behave similarly to a defaultdict
    with default_value of 1. Most functions will act only on the clones that 
    pass the significance threshold (e.g. len, iter, etc). 
    Non-significant clone pvals are preserved in self.clone_and_data and will
    still be returned if queried by get or __getitem__.

    Class Attributes
    ----------------
    default_value : 1
        Default value returned if a clone not in the class is accessed.

    Attributes
    ----------
    clones_and_data : dict
        Dictionary keyed by clones, values are pvalues for each clone.
    pval_thresh : float
        Significance threshold
    significant_clones_and_pvals : dict
        Dictionary keyed by significant clones, values are pvalues for each
        clone.

    Methods
    -------
    get_significant_clones(pval_thresh)
        Returns a ClonesAndPvals object with only the clones/pvals for clones
        that are more significant than the pval_thresh provided
        (so self[clone] < pval_thresh)

    """

    default_value = 1

    def __init__(self, clones_and_pvals = {}, pval_thresh = 1e-2):
        self.clones_and_data = clones_and_pvals.copy()
        self.pval_thresh = pval_thresh
        
        self.significant_clones_and_pvals = {}
        for clone, pval in self.clones_and_data.items():
            if pval<self.pval_thresh:
                self.significant_clones_and_pvals[clone] = pval
                
    def __len__(self):
        return len(self.significant_clones_and_pvals)
    
    def __setitem__(self, clone, pval):
        self.clones_and_data[clone] = pval
        if pval < self.pval_thresh:
            self.significant_clones_and_pvals[clone] = pval
        else:
            try:
                del self.significant_clones_and_pvals[clone]
            except KeyError:
                pass
        
    def __delitem__(self, clone):
        del self.clones_and_data[clone]
        try:
            del self.significant_clones_and_pvals[clone]
        except KeyError:
            pass
        
    def __iter__(self):
        return iter(self.significant_clones_and_pvals)

    def items(self):
        return self.significant_clones_and_pvals.items()

    def keys(self):
        return self.significant_clones_and_pvals.keys()

    def values(self):
        return self.significant_clones_and_pvals.values()

    def get(self, clone, default = default_value):
        return self.clones_and_data.get(clone, default)

    def get_significant_clones(self, pval_thresh = None):
        """Determines which clones are significant.

        Parameters
        ----------
        pval_thresh : float
            Pvalue threshold to determine the significance of clones. Default
            will be self.pval_thresh

        Returns
        -------
        sig_clones_and_pvals : ClonesAndPvals
            ClonesAndPval object with the clones/pvals for clones that pass
            the significance test based on pval_thresh.

        """
        if pval_thresh is None:
            pval_thresh = self.pval_thresh
        sig_clones_and_pvals = ClonesAndPvals(pval_thresh=pval_thresh)

        for clone, pval in self.clones_and_data.items():
            if pval<pval_thresh:
                sig_clones_and_pvals[clone] = pval

        return sig_clones_and_pvals
    
    
    def clones(self):
        return list(self.significant_clones_and_pvals.keys())

    def clone_intersection(self, clones):
        try:
            clones = set(clones.keys())
        except AttributeError:
            clones = set(clones)

        return list(clones.intersection(self.significant_clones_and_pvals))

    def clone_union(self, clones):
        try:
            clones = set(clones.keys())
        except AttributeError:
            clones = set(clones)

        return list(clones.union(self.significant_clones_and_pvals))

    def __repr__(self):
        head_str = ', '.join(['%s: %s'%(clone, datum) for clone, datum in list(self.significant_clones_and_pvals.items())[:10]])
        if len(self.significant_clones_and_pvals) > 10:
            head_str += ', ...'
        return "ClonesAndPvals(%s)"%('Number of significant clones: %s, Pvalue thresh: %.1e, Significant clones: {%s}'%(len(self.significant_clones_and_pvals), self.pval_thresh, head_str))

class TCRClonePvalue(object):
    """Class for computing pvalues for a TCR clone.

    Attributes
    ----------

    Methods
    -------
    compute_fisher_pvalues(specific_clones_and_counts, baseline_clones_and_counts, foldchange_thresh = 1, **kwargs)
        Compute pvalues for fisher exact test.
    compute_binomial_pvalues(specific_clones_and_counts, baseline_clones_and_counts, frac_thresh = 0.5, **kwargs)
        Compute pvalues for binomial test.
    """
    def __init__(self):
        pass
    
    def __repr__(self):
        return "TCRClonePvalue"

    def compute_fisher_pvalues(self, specific_clones_and_counts, baseline_clones_and_counts, foldchange_thresh = 1, multiple_hypothesis_correction = 'bonferroni', **kwargs):
        """Compute pvalues for fisher test.

        Parameters
        ----------
        specific_clones_and_counts : ClonesAndCounts
            ClonesAndCounts object for the specific repertoire.
        baseline_clones_and_counts : ClonesAndCounts
            ClonesAndCounts object for the baseline/comparison repertoire.
        foldchange_thresh : float (default is 1)
            The fisher test will be computed with regard to this foldchange.
            For foldchange_thresh == 1, this will be the standard fisher exact
            test. For foldchange_thresh != 1, we implement this by changing the
            normalization value of the baseline/comparison repertoire to
            effectively change the frequencies of the clones in that repertoire.

        Optional Parameters (**kwargs)
        ------------------------------
        pval_cutoff : float (default is 1)
            Significance cutoff for the clones included in the returned
            clones_and_pvals.
        x_norm : int or float
            Manual normalization for baseline_clones_and_counts. Will still
            be adjusted by foldchange_thresh.
        y_norm : int or float
            Manual normalization for specific_clones_and_counts.


        Returns
        -------
        clones_and_pvals : ClonesAndPvals
            ClonesAndPval object with multiple hypothesis adjusted pvals for
            clones based off of the fisher test.

        """


        y_clones_and_counts = specific_clones_and_counts
        x_clones_and_counts = baseline_clones_and_counts

        pval_cutoff = kwargs.get('pval_cutoff', 1)
        
        if 'x_norm' in kwargs:
            x_norm = int(kwargs['x_norm']/foldchange_thresh)
        else:
            x_norm = int(x_clones_and_counts.norm/foldchange_thresh)
        if 'y_norm' in kwargs:
            y_norm = int(kwargs['y_norm'])
        else:
            y_norm = int(y_clones_and_counts.norm)

        clones = kwargs.get('clones', y_clones_and_counts.clone_union(x_clones_and_counts))

        c_count_combo_pval_dict = {}

        c_kwargs = {kw: kw_val for kw, kw_val in kwargs.items() if kw == 'pval_thresh'}
        clones_and_pvals = ClonesAndPvals(**c_kwargs)
        for clone in clones:
            x = x_clones_and_counts[clone]
            y = y_clones_and_counts[clone]

            if x/x_norm > y/y_norm:
                c_count_combo_pval_dict[(x, y)] = 1
            if (x,y) not in c_count_combo_pval_dict:
                if multiple_hypothesis_correction == 'bonferroni':
                    c_count_combo_pval_dict[(x, y)] = np.clip(fisher_exact(np.array([[x, x_norm - x], [y, y_norm - y]]))[1]*len(clones), 0, 1)
                elif multiple_hypothesis_correction is None or multiple_hypothesis_correction == 'None':
                    c_count_combo_pval_dict[(x, y)] = np.clip(fisher_exact(np.array([[x, x_norm - x], [y, y_norm - y]]))[1], 0, 1)

            if c_count_combo_pval_dict[(x, y)] <= pval_cutoff:
                clones_and_pvals[clone] = c_count_combo_pval_dict[(x, y)]

        return clones_and_pvals

    def compute_binomial_pvalues(self, specific_clones_and_counts, baseline_clones_and_counts, frac_thresh = 0.5, multiple_hypothesis_correction = 'bonferroni', **kwargs):
        """Compute pvalues for binomial test.

        Parameters
        ----------
        specific_clones_and_counts : ClonesAndCounts
            ClonesAndCounts object for the specific repertoire.
        baseline_clones_and_counts : ClonesAndCounts
            ClonesAndCounts object for the baseline/comparison repertoire.
        frac_thresh : float (default is 0.5)
            The binomial test will be computed with regard to this relative
            fraction.

        Optional Parameters (**kwargs)
        ------------------------------
        pval_cutoff : float (default is 1)
            Significance cutoff for the clones included in the returned
            clones_and_pvals.

        Returns
        -------
        clones_and_pvals : ClonesAndPvals
            ClonesAndPval object with multiple hypothesis adjusted pvals for
            clones based off of the binomial test.

        """

        y_clones_and_counts = specific_clones_and_counts
        x_clones_and_counts = baseline_clones_and_counts

        pval_cutoff = kwargs.get('pval_cutoff', 1)

        clones = kwargs.get('clones', y_clones_and_counts.clone_union(x_clones_and_counts))

        c_count_combo_pval_dict = {}

        c_kwargs = {kw: kw_val for kw, kw_val in kwargs.items() if kw == 'pval_thresh'}
        clones_and_pvals = ClonesAndPvals(**c_kwargs)

        for clone in clones:
            x = x_clones_and_counts[clone]
            y = y_clones_and_counts[clone]

            if y/(x+y) < frac_thresh:
                c_count_combo_pval_dict[(x, y)] = 1
            if (x,y) not in c_count_combo_pval_dict:
                if multiple_hypothesis_correction == 'bonferroni':
                    c_count_combo_pval_dict[(x, y)] = np.clip(binom_test(np.array([y, x]), alternative = 'greater', p = frac_thresh)*len(clones), 0, 1)
                elif multiple_hypothesis_correction is None or multiple_hypothesis_correction == 'None':
                    c_count_combo_pval_dict[(x, y)] = np.clip(binom_test(np.array([y, x]), alternative = 'greater', p = frac_thresh), 0, 1)
            clones_and_pvals[clone] = c_count_combo_pval_dict[(x, y)]

            if c_count_combo_pval_dict[(x, y)] <= pval_cutoff:
                clones_and_pvals[clone] = c_count_combo_pval_dict[(x, y)]

        return clones_and_pvals



class SequenceDefinition(object):
    """Class for TCR sequence type definition

    Determines what level of specificity of sequence definition will be used.
    Also includes magic methods for the operators =, <=, >=. This will indicate
    if a definition is

    Attributes/Keyword Arguments
    ----------------------------
    seq_type : 'ntseq' or 'aaseq'
        Defines whether the CDR3 sequence is defined on the nucleotide sequence
        or amino acid sequence level. (Default: 'ntseq')
    use_genes : bool
        If True the cassette gene information will be included. (Default: True)
    use_alleles : bool
        If True the allele information will be included in the sequence.
        (Default: False)
    use_chain : bool
        If True will specify whether it is a TRB or TRA chain. (Default: False)

    Methods
    -------
    get_seq_def()
        Returns dictionary version of sequence definition (useful for **kwargs).
    """

    def __init__(self, **kwargs):

        self.seq_type = kwargs.get('seq_type', 'ntseq')
        self.use_genes = kwargs.get('use_genes', True)
        self.use_alleles = kwargs.get('use_alleles', False)
        self.use_chain = kwargs.get('use_chain', False)

        
    def get_seq_def(self):
        """Returns the sequence definition dictionary
        
        Returns
        -------
        dict
            Sequence definition dictionary

        """
        return dict(seq_type = self.seq_type,
                    use_genes = self.use_genes,
                    use_alleles = self.use_alleles,
                    use_chain = self.use_chain)
    
    def __repr__(self):
        return "SequenceDefinition(%s)"%(', '.join([kw + '=' + str(kw_val) for kw, kw_val in self.get_seq_def().items()]))
        
        
    def __eq__(self, seq_def):
        if type(seq_def) == SequenceDefinition:
            seq_def = seq_def.get_seq_def()
        return self.get_seq_def() == seq_def

    def __le__(self, seq_def):
        if type(seq_def) == SequenceDefinition:
            seq_def = seq_def.get_seq_def()

        if self.seq_type == 'ntseq' and seq_def['seq_type'] == 'aaseq':
            return False

        return not any([self.__dict__[cond] for cond in ['use_genes', 'use_alleles', 'use_chain'] if not seq_def[cond]])

    def __ge__(self, seq_def):
        if type(seq_def) == SequenceDefinition:
            seq_def = seq_def.get_seq_def()

        if self.seq_type == 'aaseq' and seq_def['seq_type'] == 'ntseq':
            return False

        return all([self.__dict__[cond] for cond in ['use_genes', 'use_alleles', 'use_chain'] if seq_def[cond]])

class CloneDefinition(SequenceDefinition):
    """Class for T cell clone type definition

    Inherits from SequenceDefinition and adds the specification of what
    combination of TRB and TRA chains to use.

    Attributes/Keyword Arguments
    ----------------------------
    seq_type : 'ntseq' or 'aaseq'
        Defines whether the CDR3 sequence is defined on the nucleotide sequence
        or amino acid sequence level. (Default: 'ntseq')
    use_genes : bool
        If True the cassette gene information will be included. (Default: True)
    use_alleles : bool
        If True the allele information will be included in the sequence.
        (Default: False)
    use_chain : bool
        If True will specify whether it is a TRB or TRA chain. (Default: False)
    chains : list
        Specifies which chains (of TRB and TRA) are to be included.
        (Default: ['TRB', 'TRA'])

    Methods
    -------
    get_clone_def()
        Returns dictionary version of clone definition (useful for **kwargs).
    """

    def __init__(self, **kwargs):

        SequenceDefinition.__init__(self)

        self.chains = ['TRB', 'TRA']

        self.use_chain = True

        for kw, kw_val in kwargs.items():
            if kw in self.__dict__: self.__dict__[kw] = kw_val

    def get_clone_def(self):
        """Returns the clone definition dictionary
        
        Returns
        -------
        dict
            Clone definition dictionary
            
        """
        return dict(seq_type = self.seq_type,
                    use_genes = self.use_genes,
                    use_alleles = self.use_alleles,
                    use_chain = self.use_chain,
                    chains = self.chains)

    def __repr__(self):
        return "CloneDefinition(%s)"%(', '.join([kw + '=' + str(kw_val) for kw, kw_val in self.get_clone_def().items()]))

    def __eq__(self, clone_def):

        if type(clone_def) == CloneDefinition:
            clone_def = clone_def.get_clone_def()
        return self.get_clone_def() == clone_def

    def __le__(self, clone_def):
        if type(clone_def) == CloneDefinition:
            clone_def = clone_def.get_clone_def()

        if not set(self.chains).issubset(clone_def['chains']):
            return False

        if self.seq_type == 'ntseq' and clone_def['seq_type'] == 'aaseq':
            return False

        return not any([self.__dict__[cond] for cond in ['use_genes', 'use_alleles', 'use_chain'] if not clone_def[cond]])


    def __ge__(self, clone_def):
        if type(clone_def) == CloneDefinition:
            clone_def = clone_def.get_clone_def()

        if not set(clone_def['chains']).issubset(self.chains):
            return False

        if self.seq_type == 'aaseq' and clone_def['seq_type'] == 'ntseq':
            return False

        return all([self.__dict__[cond] for cond in ['use_genes', 'use_alleles', 'use_chain'] if clone_def[cond]])

class TCRseq(SequenceDefinition):
    """Class for TCR sequences (single chain)

    The class stores all provided sequence information for a TCR sequence
    (single chain -- only the TRB or TRA) and will return a standardized syntax
    string based on the SequenceDefinition that can be used as a hash for the
    sequence. Inherits all of the attributes of SequenceDefinition.
    
    Attributes
    ----------
    ntseq : str
        Nucleotide sequence of TCR CDR3 sequence (including conserved C and F)
    aaseq : str
        Amino acid sequence of TCR CDR3 sequence (including conserved C and F)
    chain : 'TRB' or 'TRA'
        Specifies what chain the sequence corresponds to
    v_genes : list
        List of V genes associated with the sequence
    j_genes : list
        List of J genes associated with the sequence
    v_alleles : list
        List of V alleles associated with the sequence
    j_alleles : list
        List of J alleles associated with the sequence
    
    Methods
    -------
    seq_rep(**kwargs)
        Returns standardized sequence string based on the sequence definition
        (modified by any **kwargs)
    set_chain(chain)
        Sets self.chain
    add_gene(gene)
        Adds genes/alleles to gene/allele lists
    """

    def __init__(self, ntseq = None, aaseq = None, seq_rep = None, **kwargs):


        SequenceDefinition.__init__(self, **kwargs)

        self.ntseq = ntseq
        if ntseq is not None:
            self.aaseq = nt2aa(ntseq)
            if aaseq is not None and aaseq != self.aaseq:
                raise SequenceMismatch(self.aaseq, aaseq)
        else:
            self.aaseq = aaseq
            if self.aaseq is not None:
                self.seq_type = 'aaseq'

        self.chain = None
        self.v_genes = []
        self.j_genes = []
        self.v_alleles = []
        self.j_alleles = []

        for kw, kw_val in kwargs.items():
            if kw.lower() == 'chain':
                self.set_chain(kw_val.upper())
            elif kw.lower()[0] in 'vdj':
                self.add_gene(kw_val, gene_type = kw.lower())

        if seq_rep is not None:
            self.load_from_seq_rep(seq_rep, **kwargs)
        
    def set_chain(self, chain):
        """Sets chain of the sequence

        Parameters
        ----------
        chain : 'TRB' or 'TRA'
            Chain of sequence

        Raises
        ------
        ChainMismatch
            Error if chain is already specified and different than input

        Attributes modified
        -------------------
        self.chain

        """

        if self.chain is None:
            self.chain = chain
        elif chain != self.chain:
            raise ChainMismatch(self.chain, chain)

    def add_gene(self, gene_name, gene_type = '', reset = False):
        """Method to add genes/alleles to sequence info.
        
        This function will also infer/check the chain of the sequence based on
        provided gene names.
        
        Parameters
        ----------
        gene_name : str
            Gene/allele name
        gene_type : 'V' or 'J', optional
            Specified type of gene. The default is ''.
        reset : bool, optional
            Resets the gene list to only include provided gene. 
            The default is False.

        Raises
        ------
        UnknownGene
        ChainMismatch

        Attributes Modified
        -------------------
        self.chain
        self.v_genes
        self.v_alleles
        self.j_genes
        self.j_alleles

        """

        if 'A' in gene_name.upper():
            self.set_chain('TRA')
        elif 'B' in gene_name.upper():
            self.set_chain('TRB')

        if gene_type.lower() == 'v' or 'v' in gene_name.lower():
            if reset:
                self.v_genes = [gene_to_num_str(gene_name.lower(), 'v').split('*')[0]]
                self.v_alleles = [gene_to_num_str(gene_name.lower(), 'v')]
            else:
                self.v_genes.append(gene_to_num_str(gene_name.lower(), 'v').split('*')[0])
                self.v_alleles.append(gene_to_num_str(gene_name.lower(), 'v'))
        elif gene_type.lower() == 'j' or 'j' in gene_name.lower():
            if reset:
                self.j_genes = [gene_to_num_str(gene_name.lower(), 'j').split('*')[0]]
                self.j_alleles = [gene_to_num_str(gene_name.lower(), 'j')]
            else:
                self.j_genes.append(gene_to_num_str(gene_name.lower(), 'j').split('*')[0])
                self.j_alleles.append(gene_to_num_str(gene_name.lower(), 'j'))
        else:
            raise UnknownGene(gene_name, gene_type)
    
    def load_from_seq_rep(self, seq_rep, **kwargs):
        """Load Sequence info from seq_rep representation

        Parameters
        ----------
        seq_rep : str
            String representation for sequence information
        **kwargs : keywords
            Input for SequenceDefinition

        """
        split_seq_rep = seq_rep.split(',')
        if split_seq_rep[0] in ['TRA', 'TRB']:
            self.set_chain(split_seq_rep[0])
            split_seq_rep = split_seq_rep[1:]
            self.use_chain = True
        if is_ntseq(split_seq_rep[0]):
            self.ntseq = split_seq_rep[0]
            self.aaseq = nt2aa(self.ntseq)
        elif is_aaseq(split_seq_rep[0]):
            self.ntseq = None
            self.aaseq = split_seq_rep[0]
            self.seq_type = 'aaseq'
        if len(split_seq_rep[1:]) == 0:
            self.use_alleles = False
            self.use_genes = False
        else:
            for g in split_seq_rep[1:]:
                if '*' in g: self.use_alleles = True
                self.add_gene(g)
        
        for kw, kw_val in kwargs.items():
            if kw in ['seq_type', 'use_genes', 'use_alleles', 'use_chain']:
                self.__dict__[kw] = kw_val      
    
    def seq_rep(self, **kwargs):
        """Returns standardized sequence string based on sequence definition.

        Returns
        -------
        str
            Sequence string

        """

        seq_def = self.get_seq_def()
        for kw, kw_val in kwargs.items():
            if kw in seq_def: seq_def[kw] = kw_val


        if seq_def['use_chain'] and self.chain is not None:
            s = [self.chain]
        else:
            s = []

        if seq_def['seq_type'] == 'ntseq' and self.ntseq is not None:
            s.append(self.ntseq)
        elif seq_def['seq_type'] == 'aaseq' and self.aaseq is not None:
            s.append(self.aaseq)

        if seq_def['use_alleles']:
            if len(self.v_alleles) > 0: s.append('/'.join(self.v_alleles))
            if len(self.j_alleles) > 0: s.append('/'.join(self.j_alleles))
        elif seq_def['use_genes']:
            if len(self.v_genes) > 0: s.append('/'.join(self.v_genes))
            if len(self.j_genes) > 0: s.append('/'.join(self.j_genes))



        #self.seq = self.seq_rep()
        return ','.join(s)

    def compare_full_sequence(self, tcr_seq):
        if type(self) != type(tcr_seq):
            raise ValueError
        return self.__dict__ == tcr_seq.__dict__

    def __repr__(self):
        return "TCRseq(%s)"%(self.seq_rep())

#%

class TcellClone(CloneDefinition):
    """Class for TCR clones (single or paired chains)

    The class stores all provided sequence information for TRA and TRB 
    sequences and will return a standardized syntax string based on the 
    CloneDefinition that can be used as a hash for the clone. Inherits all of 
    the attributes of CloneDefinition.
    
    Attributes
    ----------
    TRA : list of TCRseq
        List of all TCRseq associated with the clone's TRA
    TRB : list of TCRseq
        List of all TCRseq associated with clone's TRB
    count : float
        Count associated with the clone. Generally cell count.
    
    Methods
    -------
    clone_rep(**kwargs)
        Returns standardized sequence string based on the clone definition
        (modified by any **kwargs)
    split_clone_rep(**kwargs)
        Returns all possible standardized sequence string with 1 TRA and 1 TRB
        based on the clone definition (modified by any **kwargs)
    load_TRB(**kwargs)
        Load TRB sequence
    load_TRA(**kwargs)
        Load TRA sequence
    """
    
    def __init__(self, clone_rep = None, **kwargs):

        CloneDefinition.__init__(self, **kwargs)

        self.TRB = []
        self.TRA = []
        self.count = 0

        for kw, kw_val in kwargs.items():
            if kw.upper() == 'TRB':
                self.load_TRB(**kw_val)
            elif kw.upper() == 'TRA':
                self.load_TRA(**kw_val)
            elif kw in self.__dict__:
                self.__dict__[kw] = kw_val
        
        if clone_rep is not None:
            self.load_from_clone_rep(clone_rep, **kwargs)
        
        if len(self.TRB) > 0 and len(self.TRA) == 0:
            self.chains = [c for c in self.chains if c!='TRA']
        elif len(self.TRA) > 0 and len(self.TRB) == 0:
            self.chains = [c for c in self.chains if c!='TRB']
    
    def load_from_clone_rep(self, clone_rep, **kwargs):
        """Load Clone info from clone_rep representation

        Parameters
        ----------
        clone_rep : str
            String representation for clone information
        **kwargs : keywords
            Input for TCRSeq.__init__ to specify TRB sequence

        Attributes modified
        -------------------
        self.TRA
        self.TRB

        """
        seq_reps = clone_rep.split(';')
        for seq_rep in seq_reps:
            if seq_rep.startswith('TRA'):
                self.load_TRA(seq_rep = seq_rep, **kwargs)
                if self.TRA[-1].use_alleles: self.use_alleles = True
            elif seq_rep.startswith('TRB'):
                self.load_TRB(seq_rep = seq_rep, **kwargs)
                if self.TRB[-1].use_alleles: self.use_alleles = True
    
    def load_TRB(self, **kwargs):
        """Load TRB sequence

        Parameters
        ----------
        **kwargs : keywords
            Input for TCRSeq.__init__ to specify TRB sequence

        Attributes modified
        -------------------
        self.TRB

        """
        kwargs['chain'] = 'TRB'
        TRB_seq = TCRseq(**kwargs)
        if TRB_seq.seq_type == 'aaseq' and self.seq_type == 'ntseq':
            self.seq_type = 'aaseq'
        self.TRB.append(TRB_seq)

    def load_TRA(self, **kwargs):
        """Load TRA sequence

        Parameters
        ----------
        **kwargs : keywords
            Input for TCRSeq.__init__ to specify TRA sequence

        Attributes modified
        -------------------
        self.TRA

        """
        kwargs['chain'] = 'TRA'
        TRA_seq = TCRseq(**kwargs)
        if TRA_seq.seq_type == 'aaseq' and self.seq_type == 'ntseq':
            self.seq_type = 'aaseq'
        self.TRA.append(TRA_seq)

    def clone_rep(self, **kwargs):
        """Returns standardized clone string based on clone definition.

        Returns
        -------
        str
            Clone string

        """

        clone_def = self.get_clone_def()
        for kw, kw_val in kwargs.items():
            if kw in clone_def: clone_def[kw] = kw_val

        return ';'.join([c_seq.seq_rep(**clone_def) for c_seq in sum([self.__dict__[c_chain] for c_chain in clone_def['chains']], [])])

    def split_clone_rep(self, **kwargs):
        """Returns all possible clone strings based on clone definition.
        
        If a clone has multiple TRA or TRB sequences this will enumerate the
        possible combinations.

        Returns
        -------
        tuple
            Tuple of clone strings

        """
        clone_def = self.get_clone_def()
        for kw, kw_val in kwargs.items():
            if kw in clone_def: clone_def[kw] = kw_val

        if len(clone_def['chains']) <= 1:
            return tuple([c_seq.seq_rep(**clone_def) for c_seq in self.__dict__[clone_def['chains'][0]]])
        else:
            return tuple([';'.join([c_seq1.seq_rep(**clone_def), c_seq2.seq_rep(**clone_def)]) for c_seq1 in self.__dict__[clone_def['chains'][0]] for c_seq2 in self.__dict__[clone_def['chains'][1]]])

    def __repr__(self):
        return "TcellClone(%s)"%(self.clone_rep())
    


class InputError(Error):
    """Exception for inconsistent input

    Attributes
    ----------
    input_kw : str
    input_arg

    """

    def __init__(self, input_kw, input_arg):
        self.input_kw = input_kw
        self.input_arg = input_arg

class ChainMismatch(Error):
    """Exception for mismatched chains

    Attributes
    ----------
    chainA : str
        TCR chain (i.e. TRA or TRB)
    chainB : str
        TCR chain (i.e. TRA or TRB)

    """

    def __init__(self, chainA, chainB):
        self.chainA = chainA
        self.chainB = chainB

class UnknownGene(Error):
    """Exception for unknown or mischaracterized gene

    Attributes
    ----------
    gene_name : str
        Gene or allele name
    gene_type : char
        Genomic cassette type. (i.e. V, D, or J)

    """

    def __init__(self, gene_name, gene_type = None):
        self.gene_name = gene_name
        self.gene_type = gene_type

class SequenceMismatch(Error):
    """Exception for unknown or mischaracterized gene

    Attributes
    ----------
    gene_name : str
        Gene or allele name
    gene_type : char
        Genomic cassette type. (i.e. V, D, or J)

    """

    def __init__(self, seqA, seqB):
        self.seqA = seqA
        self.seqB = seqB

class PhenotypeMismatch(Error):
    """Exception for mismatched phenotype definitions

    Attributes
    ----------
    phenotypesA : list or int
        list of phenotypes or len(phenotypes)
    phenotypesB : list or int
        list of phenotypes or len(phenotypes)

    """

    def __init__(self, phenotypesA, phenotypesB):
        self.phenotypesA = phenotypesA
        self.phenotypesB = phenotypesB


class SankeyNode(object):
    """Object for sankey node"""
    
    def __init__(self, x, y, val, dx = 0.2, color = None, **kwargs):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = val
        self.x_gap = 0.05
        
        self.max_x = self.x + self.dx/2
        self.min_x = self.x - self.dx/2
        self.max_y = self.y + self.dy
        self.min_y = self.y
        
        self.color = color
        self.patch = mpatches.Rectangle([self.min_x+self.x_gap, self.min_y], self.dx-2*self.x_gap, self.dy, facecolor = self.color, edgecolor = 'None')
    
    def plot(self, ax):
        ax.add_patch(self.patch)


    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

    def plot_node_connection(self, destination_node, ax, **kwargs):
        num_segments = 500                                 
        discretize = np.linspace(0, 1, num_segments)

        # Shape of the connection
        x = self.max_x + (destination_node.min_x - self.max_x) * discretize
        y_shape = 1 / (1 + (10**2 / np.power(10, 4 * discretize)))
        y_shape = (y_shape - y_shape[0]) / (y_shape[-1] - y_shape[0])
        y_top = self.max_y + (destination_node.max_y - self.max_y) * y_shape
        y_bot = self.min_y + (destination_node.min_y - self.min_y) * y_shape

        # Color interpolation
        start_color = np.array(mcolors.to_rgb(self.color))
        end_color = np.array(mcolors.to_rgb(destination_node.color)) if destination_node.color else start_color

        for i in range(num_segments - 1):
            interp_color = (1 - discretize[i]) * start_color + discretize[i] * end_color
            ax.fill_between(x[i:i+2], y_top[i:i+2], y_bot[i:i+2], facecolor=interp_color, edgecolor='none')


def plot_pheno_sankey(phenotypes, cell_repertoires, clones = None, **kwargs):

    fontsize = kwargs.get('fontsize', 12)
    tick_fontsize = kwargs.get('tick_fontsize', fontsize) 

    fontsize_dict = dict(xlabel_fontsize = kwargs.get('label_fontsize', fontsize),
                         ylabel_fontsize = kwargs.get('label_fontsize', fontsize),
                         title_fontsize = kwargs.get('title_fontsize', fontsize),
                         xtick_fontsize = kwargs.get('xtick_fontsize', tick_fontsize),
                         legend_fontsize = kwargs.get('legend_fontsize', fontsize)
                         )
    for kw, val in kwargs.items():
        if kw in fontsize_dict: fontsize_dict[kw] = val

    phenotypes = Phenotypes(phenotypes)
    n_reps = len(cell_repertoires)
    
    
    T10_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    colors = kwargs.get('colors', T10_colors)
    colors = tcri_colors
    phenotype_colors = kwargs.get('phenotype_colors', {phenotype: colors[i] for i, phenotype in enumerate(phenotypes)})

    normalize = kwargs.get('normalize', True)

    if 'times' in kwargs:
        times = np.array(kwargs['times'])*kwargs.get('time_rescale', 1)
        dx = (max(times) - min(times)) / 100
    else:
        times = list(range(n_reps))
    
    origin_nodes = [{} for _ in range(n_reps - 1)]
    destination_nodes = [{} for _ in range(n_reps - 1)]
    plot_nodes = [{} for _ in range(n_reps)]

    if n_reps == 1:
        i = 0
        origin_rep = cell_repertoires[0]
        c_origin_node_vals = np.zeros(len(phenotypes))

        if clones is None:
            c_clones = origin_rep.clones
        else:
            c_clones = clones
        for clone in c_clones:
            if normalize:
                origin_phenotype_vec = origin_rep.get_phenotype_counts(clone, phenotypes)/origin_rep.norm
            else:
                origin_phenotype_vec = origin_rep.get_phenotype_counts(clone, phenotypes)
            c_origin_node_vals += origin_phenotype_vec
                            
        origin_main_node_ys = np.array([0] + list(np.cumsum(c_origin_node_vals[:-1])))
        running_origin_node_ys = origin_main_node_ys.copy()
        for j, origin_phenotype in enumerate(phenotypes):
             plot_nodes[i][origin_phenotype] = SankeyNode(times[i], origin_main_node_ys[j], c_origin_node_vals[j], dx = dx, color = phenotype_colors[origin_phenotype], **kwargs)
    else:
        for i, origin_rep, dest_rep in zip(range(n_reps - 1), cell_repertoires[:-1], cell_repertoires[1:]):
            c_origin_flow_array = np.zeros((len(phenotypes), len(phenotypes)))
            c_dest_flow_array = np.zeros((len(phenotypes), len(phenotypes)))
            
            c_origin_node_vals = np.zeros(len(phenotypes))
            c_dest_node_vals = np.zeros(len(phenotypes))
            if clones is None:
                c_clones = origin_rep.clone_union(dest_rep)
            else:
                c_clones = clones
            for clone in c_clones:
                if normalize:
                    origin_phenotype_vec = origin_rep.get_phenotype_counts(clone, phenotypes)/origin_rep.norm
                    dest_phenotype_vec = dest_rep.get_phenotype_counts(clone, phenotypes)/dest_rep.norm
                else:
                    origin_phenotype_vec = origin_rep.get_phenotype_counts(clone, phenotypes)
                    dest_phenotype_vec = dest_rep.get_phenotype_counts(clone, phenotypes)
                c_origin_node_vals += origin_phenotype_vec
                c_dest_node_vals += dest_phenotype_vec
                if np.sum(origin_phenotype_vec) > 0 and np.sum(dest_phenotype_vec) > 0:
                    c_origin_flow_array += np.dot(origin_phenotype_vec.reshape(-1, 1), (dest_phenotype_vec/np.sum(dest_phenotype_vec)).reshape(1, -1))
                    c_dest_flow_array += np.dot(origin_phenotype_vec.reshape(-1, 1)/np.sum(origin_phenotype_vec), dest_phenotype_vec.reshape(1, -1))
                    
                    #c_dest_flow_array += np.dot(dest_phenotype_vec.reshape(-1, 1), (origin_phenotype_vec/np.sum(origin_phenotype_vec)).reshape(1, -1))
                    
            origin_main_node_ys = np.array([0] + list(np.cumsum(c_origin_node_vals[:-1])))
            dest_main_node_ys = np.array([0] + list(np.cumsum(c_dest_node_vals[:-1])))
            running_origin_node_ys = origin_main_node_ys.copy()
            #running_dest_node_ys = dest_main_node_ys.copy()
            #running_dest_node_ys = np.cumsum(c_dest_node_vals)
            running_dest_node_ys = np.cumsum(c_dest_node_vals) - np.sum(c_dest_flow_array, axis = 0)
            for j, origin_phenotype in enumerate(phenotypes):
                plot_nodes[i][origin_phenotype] = SankeyNode(times[i], origin_main_node_ys[j], c_origin_node_vals[j], dx = dx, color = phenotype_colors[origin_phenotype], **kwargs)
                
                for k, dest_phenotype in enumerate(phenotypes):
                    origin_nodes[i][(origin_phenotype, dest_phenotype)] = SankeyNode(times[i], running_origin_node_ys[j], c_origin_flow_array[j, k], dx = dx, color = phenotype_colors[origin_phenotype], **kwargs)
                    running_origin_node_ys[j] += c_origin_flow_array[j, k]
                    
                    # running_dest_node_ys[k] -= c_dest_flow_array[j, k]
                    # destination_nodes[i][(origin_phenotype, dest_phenotype)] = SankeyNode(times[i+1], running_dest_node_ys[k], c_dest_flow_array[j, k], **kwargs)
                    
                    destination_nodes[i][(origin_phenotype, dest_phenotype)] = SankeyNode(times[i+1], running_dest_node_ys[k], c_dest_flow_array[j, k], dx = dx, color = phenotype_colors[dest_phenotype], **kwargs)
                    running_dest_node_ys[k] += c_dest_flow_array[j, k]
                    
            if i == n_reps - 2:
                for j, phenotype in enumerate(phenotypes):
                    plot_nodes[i+1][phenotype] = SankeyNode(times[i+1], dest_main_node_ys[j], c_dest_node_vals[j], dx = dx, color = phenotype_colors[phenotype], **kwargs)
      
    
    if 'ax' in kwargs:
        ax = kwargs['ax']
    elif 'figsize' in kwargs:
        fig, ax = plt.subplots(figsize = kwargs['figsize'])
    else:
        fig, ax = plt.subplots(figsize = (9, 5))
        
    
    for i in range(n_reps):
        for phenotype in phenotypes:
            plot_nodes[i][phenotype].plot(ax = ax)
        if i < n_reps -1:
            for origin_phenotype in phenotypes:
                for dest_phenotype in phenotypes:
                    origin_nodes[i][(origin_phenotype, dest_phenotype)].plot_node_connection(destination_nodes[i][(origin_phenotype, dest_phenotype)], ax = ax, alpha = 0.5)
    
    if 'xlim' in kwargs:
        ax.set_xlim(kwargs['xlim'])
    if 'ylim' in kwargs:
        ax.set_ylim(kwargs['ylim'])
    else:
        if kwargs.get('normalize', True):
            ax.set_ylim([0, min(ax.get_ylim()[1], 1)])
        else:
            ax.set_ylim([0, ax.get_ylim()[1]])
        
    
    if kwargs.get('show_legend', True):
        ax.legend([plot_nodes[0][phenotype].patch for phenotype in phenotypes], list(phenotypes),frameon=True, fontsize=fontsize_dict['legend_fontsize'],bbox_to_anchor=(1.05, 1), loc='upper left')

    if kwargs.get('plot_seperate_legend', False):
        legend_fig, legend_ax = plt.subplots(figsize = (4, 3))
        legend_ax.legend([plot_nodes[0][phenotype].patch for phenotype in phenotypes], list(phenotypes), frameon = False, fontsize = fontsize_dict['legend_fontsize'])
        legend_fig.tight_layout()
        legend_ax.set_axis_off()

    if 'names' in kwargs:
        names = kwargs['names']
    else:
        try:
            names = [cell_repertoires.name for cell_repertoires in cell_repertoires]
        except AttributeError:
            names = None
        
    if 'xlabel' in kwargs:
        ax.set_xlabel(kwargs['xlabel'], fontsize = fontsize_dict['xlabel_fontsize'])
    elif 'times' not in kwargs and names is not None:
        plt.xticks(times, names, rotation = -30, fontsize = fontsize_dict['xtick_fontsize'])
        plt.setp(ax.xaxis.get_majorticklabels(), ha="left", rotation_mode="anchor")
    if 'ylabel' in kwargs:
        ax.set_ylabel(kwargs['ylabel'], fontsize = fontsize_dict['ylabel_fontsize'])
    else:
        if kwargs.get('normalize', True):
            ax.set_ylabel('Fraction', fontsize = fontsize_dict['ylabel_fontsize'])
        else:
            ax.set_ylabel('Cell Counts', fontsize = fontsize_dict['ylabel_fontsize'])

    if 'title' in kwargs:
        ax.set_title(kwargs['title'], fontsize = fontsize_dict['title_fontsize'])

    if 'ax' not in kwargs:
        fig.tight_layout()

    if 'savefig_filename' in kwargs and 'ax' not in kwargs:
        fig.savefig(kwargs['savefig_filename'])

    if kwargs.get('return_axes', False):
        return fig, ax    
    

def plot_pheno_sankey_bulk_norm(phenotypes, cell_repertoires, bulk_cell_repertoires, clones = None, **kwargs):

    fontsize = kwargs.get('fontsize', 12)
    tick_fontsize = kwargs.get('tick_fontsize', fontsize) 

    fontsize_dict = dict(xlabel_fontsize = kwargs.get('label_fontsize', fontsize),
                         ylabel_fontsize = kwargs.get('label_fontsize', fontsize),
                         title_fontsize = kwargs.get('title_fontsize', fontsize),
                         xtick_fontsize = kwargs.get('xtick_fontsize', tick_fontsize),
                         legend_fontsize = kwargs.get('legend_fontsize', fontsize)
                         )
    for kw, val in kwargs.items():
        if kw in fontsize_dict: fontsize_dict[kw] = val

    
    
    phenotypes = Phenotypes(phenotypes)
    n_reps = len(cell_repertoires)
    
    
    T10_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    colors = kwargs.get('colors', T10_colors)
    phenotype_colors = kwargs.get('phenotype_colors', {phenotype: colors[i] for i, phenotype in enumerate(phenotypes)})

    if 'times' in kwargs:
        times = np.array(kwargs['times'])*kwargs.get('time_rescale', 1)
        
    else:
        times = list(range(n_reps))
    
    dx = (max(times) - min(times)) / 100
    
    origin_nodes = [{} for _ in range(n_reps - 1)]
    destination_nodes = [{} for _ in range(n_reps - 1)]
    plot_nodes = [{} for _ in range(n_reps)]

    if n_reps == 1:
        i = 0
        origin_rep = cell_repertoires[0]
        origin_bulk_rep = bulk_cell_repertoires[0]
        c_origin_node_vals = np.zeros(len(phenotypes))

        if clones is None:
            c_clones = origin_rep.clones
        else:
            c_clones = clones
        for clone in c_clones:
            origin_phenotype_vec = (origin_rep.get_phenotype_counts(clone, phenotypes)/np.sum(origin_rep.get_phenotype_counts(clone, phenotypes)))*origin_bulk_rep.get_frequency(clone)
            c_origin_node_vals += origin_phenotype_vec
                            
        origin_main_node_ys = np.array([0] + list(np.cumsum(c_origin_node_vals[:-1])))
        running_origin_node_ys = origin_main_node_ys.copy()
        for j, origin_phenotype in enumerate(phenotypes):
            plot_nodes[i][origin_phenotype] = SankeyNode(times[i], origin_main_node_ys[j], c_origin_node_vals[j], dx = dx, color = phenotype_colors[origin_phenotype], **kwargs)
    else:
        for i, origin_rep, dest_rep, origin_bulk_rep, dest_bulk_rep in zip(range(n_reps - 1), cell_repertoires[:-1], cell_repertoires[1:], bulk_cell_repertoires[:-1], bulk_cell_repertoires[1:]):
            c_origin_flow_array = np.zeros((len(phenotypes), len(phenotypes)))
            c_dest_flow_array = np.zeros((len(phenotypes), len(phenotypes)))
            
            c_origin_node_vals = np.zeros(len(phenotypes))
            c_dest_node_vals = np.zeros(len(phenotypes))
            if clones is None:
                c_clones = origin_rep.clone_union(dest_rep)
            else:
                c_clones = clones
            for clone in c_clones:
                origin_phenotype_vec = (origin_rep.get_phenotype_counts(clone, phenotypes)/np.sum(origin_rep.get_phenotype_counts(clone, phenotypes)))*origin_bulk_rep.get_frequency(clone)
                dest_phenotype_vec = (dest_rep.get_phenotype_counts(clone, phenotypes)/np.sum(dest_rep.get_phenotype_counts(clone, phenotypes)))*dest_bulk_rep.get_frequency(clone)
                c_origin_node_vals += origin_phenotype_vec
                c_dest_node_vals += dest_phenotype_vec
                if np.sum(origin_phenotype_vec) > 0 and np.sum(dest_phenotype_vec) > 0:
                    c_origin_flow_array += np.dot(origin_phenotype_vec.reshape(-1, 1), (dest_phenotype_vec/np.sum(dest_phenotype_vec)).reshape(1, -1))
                    c_dest_flow_array += np.dot(origin_phenotype_vec.reshape(-1, 1)/np.sum(origin_phenotype_vec), dest_phenotype_vec.reshape(1, -1))
                    
                    #c_dest_flow_array += np.dot(dest_phenotype_vec.reshape(-1, 1), (origin_phenotype_vec/np.sum(origin_phenotype_vec)).reshape(1, -1))
                    
            origin_main_node_ys = np.array([0] + list(np.cumsum(c_origin_node_vals[:-1])))
            dest_main_node_ys = np.array([0] + list(np.cumsum(c_dest_node_vals[:-1])))
            running_origin_node_ys = origin_main_node_ys.copy()
            #running_dest_node_ys = dest_main_node_ys.copy()
            #running_dest_node_ys = np.cumsum(c_dest_node_vals)
            running_dest_node_ys = np.cumsum(c_dest_node_vals) - np.sum(c_dest_flow_array, axis = 0)
            for j, origin_phenotype in enumerate(phenotypes):
                plot_nodes[i][origin_phenotype] = SankeyNode(times[i], origin_main_node_ys[j], c_origin_node_vals[j], dx = dx, color = phenotype_colors[origin_phenotype], **kwargs)
                
                for k, dest_phenotype in enumerate(phenotypes):
                    origin_nodes[i][(origin_phenotype, dest_phenotype)] = SankeyNode(times[i], running_origin_node_ys[j], c_origin_flow_array[j, k], dx = dx, color = phenotype_colors[origin_phenotype], **kwargs)
                    running_origin_node_ys[j] += c_origin_flow_array[j, k]
                    
                    # running_dest_node_ys[k] -= c_dest_flow_array[j, k]
                    # destination_nodes[i][(origin_phenotype, dest_phenotype)] = SankeyNode(times[i+1], running_dest_node_ys[k], c_dest_flow_array[j, k], **kwargs)
                    
                    destination_nodes[i][(origin_phenotype, dest_phenotype)] = SankeyNode(times[i+1], running_dest_node_ys[k], c_dest_flow_array[j, k], dx = dx, **kwargs)
                    running_dest_node_ys[k] += c_dest_flow_array[j, k]
                    
            if i == n_reps - 2:
                for j, phenotype in enumerate(phenotypes):
                    plot_nodes[i+1][phenotype] = SankeyNode(times[i+1], dest_main_node_ys[j], c_dest_node_vals[j], dx = dx, color = phenotype_colors[phenotype], **kwargs)
      
    
    if 'ax' in kwargs:
        ax = kwargs['ax']
    elif 'figsize' in kwargs:
        fig, ax = plt.subplots(figsize = kwargs['figsize'])
    else:
        fig, ax = plt.subplots(figsize = (9, 5))
        
    
    for i in range(n_reps):
        for phenotype in phenotypes:
            plot_nodes[i][phenotype].plot(ax = ax)
        if i < n_reps -1:
            for origin_phenotype in phenotypes:
                for dest_phenotype in phenotypes:
                    origin_nodes[i][(origin_phenotype, dest_phenotype)].plot_node_connection(destination_nodes[i][(origin_phenotype, dest_phenotype)], ax = ax, alpha = 0.5)
    
    if 'xlim' in kwargs:
        ax.set_xlim(kwargs['xlim'])
    if 'ylim' in kwargs:
        ax.set_ylim(kwargs['ylim'])
    else:
        if kwargs.get('normalize', True):
            ax.set_ylim([0, min(ax.get_ylim()[1], 1)])
        else:
            ax.set_ylim([0, ax.get_ylim()[1]])
        
    
    if kwargs.get('show_legend', True):
        ax.legend([plot_nodes[0][phenotype].patch for phenotype in phenotypes], list(phenotypes), frameon = True, fontsize = fontsize_dict['legend_fontsize'], bbox_to_anchor=(1, 0.9))
    
    if kwargs.get('plot_seperate_legend', False):
        legend_fig, legend_ax = plt.subplots(figsize = (4, 3))
        legend_ax.legend([plot_nodes[0][phenotype].patch for phenotype in phenotypes], list(phenotypes), frameon = False, fontsize = fontsize_dict['legend_fontsize'])
        legend_fig.tight_layout()
        legend_ax.set_axis_off()

    if 'names' in kwargs:
        names = kwargs['names']
    else:
        try:
            names = [cell_repertoires.name for cell_repertoires in cell_repertoires]
        except AttributeError:
            names = None
        
    if 'xlabel' in kwargs:
        ax.set_xlabel(kwargs['xlabel'], fontsize = fontsize_dict['xlabel_fontsize'])
    elif 'times' not in kwargs and names is not None:
        plt.xticks(times, names, rotation = -30, fontsize = fontsize_dict['xtick_fontsize'])
        plt.setp(ax.xaxis.get_majorticklabels(), ha="left", rotation_mode="anchor")
    if 'ylabel' in kwargs:
        ax.set_ylabel(kwargs['ylabel'], fontsize = fontsize_dict['ylabel_fontsize'])
    else:
        if kwargs.get('normalize', True):
            ax.set_ylabel('Fraction', fontsize = fontsize_dict['ylabel_fontsize'])
        else:
            ax.set_ylabel('Cell Counts', fontsize = fontsize_dict['ylabel_fontsize'])

    if 'title' in kwargs:
        ax.set_title(kwargs['title'], fontsize = fontsize_dict['title_fontsize'])

    if 'ax' not in kwargs:
        fig.tight_layout()

    if 'savefig_filename' in kwargs and 'ax' not in kwargs:
        fig.savefig(kwargs['savefig_filename'])

        
    
    
    if kwargs.get('return_axes', False):
        return fig, ax
def plot_agg_pheno_sankey(phenotypes, cell_repertoires_by_pt, clones_by_pt = None, **kwargs):

    fontsize = kwargs.get('fontsize', 12)
    tick_fontsize = kwargs.get('tick_fontsize', fontsize) 

    fontsize_dict = dict(xlabel_fontsize = kwargs.get('label_fontsize', fontsize),
                         ylabel_fontsize = kwargs.get('label_fontsize', fontsize),
                         title_fontsize = kwargs.get('title_fontsize', fontsize),
                         xtick_fontsize = kwargs.get('xtick_fontsize', tick_fontsize),
                         legend_fontsize = kwargs.get('legend_fontsize', fontsize)
                         )
    for kw, val in kwargs.items():
        if kw in fontsize_dict: fontsize_dict[kw] = val

    
    
    phenotypes = Phenotypes(phenotypes)
    n_reps = cell_repertoires_by_pt.shape[1]
    n_pts = cell_repertoires_by_pt.shape[0]
    
    T10_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    colors = kwargs.get('colors', T10_colors)
    phenotype_colors = kwargs.get('phenotype_colors', {phenotype: colors[i] for i, phenotype in enumerate(phenotypes)})

    
    if 'times' in kwargs:
        times = np.array(kwargs['times'])*kwargs.get('time_rescale', 1)
        
    else:
        times = list(range(n_reps))
    
    dx = (max(times) - min(times)) / 100
    
    origin_nodes = [{} for _ in range(n_reps - 1)]
    destination_nodes = [{} for _ in range(n_reps - 1)]
    plot_nodes = [{} for _ in range(n_reps)]

    if n_reps == 1:
        i = 0
        c_origin_node_vals = np.zeros(len(phenotypes))
        for pt_ind, origin_rep in enumerate(cell_repertoires_by_pt[:, 0]):
        
            if clones_by_pt is None or clones_by_pt[pt_ind] is None:
                c_clones = origin_rep.clones
            else:
                c_clones = clones_by_pt[pt_ind]
            for clone in c_clones:
                origin_phenotype_vec = (origin_rep.get_phenotype_counts(clone, phenotypes)/origin_rep.norm)/n_pts
                c_origin_node_vals += origin_phenotype_vec
                            
        origin_main_node_ys = np.array([0] + list(np.cumsum(c_origin_node_vals[:-1])))
        running_origin_node_ys = origin_main_node_ys.copy()
        for j, origin_phenotype in enumerate(phenotypes):
            plot_nodes[i][origin_phenotype] = SankeyNode(times[i], origin_main_node_ys[j], c_origin_node_vals[j], dx = dx, color = phenotype_colors[origin_phenotype], **kwargs)
    else:
        for i in range(n_reps - 1):
            origin_reps = cell_repertoires_by_pt[:, i]
            dest_reps = cell_repertoires_by_pt[:, i+1]
            
            c_origin_flow_array = np.zeros((len(phenotypes), len(phenotypes)))
            c_dest_flow_array = np.zeros((len(phenotypes), len(phenotypes)))
            
            c_origin_node_vals = np.zeros(len(phenotypes))
            c_dest_node_vals = np.zeros(len(phenotypes))
            
            for pt_ind, origin_rep, dest_rep in zip(range(n_pts), origin_reps, dest_reps):
                if clones_by_pt is None or clones_by_pt[pt_ind] is None:
                    c_clones = origin_rep.clones
                else:
                    c_clones = clones_by_pt[pt_ind]
                for clone in c_clones:
    
                    origin_phenotype_vec = (origin_rep.get_phenotype_counts(clone, phenotypes)/origin_rep.norm)/n_pts
                    dest_phenotype_vec = (dest_rep.get_phenotype_counts(clone, phenotypes)/dest_rep.norm)/n_pts
                    
                    c_origin_node_vals += origin_phenotype_vec
                    c_dest_node_vals += dest_phenotype_vec
                    if np.sum(origin_phenotype_vec) > 0 and np.sum(dest_phenotype_vec) > 0:
                        c_origin_flow_array += np.dot(origin_phenotype_vec.reshape(-1, 1), (dest_phenotype_vec/np.sum(dest_phenotype_vec)).reshape(1, -1))
                        c_dest_flow_array += np.dot(origin_phenotype_vec.reshape(-1, 1)/np.sum(origin_phenotype_vec), dest_phenotype_vec.reshape(1, -1))
                        
                        #c_dest_flow_array += np.dot(dest_phenotype_vec.reshape(-1, 1), (origin_phenotype_vec/np.sum(origin_phenotype_vec)).reshape(1, -1))
                    
            origin_main_node_ys = np.array([0] + list(np.cumsum(c_origin_node_vals[:-1])))
            dest_main_node_ys = np.array([0] + list(np.cumsum(c_dest_node_vals[:-1])))
            running_origin_node_ys = origin_main_node_ys.copy()
            #running_dest_node_ys = dest_main_node_ys.copy()
            #running_dest_node_ys = np.cumsum(c_dest_node_vals)
            running_dest_node_ys = np.cumsum(c_dest_node_vals) - np.sum(c_dest_flow_array, axis = 0)
            for j, origin_phenotype in enumerate(phenotypes):
                plot_nodes[i][origin_phenotype] = SankeyNode(times[i], origin_main_node_ys[j], c_origin_node_vals[j], dx = dx, color = phenotype_colors[origin_phenotype], **kwargs)
                
                for k, dest_phenotype in enumerate(phenotypes):
                    origin_nodes[i][(origin_phenotype, dest_phenotype)] = SankeyNode(times[i], running_origin_node_ys[j], c_origin_flow_array[j, k], dx = dx, color = phenotype_colors[origin_phenotype], **kwargs)
                    running_origin_node_ys[j] += c_origin_flow_array[j, k]
                    
                    # running_dest_node_ys[k] -= c_dest_flow_array[j, k]
                    # destination_nodes[i][(origin_phenotype, dest_phenotype)] = SankeyNode(times[i+1], running_dest_node_ys[k], c_dest_flow_array[j, k], **kwargs)
                    
                    destination_nodes[i][(origin_phenotype, dest_phenotype)] = SankeyNode(times[i+1], running_dest_node_ys[k], c_dest_flow_array[j, k], dx = dx, **kwargs)
                    running_dest_node_ys[k] += c_dest_flow_array[j, k]
                    
            if i == n_reps - 2:
                for j, phenotype in enumerate(phenotypes):
                    plot_nodes[i+1][phenotype] = SankeyNode(times[i+1], dest_main_node_ys[j], c_dest_node_vals[j], dx = dx, color = phenotype_colors[phenotype], **kwargs)
      
    
    if 'ax' in kwargs:
        ax = kwargs['ax']
    elif 'figsize' in kwargs:
        fig, ax = plt.subplots(figsize = kwargs['figsize'])
    else:
        fig, ax = plt.subplots(figsize = (9, 5))
        
    
    for i in range(n_reps):
        for phenotype in phenotypes:
            plot_nodes[i][phenotype].plot(ax = ax)
        if i < n_reps -1:
            for origin_phenotype in phenotypes:
                for dest_phenotype in phenotypes:
                    origin_nodes[i][(origin_phenotype, dest_phenotype)].plot_node_connection(destination_nodes[i][(origin_phenotype, dest_phenotype)], ax = ax, alpha = 0.5)
    
    if 'xlim' in kwargs:
        ax.set_xlim(kwargs['xlim'])
    if 'ylim' in kwargs:
        ax.set_ylim(kwargs['ylim'])
    else:
        if kwargs.get('normalize', True):
            ax.set_ylim([0, min(ax.get_ylim()[1], 1)])
        else:
            ax.set_ylim([0, ax.get_ylim()[1]])
        
    
    if kwargs.get('show_legend', True):
        ax.legend([plot_nodes[0][phenotype].patch for phenotype in phenotypes], list(phenotypes), frameon = True, fontsize = fontsize_dict['legend_fontsize'], bbox_to_anchor=(1, 0.9))
    
    if kwargs.get('plot_seperate_legend', False):
        legend_fig, legend_ax = plt.subplots(figsize = (4, 3))
        legend_ax.legend([plot_nodes[0][phenotype].patch for phenotype in phenotypes], list(phenotypes), frameon = False, fontsize = fontsize_dict['legend_fontsize'])
        legend_fig.tight_layout()
        legend_ax.set_axis_off()

    if 'names' in kwargs:
        names = kwargs['names']
    else:
        try:
            names = [cell_repertoire.name for cell_repertoire in cell_repertoires_by_pt[0]]
        except AttributeError:
            names = None
        
    if 'xlabel' in kwargs:
        ax.set_xlabel(kwargs['xlabel'], fontsize = fontsize_dict['xlabel_fontsize'])
    elif 'times' not in kwargs and names is not None:
        plt.xticks(times, names, rotation = -30, fontsize = fontsize_dict['xtick_fontsize'])
        plt.setp(ax.xaxis.get_majorticklabels(), ha="left", rotation_mode="anchor")
    if 'ylabel' in kwargs:
        ax.set_ylabel(kwargs['ylabel'], fontsize = fontsize_dict['ylabel_fontsize'])
    else:
        if kwargs.get('normalize', True):
            ax.set_ylabel('Fraction', fontsize = fontsize_dict['ylabel_fontsize'])
        else:
            ax.set_ylabel('Cell Counts', fontsize = fontsize_dict['ylabel_fontsize'])

    if 'title' in kwargs:
        ax.set_title(kwargs['title'], fontsize = fontsize_dict['title_fontsize'])

    if 'ax' not in kwargs:
        fig.tight_layout()

    if 'savefig_filename' in kwargs and 'ax' not in kwargs:
        fig.savefig(kwargs['savefig_filename'])

    
    if kwargs.get('return_axes', False):
        return fig, ax
def plot_agg_pheno_sankey_w_dict(phenotypes, cell_repertoires_by_tp, clones_by_pt = None, **kwargs):

    fontsize = kwargs.get('fontsize', 12)
    tick_fontsize = kwargs.get('tick_fontsize', fontsize) 

    fontsize_dict = dict(xlabel_fontsize = kwargs.get('label_fontsize', fontsize),
                         ylabel_fontsize = kwargs.get('label_fontsize', fontsize),
                         title_fontsize = kwargs.get('title_fontsize', fontsize),
                         xtick_fontsize = kwargs.get('xtick_fontsize', tick_fontsize),
                         legend_fontsize = kwargs.get('legend_fontsize', fontsize)
                         )
    for kw, val in kwargs.items():
        if kw in fontsize_dict: fontsize_dict[kw] = val

    
    
    phenotypes = Phenotypes(phenotypes)
    n_phases = len(cell_repertoires_by_tp)
    
    
    T10_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    colors = kwargs.get('colors', T10_colors)
    phenotype_colors = kwargs.get('phenotype_colors', {phenotype: colors[i] for i, phenotype in enumerate(phenotypes)})

    
    if 'times' in kwargs:
        times = np.array(kwargs['times'])*kwargs.get('time_rescale', 1)
        
    else:
        times = list(range(n_phases))
    
    dx = (max(times) - min(times)) / 100
    
    origin_nodes = [{} for _ in range(n_phases - 1)]
    destination_nodes = [{} for _ in range(n_phases - 1)]
    plot_nodes = [{} for _ in range(n_phases)]

    if n_phases == 1:
        i = 0
        c_origin_node_vals = np.zeros(len(phenotypes))
        c_n_pts = len(cell_repertoires_by_tp[0])
        for pt, origin_rep in cell_repertoires_by_tp[0].items():
    
            if clones_by_pt is None or clones_by_pt[pt] is None:
                c_clones = origin_rep.clones
            else:
                c_clones = clones_by_pt[pt]
            for clone in c_clones:
                origin_phenotype_vec = (origin_rep.get_phenotype_counts(clone, phenotypes)/origin_rep.norm)/c_n_pts
                c_origin_node_vals += origin_phenotype_vec
                            
        origin_main_node_ys = np.array([0] + list(np.cumsum(c_origin_node_vals[:-1])))
        running_origin_node_ys = origin_main_node_ys.copy()
        for j, origin_phenotype in enumerate(phenotypes):
            plot_nodes[i][origin_phenotype] = SankeyNode(times[i], origin_main_node_ys[j], c_origin_node_vals[j], dx = dx, color = phenotype_colors[origin_phenotype], **kwargs)
    else:
        for i in range(n_phases - 1):
            print(i)
            origin_reps = cell_repertoires_by_tp[i]
            dest_reps = cell_repertoires_by_tp[i+1]
            
            c_n_origin_pts = len(origin_reps)
            c_n_dest_pts = len(dest_reps)
            
            c_origin_flow_array = np.zeros((len(phenotypes), len(phenotypes)))
            c_dest_flow_array = np.zeros((len(phenotypes), len(phenotypes)))
            
            c_origin_node_vals = np.zeros(len(phenotypes))
            c_dest_node_vals = np.zeros(len(phenotypes))
            
            for pt, origin_rep in origin_reps.items():
                if pt in dest_reps:
                    dest_rep = dest_reps[pt]
                if clones_by_pt is None or clones_by_pt[pt] is None:
                    c_clones = origin_rep.clones
                else:
                    c_clones = clones_by_pt[pt]
                for clone in c_clones:
    
                    origin_phenotype_vec = (origin_rep.get_phenotype_counts(clone, phenotypes)/origin_rep.norm)/c_n_origin_pts
                    if pt in dest_reps:
                        dest_phenotype_vec = (dest_rep.get_phenotype_counts(clone, phenotypes)/dest_rep.norm)/c_n_dest_pts
                    else:
                        dest_phenotype_vec = np.zeros(len(origin_phenotype_vec))
                    c_origin_node_vals += origin_phenotype_vec
                    c_dest_node_vals += dest_phenotype_vec
                    if np.sum(origin_phenotype_vec) > 0 and np.sum(dest_phenotype_vec) > 0:
                        c_origin_flow_array += np.dot(origin_phenotype_vec.reshape(-1, 1), (dest_phenotype_vec/np.sum(dest_phenotype_vec)).reshape(1, -1))
                        c_dest_flow_array += np.dot(origin_phenotype_vec.reshape(-1, 1)/np.sum(origin_phenotype_vec), dest_phenotype_vec.reshape(1, -1))
                        
                        #c_dest_flow_array += np.dot(dest_phenotype_vec.reshape(-1, 1), (origin_phenotype_vec/np.sum(origin_phenotype_vec)).reshape(1, -1))
                    
            origin_main_node_ys = np.array([0] + list(np.cumsum(c_origin_node_vals[:-1])))
            dest_main_node_ys = np.array([0] + list(np.cumsum(c_dest_node_vals[:-1])))
            running_origin_node_ys = origin_main_node_ys.copy()
            #running_dest_node_ys = dest_main_node_ys.copy()
            #running_dest_node_ys = np.cumsum(c_dest_node_vals)
            running_dest_node_ys = np.cumsum(c_dest_node_vals) - np.sum(c_dest_flow_array, axis = 0)
            for j, origin_phenotype in enumerate(phenotypes):
                plot_nodes[i][origin_phenotype] = SankeyNode(times[i], origin_main_node_ys[j], c_origin_node_vals[j], dx = dx, color = phenotype_colors[origin_phenotype], **kwargs)
                
                for k, dest_phenotype in enumerate(phenotypes):
                    origin_nodes[i][(origin_phenotype, dest_phenotype)] = SankeyNode(times[i], running_origin_node_ys[j], c_origin_flow_array[j, k], dx = dx, color = phenotype_colors[origin_phenotype], **kwargs)
                    running_origin_node_ys[j] += c_origin_flow_array[j, k]
                    
                    # running_dest_node_ys[k] -= c_dest_flow_array[j, k]
                    # destination_nodes[i][(origin_phenotype, dest_phenotype)] = SankeyNode(times[i+1], running_dest_node_ys[k], c_dest_flow_array[j, k], **kwargs)
                    
                    destination_nodes[i][(origin_phenotype, dest_phenotype)] = SankeyNode(times[i+1], running_dest_node_ys[k], c_dest_flow_array[j, k], dx = dx, **kwargs)
                    running_dest_node_ys[k] += c_dest_flow_array[j, k]
                    
            if i == n_phases - 2:
                for j, phenotype in enumerate(phenotypes):
                    plot_nodes[i+1][phenotype] = SankeyNode(times[i+1], dest_main_node_ys[j], c_dest_node_vals[j], dx = dx, color = phenotype_colors[phenotype], **kwargs)
      
    
    if 'ax' in kwargs:
        ax = kwargs['ax']
    elif 'figsize' in kwargs:
        fig, ax = plt.subplots(figsize = kwargs['figsize'])
    else:
        fig, ax = plt.subplots(figsize = (9, 5))
        
    
    for i in range(n_phases):
        for phenotype in phenotypes:
            plot_nodes[i][phenotype].plot(ax = ax)
        if i < n_phases -1:
            for origin_phenotype in phenotypes:
                for dest_phenotype in phenotypes:
                    origin_nodes[i][(origin_phenotype, dest_phenotype)].plot_node_connection(destination_nodes[i][(origin_phenotype, dest_phenotype)], ax = ax, alpha = 0.5)
    
    if 'xlim' in kwargs:
        ax.set_xlim(kwargs['xlim'])
    if 'ylim' in kwargs:
        ax.set_ylim(kwargs['ylim'])
    else:
        if kwargs.get('normalize', True):
            ax.set_ylim([0, min(ax.get_ylim()[1], 1)])
        else:
            ax.set_ylim([0, ax.get_ylim()[1]])
        
    
    if kwargs.get('show_legend', True):
        ax.legend([plot_nodes[0][phenotype].patch for phenotype in phenotypes], list(phenotypes), frameon = True, fontsize = fontsize_dict['legend_fontsize'], bbox_to_anchor=(1, 0.9))
    
    if kwargs.get('plot_seperate_legend', False):
        legend_fig, legend_ax = plt.subplots(figsize = (4, 3))
        legend_ax.legend([plot_nodes[0][phenotype].patch for phenotype in phenotypes], list(phenotypes), frameon = False, fontsize = fontsize_dict['legend_fontsize'])
        legend_fig.tight_layout()
        legend_ax.set_axis_off()

    if 'names' in kwargs:
        names = kwargs['names']
    else:
        try:
            names = [list(cell_repertoires.values())[0].name for cell_repertoires in cell_repertoires_by_tp]
        except AttributeError:
            names = None
        
    if 'xlabel' in kwargs:
        ax.set_xlabel(kwargs['xlabel'], fontsize = fontsize_dict['xlabel_fontsize'])
    elif 'times' not in kwargs and names is not None:
        plt.xticks(times, names, rotation = -30, fontsize = fontsize_dict['xtick_fontsize'])
        plt.setp(ax.xaxis.get_majorticklabels(), ha="left", rotation_mode="anchor")
    if 'ylabel' in kwargs:
        ax.set_ylabel(kwargs['ylabel'], fontsize = fontsize_dict['ylabel_fontsize'])
    else:
        if kwargs.get('normalize', True):
            ax.set_ylabel('Fraction', fontsize = fontsize_dict['ylabel_fontsize'])
        else:
            ax.set_ylabel('Cell Counts', fontsize = fontsize_dict['ylabel_fontsize'])

    if 'title' in kwargs:
        ax.set_title(kwargs['title'], fontsize = fontsize_dict['title_fontsize'])

    if 'ax' not in kwargs:
        fig.tight_layout()

    if 'savefig_filename' in kwargs and 'ax' not in kwargs:
        fig.savefig(kwargs['savefig_filename'])

        
    
    
    if kwargs.get('return_axes', False):
        return fig, ax

def setup_ternary_plot(**kwargs):
    
    fig, ax = plt.subplots(figsize = kwargs.get('figsize', (5.5, 5)))
    
    ax.axis('off')
    
    ax.plot(np.array([0, 1, 1/2, 0]), np.array([0, 0, np.sqrt(3)/2, 0]), 'k', lw = 2)
    
    # offset = 0.05
    # ax.text((-np.sqrt(3)/2)*offset, (-1/2)*offset, 'left label', ha = 'center', va = 'center', rotation = 120)
    # ax.text(1 + (np.sqrt(3)/2)*offset, (-1/2)*offset, 'right label', ha = 'center', va = 'center', rotation = 60)
    # ax.text(0.5, np.sqrt(3)/2 + offset, 'top label', ha = 'center', va = 'center')
    
    major_grid_arr = np.arange(0, 1, 0.2)
    minor_grid_arr = np.arange(0, 1, 0.1)
    
    if kwargs.get('gridlines_on', True):
        for grid_pnt in major_grid_arr:
            ax.plot([1-grid_pnt, (1-grid_pnt)/2], [0, np.sqrt(3)*(1-grid_pnt)/2], c = 'k', ls = ':', lw = 0.5)
            ax.plot([grid_pnt/2, 1-grid_pnt/2], [np.sqrt(3)* grid_pnt/2, np.sqrt(3)* grid_pnt/2], c = 'k', ls = ':', lw = 0.5)
            ax.plot([grid_pnt, (1+grid_pnt)/2], [0, np.sqrt(3)*(1-grid_pnt)/2], c = 'k', ls = ':', lw = 0.5)
        
        for grid_pnt in minor_grid_arr:
            ax.plot([1-grid_pnt, (1-grid_pnt)/2], [0, np.sqrt(3)*(1-grid_pnt)/2], c = 'k', ls = ':', lw = 0.25)
            ax.plot([grid_pnt/2, 1-grid_pnt/2], [np.sqrt(3)* grid_pnt/2, np.sqrt(3)* grid_pnt/2], c = 'k', ls = ':', lw = 0.25)
            ax.plot([grid_pnt, (1+grid_pnt)/2], [0, np.sqrt(3)*(1-grid_pnt)/2], c = 'k', ls = ':', lw = 0.25)

    if kwargs.get('ticks_on', True):
        tick_fontsize = kwargs.get('tick_fontsize', 12)
        tick_len = kwargs.get('tick_length', tick_fontsize/600)
        tick_offset = kwargs.get('tick_offset', tick_fontsize/400)
        tick_width = kwargs.get('tick_width', tick_fontsize/12)
        for grid_pnt in major_grid_arr[1:]:
            
            #Ticks on left for p[1]
            ax.plot([(1-grid_pnt)/2, (1-grid_pnt)/2 - np.sqrt(3)*tick_len/2], [np.sqrt(3)*(1-grid_pnt)/2, np.sqrt(3)*(1-grid_pnt)/2 + tick_len/2], c = 'k', lw = tick_width)
            ax.text((1-grid_pnt)/2 - np.sqrt(3)*(tick_len+1.5*tick_offset)/2, np.sqrt(3)*(1-grid_pnt)/2 + (tick_len+tick_offset)/2, '%.1f'%(grid_pnt), ha = 'center', va = 'center', fontsize = tick_fontsize)
    
            
            #Ticks on right for p[0]
            ax.plot([1-grid_pnt/2, 1-grid_pnt/2 + np.sqrt(3)*tick_len/2], [np.sqrt(3)* grid_pnt/2, np.sqrt(3)* grid_pnt/2 + tick_len/2], c = 'k', lw = tick_width)
            ax.text(1-grid_pnt/2 + np.sqrt(3)*(tick_len+1.5*tick_offset)/2, np.sqrt(3)* grid_pnt/2 + (tick_len+tick_offset)/2, '%.1f'%(grid_pnt), ha = 'center', va = 'center', fontsize = tick_fontsize)
    
            #Ticks on bottom for p[2]
            ax.plot([grid_pnt, grid_pnt], [0, -tick_len], c = 'k', lw = tick_width)
            ax.text(grid_pnt, -tick_len - tick_offset, '%.1f'%(grid_pnt), ha = 'center', va = 'center', fontsize = tick_fontsize)

    ax.set_aspect('equal', 'box')
    return fig, ax
    
def set_ternary_corner_label(ax, top_label = None, left_label = None, right_label = None, fontsize = 16, offset = 0.04):
    if top_label is not None: ax.text(0.5, np.sqrt(3)/2 + offset, top_label, ha = 'center', va = 'center', fontsize = fontsize)
    if left_label is not None: ax.text((-np.sqrt(3)/2)*offset, (-1/2)*offset, left_label, ha = 'center', va = 'center', rotation = 120, fontsize = fontsize)
    if right_label is not None: ax.text(1 + (np.sqrt(3)/2)*offset, (-1/2)*offset, right_label, ha = 'center', va = 'center', rotation = 60, fontsize = fontsize)

def set_ternary_axis_label(ax, right_label = None, left_label = None, bottom_label = None, fontsize = 16, offset = 0.1, add_arrows = True):
    if add_arrows:
        right_label = r'$\longleftarrow$' + right_label + r'$\longleftarrow$'
        left_label = r'$\longleftarrow$' + left_label + r'$\longleftarrow$'
        bottom_label = r'$\longrightarrow$' + bottom_label + r'$\longrightarrow$'

    if right_label is not None: ax.text(1-1/4 + np.sqrt(3)*(1.5*offset)/2, np.sqrt(3)/4 + offset/2, right_label, ha = 'center', va = 'center', rotation = -60, fontsize = fontsize)
    if left_label is not None: ax.text((1-1/2)/2 - np.sqrt(3)*(1.5*offset)/2, np.sqrt(3)*(1-1/2)/2 + offset/2, left_label, ha = 'center', va = 'center', rotation = 60, fontsize = fontsize)
    if bottom_label is not None: ax.text(1/2, -offset, bottom_label, ha = 'center', va = 'center', fontsize = fontsize)
    

def ternary_plot_projection(p):
    return np.array([1-p[1] - p[0]/2, np.sqrt(3)*p[0]/2])

def plot_pheno_ternary_change_plots(start_clones_and_phenotypes, end_clones_and_phenotypes = None, phenotypes = None, clones = None, line_type = 'arrows', c_dict = {}, s_dict = {}, kwargs_for_plots = {}, **kwargs):
    """Plots clone trajectories

    Parameters
    ----------
    clones_and_phenos : ClonesAndPhenotypes

    Returns
    -------
    None : NoneType
        If keyword return_axes == False (Default), only None is returned
    ax, cbar, fig : tuple
        If keyword return_axes == True, the axes, colorbar, and figure are
        returned. If no c_dict is provided the cbar returned will be None,
        if ax is provided as a kwargs the fig returned will be None.

    """
    
    fontsize = kwargs.get('fontsize', 16)

    fontsize_dict = {'label_fontsize': kwargs.get('label_fontsize', fontsize),
                     'title_fontsize': fontsize,
                     'cbar_label_fontsize': fontsize}
    
    for kw, val in kwargs.items():
        if kw in fontsize_dict: fontsize_dict[kw] = val

    start_marker = kwargs.get('start_marker', kwargs.get('marker', 'o'))
    end_marker = kwargs.get('end_marker', kwargs.get('marker', '^'))
    d_kwargs_for_plots = dict(
                                color = 'k',
                                size = 3,
                                alpha = 1,
                                )

    for kw, val in kwargs_for_plots.items():
        if kw == 'c':
            d_kwargs_for_plots['color'] = val
        if kw == 's':
            d_kwargs_for_plots['size'] = val
        elif kw == 'marker':
            pass
        else:
            d_kwargs_for_plots[kw] = val

    if clones is None:
        clones = start_clones_and_phenotypes.clones()
    if phenotypes is None:
        phenotypes = start_clones_and_phenotypes.phenotypes[:3]

    phenotype_names = kwargs.get('phenotype_names', {pheno: pheno for pheno in phenotypes})

    start_prob_pnts_by_clone = {c: ternary_plot_projection(np.array([start_clones_and_phenotypes[c][pheno] for pheno in phenotypes])/np.sum([start_clones_and_phenotypes[c][pheno] for pheno in phenotypes])) for c in clones if np.sum(np.array([start_clones_and_phenotypes[c][pheno] for pheno in phenotypes])) > 0}
    #Eliminate clones that cannot be plotted
    clones = [c for c in clones if c in start_prob_pnts_by_clone]
    
    color_per_clone = [c_dict.get(c, d_kwargs_for_plots['color']) for c in clones]
    size_per_clone = [s_dict.get(c, d_kwargs_for_plots['size']) for c in clones]
    d_kwargs_for_plots.__delitem__('color')
    d_kwargs_for_plots.__delitem__('size')
    if 'ax' in kwargs:
        ax = kwargs['ax']
    else:
        kwargs['figsize'] = kwargs.get('figsize', (5.45, 5))
        fig, ax = setup_ternary_plot(**kwargs)

    if end_clones_and_phenotypes is None:
        ax.scatter([start_prob_pnts_by_clone[c][0] for c in clones],[start_prob_pnts_by_clone[c][1] for c in clones], color=color_per_clone, s = [s**2 for s in size_per_clone], marker = start_marker, **d_kwargs_for_plots)
    else:
        end_prob_pnts_by_clone = {c: ternary_plot_projection(np.array([end_clones_and_phenotypes[c][pheno] for pheno in phenotypes])/np.sum([end_clones_and_phenotypes[c][pheno] for pheno in phenotypes])) for c in clones if np.sum(np.array([end_clones_and_phenotypes[c][pheno] for pheno in phenotypes])) > 0}
        for i, c in enumerate(clones):
            if c in start_prob_pnts_by_clone and c not in end_prob_pnts_by_clone:
                ax.scatter([start_prob_pnts_by_clone[c][0]], [start_prob_pnts_by_clone[c][1]], color=color_per_clone[i], s = size_per_clone[i]**2, marker = start_marker, **d_kwargs_for_plots)
            elif c not in start_prob_pnts_by_clone and c in end_prob_pnts_by_clone:
                ax.scatter([end_prob_pnts_by_clone[c][0]], [end_prob_pnts_by_clone[c][1]], color=color_per_clone[i], s = size_per_clone[i]**2, marker = end_marker, **d_kwargs_for_plots)
            elif c in start_prob_pnts_by_clone and c in end_prob_pnts_by_clone:
                if line_type == 'arrows':
                    ax.arrow(start_prob_pnts_by_clone[c][0], start_prob_pnts_by_clone[c][1],
                              end_prob_pnts_by_clone[c][0] - start_prob_pnts_by_clone[c][0],
                              end_prob_pnts_by_clone[c][1] - start_prob_pnts_by_clone[c][1],
                              length_includes_head = True,
                              width = 0.0034*size_per_clone[i],
                              facecolor=color_per_clone[i],
                              edgecolor = 'None',
                              alpha = 0.5
                             )
                    ax.arrow(start_prob_pnts_by_clone[c][0], start_prob_pnts_by_clone[c][1],
                             end_prob_pnts_by_clone[c][0] - start_prob_pnts_by_clone[c][0],
                             end_prob_pnts_by_clone[c][1] - start_prob_pnts_by_clone[c][1],
                             length_includes_head = True,
                             width = 0.0034*size_per_clone[i],
                             facecolor= 'None',
                             alpha = 1,
                             edgecolor = color_per_clone[i]
                             )
                else:
                    ax.scatter([start_prob_pnts_by_clone[c][0]], [start_prob_pnts_by_clone[c][1]], color=color_per_clone[i],s = size_per_clone[i]**2, marker = start_marker, **d_kwargs_for_plots)
                    ax.scatter([end_prob_pnts_by_clone[c][0]], [end_prob_pnts_by_clone[c][1]], color=color_per_clone[i], s = size_per_clone[i]**2, marker = end_marker, **d_kwargs_for_plots)
                    ax.plot([start_prob_pnts_by_clone[c][0], end_prob_pnts_by_clone[c][0]],[start_prob_pnts_by_clone[c][1], end_prob_pnts_by_clone[c][1]], color=color_per_clone[i], lw = 0.4*size_per_clone[i], **d_kwargs_for_plots)

    set_ternary_corner_label(ax, top_label = phenotype_names[phenotypes[0]], left_label = phenotype_names[phenotypes[1]], right_label = phenotype_names[phenotypes[2]], fontsize = fontsize_dict['label_fontsize'], offset = 0.04)
    set_ternary_axis_label(ax, right_label = 'Prob(%s)'%(phenotype_names[phenotypes[0]]), left_label = 'Prob(%s)'%(phenotype_names[phenotypes[1]]), bottom_label = 'Prob(%s)'%(phenotype_names[phenotypes[2]]), fontsize = 12)
    
    
    if 'title' in kwargs:
        ax.set_title(kwargs['title'], fontsize = fontsize_dict['title_fontsize'])

    if 'ax' not in kwargs:
        fig.tight_layout()

    if 'savefig_filename' in kwargs and 'ax' not in kwargs:
        fig.savefig(kwargs['savefig_filename'])

    if kwargs.get('return_axes', False):
        if 'ax' in kwargs:
            return ax
        else:
            return fig, ax
    else:
        return None


class Phenotypes(object):

    def __init__(self, phenotypes):
        self.phenotypes = [phenotype for phenotype in phenotypes] #account for varying input types
        self.phenotypes_inds = {phenotype: i for i, phenotype in enumerate(self.phenotypes)}
        
    def __repr__(self):
        return "Phenotypes(%s)" %(', '.join(self.phenotypes))
    
    def __len__(self):
        return len(self.phenotypes)

    def __getitem__(self, i):
        return self.phenotypes[i]
    
    def __iter__(self):
        return iter(self.phenotypes)

class Genes(object):
    def __init__(self, genes = []):
        self.genes = list(genes) #account for varying input types
        self.gene_inds = {gene: i for i, gene in enumerate(self.genes)}
        
        #self.gene_inds = kwargs.get('gene_inds', {gene: i for i, gene in enumerate(self.genes)})
        
    def __repr__(self):
        return "Genes(%s)" %(', '.join(self.genes))
    
    def __len__(self):
        return len(self.genes)
    
    def add_genes(self, g):
        if type(g) == str:
            if g not in self.gene_inds:
                self.genes.append(g)
                self.gene_inds[g] = len(self.genes)-1
        else:
            for gene in g:
                self.add_genes(gene)

    def __getitem__(self, x):
        try:
            return self.gene_inds[x]
        except KeyError:
            try:
                return self.genes[x]
            except:
                return KeyError
            
    def __iter__(self):
        return iter(self.genes)
    
class GeneExpr(object):
    
    def __init__(self, genes = [], gene_expr_vec = None, **kwargs):
        self.genes = genes
        if gene_expr_vec is not None:
            self.gene_expr_vec = gene_expr_vec
        else:
            self.gene_expr_vec = np.zeros(len(self.genes))

    def __repr__(self):
        return "GenesExpr(%i total genes, %i expressed genes)" %(len(self.gene_expr_vec), np.sum(self.gene_expr_vec > 0))
    
    def __len__(self):
        return len(self.genes)

    def __getitem__(self, gene):
        return self.gene_expr_vec[self.genes[gene]]
    
    def __iter__(self):
        return iter(self.genes)
    
    def items(self):
        return zip(self.genes.genes, self.gene_expr_vec)

    def keys(self):
        return list(self.genes)

    def values(self):
        return list(self.gene_expr_vec)

    def get(self, gene, default = 0):
        if gene in self.genes.gene_inds:
            return self.gene_expr_vec[self.genes.gene_inds[gene]]
        else:
            return default
        

class PhenotypesAndCounts(object):

    def __init__(self, phenotypes_and_counts = {}, **kwargs):
        phenotypes = kwargs.get('phenotypes', phenotypes_and_counts.keys())
        self.phenotypes = Phenotypes(phenotypes)
        self.phenotypes_and_counts = {phenotype: phenotypes_and_counts.get(phenotype, 0) for phenotype in self.phenotypes}
    
    def __repr__(self):
        return "PhenotypesAndCounts(%s)"%(', '.join(['%s: %s'%(pheno, count) for pheno, count in self.phenotypes_and_counts.items()]))
    
    def __len__(self):
        return len(self.phenotypes_and_counts)
    
    def __add__(self, phenotypes_and_countsB):
        return PhenotypesAndCounts({pheno: self.phenotypes_and_counts[pheno] + phenotypes_and_countsB[pheno] for pheno in self.phenotypes})

    def __getitem__(self, phenotype):
        return self.phenotypes_and_counts.get(phenotype, 0)

    def __setitem__(self, phenotype, count):
        self.phenotypes_and_counts[phenotype] = count

    def __iter__(self):
        return iter(self.phenotypes_and_counts)

    def items(self):
        return self.phenotypes_and_counts.items()

    def keys(self):
        return self.phenotypes_and_counts.keys()

    def values(self):
        return self.phenotypes_and_counts.values()

    def get(self, phenotype, default = 0):
        return self.phenotypes_and_counts.get(phenotype, default)


class Tcell(TcellClone, PhenotypesAndCounts):
    def __init__(self, **kwargs):
        TcellClone.__init__(self, **kwargs)
        PhenotypesAndCounts.__init__(self, **kwargs)
        self.barcode = kwargs.get('barcode', None)
        self.gene_expr = GeneExpr(**kwargs)
        if len(self.phenotypes_and_counts) == 0:
            self.count = kwargs.get('count', 1)
            self.phenotype = None
        else:
            self.phenotype = sorted(self.phenotypes, key = self.phenotypes_and_counts.get, reverse = True)[0]
            self.count = kwargs.get('count', sum(self.phenotypes_and_counts.values()))

    def __repr__(self):
        return 'Tcell(clone: %s, phenotypes: %s)'%(self.clone_rep(), str(self.phenotypes_and_counts))


class ClonesAndPhenotypes(Phenotypes):
    """Base class for joint distribution between clones and phenotypes.
    """

    def __init__(self, clones_and_phenos = {}, phenotypes = []):
        
        if len(phenotypes) == 0 and len(clones_and_phenos) > 0: 
            phenotypes = list(clones_and_phenos.values())[0].keys()
        Phenotypes.__init__(self, phenotypes)
        self.clones_and_phenos = {clone: PhenotypesAndCounts(phenotypes_and_counts = phenotypes_and_counts, phenotypes = self.phenotypes) for clone, phenotypes_and_counts in clones_and_phenos.items()}
        self.norm = sum(sum(self.clones_and_phenos.values(), PhenotypesAndCounts(phenotypes = phenotypes)).values())
    
    def __repr__(self):
        if len(self) > 100:
            return 'ClonesAndPhenotypes({%s%s%s})'%(', '.join(['%s: %s'%(clone,str(self.clones_and_phenos[clone])) for clone in self.clones[:20]]), '\n...\n', ', '.join(['%s: %s'%(clone,str(self.clones_and_phenos[clone])) for clone in self.clones[-20:]]))
        else:
            return 'ClonesAndPhenotypes(%s)'%(self.clones_and_phenos.__repr__())
    
    def __len__(self):
        return len(self.clones_and_phenos)

    def __getitem__(self, clone_or_pheno):
        if clone_or_pheno in self.phenotypes:
            return ClonesAndCounts({clone: pheno_counts[clone_or_pheno] for clone, pheno_counts in self.clones_and_phenos.items()})
        else:
            return self.clones_and_phenos.get(clone_or_pheno, PhenotypesAndCounts(phenotypes = self.phenotypes))

    def __setitem__(self, clone, pheno_counts):
        self.norm += sum(pheno_counts.values()) - sum(self.clones_and_phenos.get(clone, {'': 0}).values())
        self.clones_and_phenos[clone] = pheno_counts

    def __delitem__(self, clone):
        self.norm -= sum(self.clones_and_phenos.get(clone, {'': 0}).values())
        del self.clones_and_phenos[clone]

    def __iter__(self):
        return iter(self.clones_and_phenos)

    def items(self):
        return self.clones_and_phenos.items()

    def keys(self):
        return self.clones_and_phenos.keys()

    def values(self):
        return self.clones_and_phenos.values()

    def get(self, clone_or_pheno, default = None):
        if clone_or_pheno in self.phenotypes:
            return ClonesAndCounts({clone: pheno_counts[self.phenotypes_dict[clone_or_pheno]] for clone, pheno_counts in self.clones_and_phenos.items()})
        else:
            return self.clones_and_phenos.get(clone_or_pheno, default)

    def get_phenotype_counts(self, clone, phenotypes):
        c_counts = self.clones_and_phenos[clone]
        return np.array([c_counts[phenotype] for phenotype in phenotypes])
    
    def clones(self):
        return list(self.clones_and_phenos.keys())

    def clone_intersection(self, clones):
        try:
            clones = set(clones.keys())
        except AttributeError:
            clones = set(clones)

        return list(clones.intersection(self.clones_and_phenos))

    def clone_union(self, clones):
        try:
            clones = set(clones.keys())
        except AttributeError:
            clones = set(clones)

        return list(clones.union(self.clones_and_phenos))

class CellRepertoire(CloneDefinition, ClonesAndPhenotypes, TCRClonePvalue):

    def __init__(self, **kwargs):

        CloneDefinition.__init__(self, **kwargs)
        ClonesAndPhenotypes.__init__(self, **{kw: kw_val for kw, kw_val in kwargs.items() if kw in ['clones_and_phenos', 'phenotypes']})
        TCRClonePvalue.__init__(self)

        self.name = ''
        self.filenames = []
        self.cell_list = []
        self.pvalues = {}
        
        for kw, kw_val in kwargs.items():
            if kw in self.__dict__:
                self.__dict__[kw] = kw_val
            elif 'filename' in kw:
                if 'adaptive' in kw:
                    self.load_adaptive_file(kw_val)
                elif '10x_clonotype' in kw:
                    self.load_10x_clonotypes(kw_val)
        
        if len(self.cell_list) > 0:
            self._set_consistency()
    
    def __repr__(self):
        return 'CellRepertoire'
    
    def _set_consistency(self):
        self.clones_and_phenos = self.get_clones_and_phenos()
        self.clones = self.clones_and_phenos.clones()
        self.joint_distribution = self.get_clones_and_phenos_joint_distribution()
        self.norm = len(self.cell_list)
    
    def get_clones_and_phenos(self, **kwargs):
        c_clone_def = self.get_clone_def()
        for kw, kw_val in kwargs.items():
            if kw in c_clone_def: c_clone_def[kw] = kw_val
        clones_and_phenos = ClonesAndPhenotypes(phenotypes = self.phenotypes)
        for cell in self.cell_list:
            clones_and_phenos[cell.clone_rep(**c_clone_def)] += cell
        return clones_and_phenos
        
    def get_clones_and_phenos_joint_distribution(self, **kwargs):
        clones_and_phenos = self.get_clones_and_phenos(**kwargs)
        joint_dist = np.array([[clones_and_phenos[clone][pheno] for pheno in self.phenotypes] for clone in self.clones])
        return joint_dist/np.sum(joint_dist)
        
    # def get_clones_and_counts(self, clones = None, phenotypes = None, **kwargs):
    #     c_clone_def = self.get_clone_def()
    #     for kw, kw_val in kwargs.items():
    #         if kw in c_clone_def: c_clone_def[kw] = kw_val
    #     clone_and_count_dict = ClonesAndCounts()
    #     if clones is None:
    #         for cell in self.cell_list:
    #             clone_and_count_dict[cell.clone_rep(**c_clone_def)] += cell.count
    #     else:
    #         clones = set(clones)
    #         c_norm = 0
    #         for clone in self.cell_list:
    #             c_clone_rep = clone.clone_rep(**c_clone_def)
    #             c_norm += clone.count
    #             if c_clone_rep in clones:
    #                 clone_and_count_dict[c_clone_rep] += clone.count
    #         clone_and_count_dict.set_norm(norm = c_norm)
    #     return clone_and_count_dict

    def get_phenotypic_flux(self, CellRepertoireB, phenotypes = None, min_cell_count = 0):
        
        phenotypic_flux = ClonesAndData()
        if phenotypes is None:
            phenotypes = self.phenotypes
        for clone in self.clones:
            c_phenosA = np.array([self.clones_and_phenos[clone][pheno] for pheno in phenotypes])
            c_phenosB = np.array([CellRepertoireB[clone][pheno] for pheno in phenotypes])
            if sum(c_phenosA) > min_cell_count and sum(c_phenosB) > min_cell_count:
                phenotypic_flux[clone] = np.sum(np.abs(c_phenosA/np.sum(c_phenosA) - c_phenosB/np.sum(c_phenosB)))
        
        return phenotypic_flux

    def get_significant_clones(self, name, pval_thresh):
#        if name not in self.pvalues:
#            InputError('name', name)
        return self.pvalues[name].get_significant_clones(pval_thresh)

    def load_10x_data(self, infile_names, ntseq_index = 4, count_index = 1, id_index = 0):
        if type(infile_names) is str:
            infile_names = [infile_names]
        for infile_name in infile_names:
            #print('Loading 10x clonotypes file: %s'%(infile_name))
            with open(infile_name, 'r') as infile:
                all_L = [l.split(',') for l in infile.read().strip().split('\n') if len(l) > 0]
            self.filenames.append(infile_name)

            for i, l in enumerate(all_L[1:]):
                try:
                    c_seqs = l[ntseq_index].split(';')

                    c_clone = TcellClone(count = float(l[count_index]))
                    for c_seq in c_seqs:
                        c_chain, c_ntseq = tuple(c_seq.split(':'))
                        if c_chain == 'TRA':
                            c_clone.load_TRA(ntseq = c_ntseq)
                        elif c_chain == 'TRB':
                            c_clone.load_TRB(ntseq = c_ntseq)

                    self.full_clone_list.append(c_clone)
                except:
                    print(i, l)

        self.set_clones_and_counts_attr()


def draw_clone_bars(data_dict, dict_order=None, ll=0.5, bk_th=0.0008, save_name=None, hatched=False, title=None, create_new = True):
    """
    Created on Wed Jul 26 13:04:54 2023
    @author: elhanaty
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    # data_dict is a dict of dict, first level is samples and second is clones, value is fractional size: data_dict[samp1][clone]=0.0001
    # ll is lower limit for y axis (fraction of sequences)
    # bk_th is the threshold for clone size that would be part of black bar
    # save_name is the name of the file to save the plot, None would not save
    # hatched - if True it would use twice the same colors, but hatch them the second time, so you get twice the number of clones colored
    n_large_clones = 12 # how many different clones will be colored
    clrs = sns.color_palette('Paired', n_colors=n_large_clones) #palette for the clones
    if hatched: #if using hatching, add all colors twice
        clrs = clrs[:-1]*2
        n_large_clones *= 2
    clrs = clrs + [(1,1,1)] #add one color more color for non colored clones (white)
    all_large_clones_dict = {} #dict for the largest clones over all samples, values would be fractional size
    for sample in data_dict:
        all_large_clones_dict = {c: max(all_large_clones_dict.get(c,0), data_dict[sample].get(c,0))
                                 for c in set(sorted(data_dict[sample],key=data_dict[sample].get,)[-n_large_clones:]
                                              + list(all_large_clones_dict.keys()))}
    all_large_clones = {c: i for i,c in enumerate(sorted(all_large_clones_dict,key=all_large_clones_dict.get,)[-n_large_clones:])}
    # this dict has the largest clones found, but value now is index of color to be used in clrs list
    if create_new:
        fig = plt.figure(figsize=(20,10))
    if not dict_order:
        dict_order = data_dict.keys()
    for j,samp in enumerate(dict_order):
        b = 0 # bottom of current bar
        t = 0 # top of current bar
        for c in sorted(data_dict[samp], key=data_dict[samp].get,): #loop over all clones in sample from small to large
            if b == 0 and data_dict[samp].get(c,0) < bk_th: #as long as clones are smaller than threshold
                t += data_dict[samp].get(c,0) #add them together to top
            else:
                if b == 0:
                    plt.bar(j, t, color = 'black', edgecolor='black') # use one black bar for all clones below threshold
                    b = t #advance bottom to where top was
                color_index = all_large_clones.get(c, n_large_clones) # get correct color for next clone
                plt.bar(j, data_dict[samp].get(c, 0), bottom=b, color=clrs[color_index],
                        edgecolor='black', hatch='x' if hatched and n_large_clones / 2 <= color_index < n_large_clones else '')
                b += data_dict[samp].get(c,0)
    samp_labels = ['\n'.join(str(x).split('_')) for x in dict_order]
    plt.xticks(range(len(samp_labels)), samp_labels, rotation=0)
    plt.ylabel('fraction of all sequences')
    plt.ylim([ll,1])
    if title:
        plt.title(title)
    if save_name:
        plt.savefig(save_name)
    #plt.show()