"""
hrp.py
Module for Hierarchical Risk Parity (HRP) portfolio allocation.
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

def correl_dist(corr):
    """Convert correlation matrix to distance matrix."""
    dist = np.sqrt(0.5 * (1 - corr))
    return dist

def get_quasi_diag(link):
    """Sort clustered items by distance (quasi-diagonalization)."""
    link = link.astype(int)
    sort_ix = list([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while any([i >= num_items for i in sort_ix]):
        new_sort_ix = []
        for i in sort_ix:
            if i >= num_items:
                k = i - num_items
                new_sort_ix += [link[k, 0], link[k, 1]]
            else:
                new_sort_ix.append(i)
        sort_ix = new_sort_ix
    return np.array(sort_ix, dtype=int)

def get_cluster_var(cov, cluster_items):
    """Compute the variance per cluster."""
    cov_ = cov.loc[cluster_items, cluster_items]
    w_ = get_ivp(cov_).reshape(-1, 1)
    var = np.dot(np.dot(w_.T, cov_), w_)
    return float(var.item())

def get_ivp(cov):
    """Compute inverse-variance portfolio weights."""
    ivp = 1. / np.diag(cov)
    ivp /= ivp.sum()
    return ivp

def hrp_allocation(cov, corr):
    """Compute HRP weights given covariance and correlation matrices."""
    dist = correl_dist(corr)
    link = linkage(squareform(dist), method='ward')
    sort_ix = get_quasi_diag(link)
    items = list(cov.index)
    sorted_items = [items[i] for i in sort_ix]
    weights = pd.Series(1.0, index=sorted_items)
    clusters = [sorted_items]
    while len(clusters) > 0:
        clusters_ = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            split = int(len(cluster) / 2)
            c1 = cluster[:split]
            c2 = cluster[split:]
            var1 = get_cluster_var(cov, c1)
            var2 = get_cluster_var(cov, c2)
            alpha = 1 - var1 / (var1 + var2)
            weights[c1] *= alpha
            weights[c2] *= 1 - alpha
            clusters_ += [c1, c2]
        clusters = clusters_
    return weights / weights.sum()
