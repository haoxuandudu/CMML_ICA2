# %%
"""
CMML3 Mini-project 2: MultiVI benchmark on annotated 10x PBMC Multiome data.

Run as a percent-cell notebook in VS Code/Jupyter, or as:
    python notebooks/01_multivi_benchmark.py

Outputs:
    figures/figure1_umap_methods.png
    figures/figure2_metrics.png
    tables/method_metrics.csv
    results/report_values.json
"""

# %%
from __future__ import annotations

import json
import math
import os
import random
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

# Heavy single-cell imports are intentionally grouped here so missing
# dependencies fail early with a clear message.
import anndata as ad
import mudata as md
import muon as mu
import scanpy as sc
import scvi
import umap


# %%
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
FIGURE_DIR = PROJECT_DIR / "figures"
TABLE_DIR = PROJECT_DIR / "tables"
RESULT_DIR = PROJECT_DIR / "results"

for folder in [DATA_DIR, FIGURE_DIR, TABLE_DIR, RESULT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

SEED = 7
N_CELLS = int(os.environ.get("CMML3_N_CELLS", 5000))
N_GENES = int(os.environ.get("CMML3_N_GENES", 3000))
N_PEAKS = int(os.environ.get("CMML3_N_PEAKS", 5000))
N_COMPONENTS = int(os.environ.get("CMML3_N_COMPONENTS", 30))
MULTIVI_EPOCHS = int(os.environ.get("CMML3_MULTIVI_EPOCHS", 100))
MIN_CELLS_PER_TYPE = int(os.environ.get("CMML3_MIN_CELLS_PER_TYPE", 10))
MOFA_ITER = int(os.environ.get("CMML3_MOFA_ITER", 1000))
COBOLT_EPOCHS = int(os.environ.get("CMML3_COBOLT_EPOCHS", 100))
COBOLT_BATCH_SIZE = int(os.environ.get("CMML3_COBOLT_BATCH_SIZE", 128))
SPLIT_GROUPS = ("rna_only", "paired", "atac_only")
MARKER_GENES = {
    "Monocytes": ["LYZ", "S100A8", "S100A9"],
    "B cells": ["MS4A1", "CD79A"],
    "NK cells": ["NKG7", "GNLY"],
    "T cells": ["CD3D", "CD3E", "IL7R"],
}

random.seed(SEED)
np.random.seed(SEED)
scvi.settings.seed = SEED
sns.set_theme(style="white", context="notebook")

RNA_URL = "https://zenodo.org/records/19581816/files/10x-Multiome-Pbmc10k-RNA.h5ad?download=1"
ATAC_URL = "https://zenodo.org/records/19581816/files/10x-Multiome-Pbmc10k-ATAC.h5ad?download=1"
RNA_PATH = DATA_DIR / "10x-Multiome-Pbmc10k-RNA.h5ad"
ATAC_PATH = DATA_DIR / "10x-Multiome-Pbmc10k-ATAC.h5ad"


# %%
def download_if_missing(url: str, path: Path) -> None:
    if path.exists():
        print(f"Found {path.name}")
        return
    print(f"Downloading {path.name} ...")
    urllib.request.urlretrieve(url, path)
    print(f"Saved {path}")


download_if_missing(RNA_URL, RNA_PATH)
download_if_missing(ATAC_URL, ATAC_PATH)


# %%
rna = sc.read_h5ad(RNA_PATH)
atac = sc.read_h5ad(ATAC_PATH)

common = rna.obs_names.intersection(atac.obs_names)
rna = rna[common].copy()
atac = atac[common].copy()

if "cell_type" not in rna.obs:
    raise KeyError("Expected rna.obs['cell_type'] from the annotated Zenodo object.")

valid = rna.obs["cell_type"].notna().to_numpy()
rna = rna[valid].copy()
atac = atac[rna.obs_names].copy()

if rna.n_obs > N_CELLS:
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(rna.n_obs, size=N_CELLS, replace=False)
    chosen.sort()
    rna = rna[chosen].copy()
    atac = atac[rna.obs_names].copy()

cell_type_counts = rna.obs["cell_type"].value_counts()
keep_cell_types = cell_type_counts[cell_type_counts >= MIN_CELLS_PER_TYPE].index
if len(keep_cell_types) < len(cell_type_counts):
    dropped = cell_type_counts[cell_type_counts < MIN_CELLS_PER_TYPE]
    print(
        "Dropping rare cell types with fewer than "
        f"{MIN_CELLS_PER_TYPE} cells: {dropped.to_dict()}"
    )
    keep_cells = rna.obs["cell_type"].isin(keep_cell_types).to_numpy()
    rna = rna[keep_cells].copy()
    atac = atac[rna.obs_names].copy()

print(rna)
print(atac)
print(rna.obs["cell_type"].value_counts())
rna_marker_source = rna.copy()


# %%
def ensure_csr(x):
    return x.tocsr() if sparse.issparse(x) else sparse.csr_matrix(x)


rna.X = ensure_csr(rna.X)
atac.X = ensure_csr(atac.X)
rna.layers["counts"] = rna.X.copy()
atac.layers["counts"] = atac.X.copy()


# %%
def select_hv_genes(adata: ad.AnnData, n_top: int) -> ad.AnnData:
    work = adata.copy()
    sc.pp.filter_genes(work, min_cells=max(10, int(0.002 * work.n_obs)))
    sc.pp.normalize_total(work, target_sum=1e4)
    sc.pp.log1p(work)
    sc.pp.highly_variable_genes(work, n_top_genes=min(n_top, work.n_vars), flavor="cell_ranger")
    selected = work.var_names[work.var["highly_variable"]]
    out = adata[:, selected].copy()
    out.X = ensure_csr(out.layers["counts"] if "counts" in out.layers else out.X)
    return out


def select_common_peaks(adata: ad.AnnData, n_top: int) -> ad.AnnData:
    x = ensure_csr(adata.X)
    detected = np.asarray((x > 0).sum(axis=0)).ravel()
    keep = detected >= max(10, int(0.002 * adata.n_obs))
    idx = np.where(keep)[0]
    if idx.size > n_top:
        idx = idx[np.argsort(detected[idx])[-n_top:]]
    idx = np.sort(idx)
    out = adata[:, idx].copy()
    out.X = (ensure_csr(out.X) > 0).astype(np.float32).tocsr()
    return out


rna = select_hv_genes(rna, N_GENES)
atac = select_common_peaks(atac, N_PEAKS)
print(f"Selected {rna.n_vars} genes and {atac.n_vars} peaks")


# %%
def make_missing_modality_benchmark(rna_in: ad.AnnData, atac_in: ad.AnnData) -> md.MuData:
    rng = np.random.default_rng(SEED)
    labels = rna_in.obs["cell_type"].astype(str).to_numpy()
    split_indices = {group: [] for group in SPLIT_GROUPS}

    for cell_type in sorted(np.unique(labels)):
        cell_type_idx = np.where(labels == cell_type)[0]
        shuffled = rng.permutation(cell_type_idx)
        chunks = np.array_split(shuffled, len(SPLIT_GROUPS))
        for group, chunk in zip(SPLIT_GROUPS, chunks):
            split_indices[group].extend(chunk.tolist())

    rna_idx = np.asarray(split_indices["rna_only"], dtype=int)
    paired_idx = np.asarray(split_indices["paired"], dtype=int)
    atac_idx = np.asarray(split_indices["atac_only"], dtype=int)

    row_indices = np.concatenate([rna_idx, paired_idx, atac_idx])
    modality = (
        ["rna_only"] * len(rna_idx)
        + ["paired"] * len(paired_idx)
        + ["atac_only"] * len(atac_idx)
    )
    source_cells = rna_in.obs_names[row_indices].to_numpy()
    obs_names = [f"{cell}__{mod}" for cell, mod in zip(source_cells, modality)]

    obs = pd.DataFrame(
        {
            "source_cell": source_cells,
            "modality": modality,
            "cell_type": rna_in.obs.loc[source_cells, "cell_type"].astype(str).to_numpy(),
            "split_design": "stratified_equal_thirds_by_cell_type",
        },
        index=obs_names,
    )

    zero_rna = sparse.csr_matrix((len(atac_idx), rna_in.n_vars), dtype=np.float32)
    zero_atac = sparse.csr_matrix((len(rna_idx), atac_in.n_vars), dtype=np.float32)

    rna_x = sparse.vstack(
        [rna_in.X[rna_idx], rna_in.X[paired_idx], zero_rna],
        format="csr",
    )
    atac_x = sparse.vstack(
        [zero_atac, atac_in.X[paired_idx], atac_in.X[atac_idx]],
        format="csr",
    )

    rna_mod = ad.AnnData(X=rna_x, obs=obs.copy(), var=rna_in.var.copy())
    atac_mod = ad.AnnData(X=atac_x, obs=obs.copy(), var=atac_in.var.copy())
    mdata_out = md.MuData({"rna": rna_mod, "atac": atac_mod})
    mdata_out.obs = obs.copy()
    return mdata_out


mdata = make_missing_modality_benchmark(rna, atac)
print(mdata)
print(mdata.obs[["modality", "cell_type"]].value_counts().head())
split_by_cell_type = (
    mdata.obs.groupby(["cell_type", "modality"], observed=True)
    .size()
    .unstack(fill_value=0)
    .reindex(columns=list(SPLIT_GROUPS), fill_value=0)
)
split_by_cell_type.to_csv(TABLE_DIR / "split_by_cell_type.csv")


# %%
def log_norm_sparse(x: sparse.csr_matrix, target_sum: float = 1e4) -> sparse.csr_matrix:
    x = ensure_csr(x).astype(np.float32)
    row_sums = np.asarray(x.sum(axis=1)).ravel()
    scale = np.divide(target_sum, row_sums, out=np.zeros_like(row_sums, dtype=np.float32), where=row_sums > 0)
    out = x.multiply(scale[:, None]).tocsr()
    out.data = np.log1p(out.data)
    return out


def tfidf_lsi(x: sparse.csr_matrix, n_components: int) -> np.ndarray:
    x = ensure_csr(x).astype(np.float32)
    row_sums = np.asarray(x.sum(axis=1)).ravel()
    tf = x.multiply(np.divide(1.0, row_sums, out=np.zeros_like(row_sums, dtype=np.float32), where=row_sums > 0)[:, None])
    df = np.asarray((x > 0).sum(axis=0)).ravel()
    idf = np.log1p(x.shape[0] / (1.0 + df)).astype(np.float32)
    tfidf = tf.multiply(idf).tocsr()
    svd = TruncatedSVD(n_components=n_components, random_state=SEED)
    return svd.fit_transform(tfidf)


def save_marker_validation(adata: ad.AnnData) -> None:
    marker_to_var = {}
    var_names = pd.Index(adata.var_names.astype(str))
    gene_names = (
        adata.var["gene_name"].astype(str)
        if "gene_name" in adata.var
        else pd.Series(var_names, index=var_names)
    )

    for marker in [gene for genes in MARKER_GENES.values() for gene in genes]:
        if marker in var_names:
            marker_to_var[marker] = marker
            continue
        matches = gene_names.index[gene_names.to_numpy() == marker]
        if len(matches):
            marker_to_var[marker] = matches[0]

    if not marker_to_var:
        print("No configured marker genes were found; skipping marker validation.")
        return

    marker_names = list(marker_to_var.keys())
    var_ids = [marker_to_var[marker] for marker in marker_names]
    marker_adata = adata[:, var_ids].copy()
    marker_expr = pd.DataFrame(
        log_norm_sparse(marker_adata.X).toarray(),
        index=marker_adata.obs_names,
        columns=marker_names,
    )
    marker_expr["cell_type"] = adata.obs["cell_type"].astype(str).to_numpy()
    marker_means = marker_expr.groupby("cell_type").mean()
    marker_means.to_csv(TABLE_DIR / "marker_gene_means.csv")

    scaled = marker_means.copy()
    scaled = (scaled - scaled.mean(axis=0)) / scaled.std(axis=0).replace(0, np.nan)
    scaled = scaled.fillna(0)
    fig_height = max(4, 0.28 * scaled.shape[0])
    fig_width = max(7, 0.45 * scaled.shape[1])
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    sns.heatmap(scaled, cmap="vlag", center=0, linewidths=0.2, ax=ax)
    ax.set_title("PBMC marker expression by transferred cell type")
    ax.set_xlabel("Marker gene")
    ax.set_ylabel("Cell type")
    fig.savefig(FIGURE_DIR / "supp_marker_expression.png", dpi=300)
    plt.close(fig)


save_marker_validation(rna_marker_source)


def run_rna_pca_baseline(mdata_in: md.MuData) -> np.ndarray:
    rna_log = log_norm_sparse(mdata_in.mod["rna"].X)
    rna_pca = TruncatedSVD(n_components=N_COMPONENTS, random_state=SEED).fit_transform(rna_log)
    return StandardScaler().fit_transform(rna_pca)


def run_mofa_plus(mdata_in: md.MuData) -> np.ndarray:
    rna_log = log_norm_sparse(mdata_in.mod["rna"].X)
    atac_lsi_like = tfidf_lsi(mdata_in.mod["atac"].X, min(N_COMPONENTS * 2, 50))

    # MOFA+ accepts multi-view continuous matrices. We keep RNA as log-normalized
    # HVGs and use ATAC LSI components as a compact accessibility view to keep
    # memory and runtime stable for an in-course benchmark.
    mofa_rna = ad.AnnData(
        X=rna_log.toarray().astype(np.float32),
        obs=mdata_in.obs.copy(),
        var=mdata_in.mod["rna"].var.copy(),
    )
    mofa_atac = ad.AnnData(
        X=atac_lsi_like.astype(np.float32),
        obs=mdata_in.obs.copy(),
        var=pd.DataFrame(
            index=[f"ATAC_LSI_{i + 1}" for i in range(atac_lsi_like.shape[1])]
        ),
    )
    mofa_data = md.MuData({"rna": mofa_rna, "atac_lsi": mofa_atac})
    mofa_data.obs = mdata_in.obs.copy()

    outfile = RESULT_DIR / "mofa_model.hdf5"
    mu.tl.mofa(
        mofa_data,
        n_factors=N_COMPONENTS,
        convergence_mode="fast",
        seed=SEED,
        outfile=str(outfile),
        n_iterations=MOFA_ITER,
    )
    if "X_mofa" not in mofa_data.obsm:
        raise KeyError("MOFA+ did not create mdata.obsm['X_mofa'].")
    return np.asarray(mofa_data.obsm["X_mofa"])


def run_cobolt(mdata_in: md.MuData) -> np.ndarray:
    try:
        from cobolt.model import Cobolt
        from cobolt.utils import MultiomicDataset, SingleData
    except ImportError as exc:
        raise ImportError(
            "Cobolt is required for this benchmark. Install it with "
            "`pip install git+https://github.com/epurdom/cobolt.git`."
        ) from exc

    single_data = []
    obs = mdata_in.obs.copy()

    for modality_group in ["rna_only", "paired", "atac_only"]:
        row_idx = np.where(obs["modality"].to_numpy() == modality_group)[0]
        if row_idx.size == 0:
            continue
        barcodes = obs.index[row_idx].astype(str).to_numpy()

        if modality_group in {"rna_only", "paired"}:
            single_data.append(
                SingleData(
                    feature_name="rna",
                    dataset_name=modality_group,
                    feature=mdata_in.mod["rna"].var_names.astype(str).to_numpy(),
                    count=ensure_csr(mdata_in.mod["rna"].X[row_idx]).astype(float),
                    barcode=barcodes,
                )
            )

        if modality_group in {"paired", "atac_only"}:
            single_data.append(
                SingleData(
                    feature_name="atac",
                    dataset_name=modality_group,
                    feature=mdata_in.mod["atac"].var_names.astype(str).to_numpy(),
                    count=ensure_csr(mdata_in.mod["atac"].X[row_idx]).astype(float),
                    barcode=barcodes,
                )
            )

    dataset = MultiomicDataset.from_singledata(*single_data)
    model = Cobolt(
        dataset=dataset,
        n_latent=N_COMPONENTS,
        batch_size=COBOLT_BATCH_SIZE,
    )
    model.train(num_epochs=COBOLT_EPOCHS)
    latent, cobolt_barcodes = model.get_all_latent(correction=True)

    # Cobolt prefixes barcodes with dataset names, e.g. "paired~cell_id".
    plain_barcodes = pd.Index([str(x).split("~", 1)[1] for x in cobolt_barcodes])
    latent_df = pd.DataFrame(latent, index=plain_barcodes)
    missing = mdata_in.obs_names.difference(latent_df.index)
    if len(missing):
        raise ValueError(f"Cobolt latent space is missing {len(missing)} observations.")
    return latent_df.loc[mdata_in.obs_names].to_numpy()


embeddings: dict[str, np.ndarray] = {}
method_runtime_seconds: dict[str, float] = {}


def run_timed(name: str, func, *args, **kwargs) -> np.ndarray:
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    method_runtime_seconds[name] = elapsed
    print(f"{name} runtime: {elapsed:.1f} seconds")
    return result


embeddings["MOFA_plus"] = run_timed("MOFA_plus", run_mofa_plus, mdata)
embeddings["Cobolt"] = run_timed("Cobolt", run_cobolt, mdata)
embeddings["RNA_PCA_baseline"] = run_timed("RNA_PCA_baseline", run_rna_pca_baseline, mdata)


# %%
def run_multivi(mdata_in: md.MuData) -> np.ndarray:
    mdata_mv = mdata_in.copy()
    mdata_mv.mod["rna"].X = ensure_csr(mdata_mv.mod["rna"].X)
    mdata_mv.mod["atac"].X = ensure_csr(mdata_mv.mod["atac"].X)
    mdata_mv.update()

    scvi.model.MULTIVI.setup_mudata(
        mdata_mv,
        modalities={
            "rna_layer": "rna",
            "atac_layer": "atac",
        },
    )
    model = scvi.model.MULTIVI(
        mdata_mv,
        n_genes=mdata_mv.mod["rna"].n_vars,
    )
    model.train(max_epochs=MULTIVI_EPOCHS, early_stopping=True)
    return model.get_latent_representation()


embeddings["MultiVI"] = run_timed("MultiVI", run_multivi, mdata)
plot_order = ["MultiVI", "MOFA_plus", "Cobolt"]
plot_embeddings = {name: embeddings[name] for name in plot_order if name in embeddings}


# %%
def knn_entropy(x: np.ndarray, categories: pd.Series, k: int = 30) -> float:
    categories = categories.astype("category")
    codes = categories.cat.codes.to_numpy()
    n_categories = len(categories.cat.categories)
    if n_categories < 2:
        return float("nan")
    k_eff = min(k + 1, x.shape[0])
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean").fit(x)
    indices = nn.kneighbors(return_distance=False)[:, 1:]
    entropies = []
    for row in indices:
        counts = np.bincount(codes[row], minlength=n_categories).astype(float)
        probs = counts[counts > 0] / counts.sum()
        entropies.append(-(probs * np.log(probs)).sum() / math.log(n_categories))
    return float(np.mean(entropies))


def safe_silhouette(x: np.ndarray, labels: pd.Series) -> float:
    labels = labels.astype(str).to_numpy()
    if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= len(labels):
        return float("nan")
    if x.shape[0] > 5000:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(x.shape[0], size=5000, replace=False)
        return float(silhouette_score(x[idx], labels[idx]))
    return float(silhouette_score(x, labels))


def evaluate_embedding(name: str, x: np.ndarray, obs: pd.DataFrame) -> dict[str, float | str]:
    labels = obs["cell_type"].astype(str)
    n_clusters = labels.nunique()
    clusters = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=20).fit_predict(x)

    asw_celltype = safe_silhouette(x, labels)
    asw_batch = safe_silhouette(x, obs["modality"])
    modality_entropy = knn_entropy(x, obs["modality"])
    label_entropy = knn_entropy(x, labels)

    return {
        "method": name,
        "ASW_celltype": asw_celltype,
        "ASW_celltype_rescaled": (asw_celltype + 1.0) / 2.0 if not np.isnan(asw_celltype) else np.nan,
        "ASW_batch": asw_batch,
        "ASW_batch_mixing_score": 1.0 - abs(asw_batch) if not np.isnan(asw_batch) else np.nan,
        "modality_neighbor_entropy_higher_is_better": modality_entropy,
        "cell_type_neighbor_entropy_lower_is_better": label_entropy,
        "ARI_kmeans_vs_cell_type": adjusted_rand_score(labels, clusters),
        "NMI_kmeans_vs_cell_type": normalized_mutual_info_score(labels, clusters),
    }


metric_records = []
for name, emb in embeddings.items():
    record = evaluate_embedding(name, emb, mdata.obs)
    record["runtime_seconds"] = method_runtime_seconds.get(name, np.nan)
    metric_records.append(record)

metrics = pd.DataFrame(metric_records)

score_cols = [
    "ASW_celltype_rescaled",
    "ASW_batch_mixing_score",
    "modality_neighbor_entropy_higher_is_better",
    "ARI_kmeans_vs_cell_type",
    "NMI_kmeans_vs_cell_type",
]
metrics["overall_mean_score"] = metrics[score_cols].mean(axis=1)
metrics = metrics.sort_values("overall_mean_score", ascending=False)
metrics.to_csv(TABLE_DIR / "method_metrics.csv", index=False)
metrics


# %%
def compute_umap(x: np.ndarray) -> np.ndarray:
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.25,
        metric="euclidean",
        random_state=SEED,
    )
    return reducer.fit_transform(x)


umaps = {name: compute_umap(emb) for name, emb in plot_embeddings.items()}


# %%
palette_cell = dict(
    zip(
        sorted(mdata.obs["cell_type"].astype(str).unique()),
        sns.color_palette("tab20", n_colors=mdata.obs["cell_type"].nunique()),
    )
)
palette_modality = {
    "rna_only": "#2C7FB8",
    "paired": "#41AB5D",
    "atac_only": "#F16913",
}

fig, axes = plt.subplots(len(umaps), 2, figsize=(10, 4 * len(umaps)), constrained_layout=True)
if len(umaps) == 1:
    axes = np.array([axes])

plot_obs = mdata.obs.copy()
for row, (name, coords) in enumerate(umaps.items()):
    df = plot_obs.assign(UMAP1=coords[:, 0], UMAP2=coords[:, 1])
    sns.scatterplot(
        data=df,
        x="UMAP1",
        y="UMAP2",
        hue="cell_type",
        palette=palette_cell,
        s=5,
        linewidth=0,
        ax=axes[row, 0],
        legend=False,
    )
    axes[row, 0].set_title(f"{name}: cell type")
    axes[row, 0].set_xticks([])
    axes[row, 0].set_yticks([])

    sns.scatterplot(
        data=df,
        x="UMAP1",
        y="UMAP2",
        hue="modality",
        palette=palette_modality,
        s=5,
        linewidth=0,
        ax=axes[row, 1],
        legend=(row == 0),
    )
    axes[row, 1].set_title(f"{name}: modality")
    axes[row, 1].set_xticks([])
    axes[row, 1].set_yticks([])

fig.savefig(FIGURE_DIR / "figure1_umap_methods.png", dpi=300)
plt.show()


# %%
metric_plot = metrics[metrics["method"].isin(plot_order)].set_index("method")[score_cols]
fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
sns.heatmap(
    metric_plot,
    annot=True,
    fmt=".2f",
    cmap="viridis",
    vmin=0,
    vmax=1,
    linewidths=0.5,
    ax=ax,
)
ax.set_title("Integration benchmark metrics")
ax.set_xlabel("")
ax.set_ylabel("")
fig.savefig(FIGURE_DIR / "figure2_metrics.png", dpi=300)
plt.show()


# %%
best = metrics.iloc[0]
cell_counts = mdata.obs["cell_type"].value_counts()
report_values = {
    "n_cells_original_subset": int(rna.n_obs),
    "n_observations_after_missingness_split": int(mdata.n_obs),
    "n_genes": int(rna.n_vars),
    "n_peaks": int(atac.n_vars),
    "n_cell_types": int(mdata.obs["cell_type"].nunique()),
    "largest_cell_type": str(cell_counts.index[0]),
    "rarest_cell_type": str(cell_counts.index[-1]),
    "split_design": "stratified_equal_thirds_by_cell_type",
    "split_group_counts": {k: int(v) for k, v in mdata.obs["modality"].value_counts().to_dict().items()},
    "methods": metrics["method"].tolist(),
    "best_method": str(best["method"]),
    "best_overall_mean_score": float(best["overall_mean_score"]),
    "best_ASW_celltype": float(best["ASW_celltype"]),
    "best_ASW_batch": float(best["ASW_batch"]),
    "best_modality_neighbor_entropy": float(best["modality_neighbor_entropy_higher_is_better"]),
    "best_ARI": float(best["ARI_kmeans_vs_cell_type"]),
    "best_NMI": float(best["NMI_kmeans_vs_cell_type"]),
    "best_runtime_seconds": float(best["runtime_seconds"]),
    "metrics_table": str(TABLE_DIR / "method_metrics.csv"),
    "figure1": str(FIGURE_DIR / "figure1_umap_methods.png"),
    "figure2": str(FIGURE_DIR / "figure2_metrics.png"),
}

with open(RESULT_DIR / "report_values.json", "w", encoding="utf-8") as f:
    json.dump(report_values, f, indent=2)

report_values
