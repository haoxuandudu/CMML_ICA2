# CMML ICA2: MultiVI single-cell multiome benchmark

This repository contains the code, report and supporting materials for a CMML3 mini-project on **integrating single-cell RNA and chromatin-accessibility data with MultiVI**.

## Project summary

The project benchmarks MultiVI against MOFA+ and Cobolt on annotated 10x PBMC Multiome RNA+ATAC data. The analysis simulates a mosaic integration setting by creating RNA-only, ATAC-only and paired observations from paired multiome cells, then evaluates whether each method mixes modality groups while preserving transferred immune-cell labels.

Current completed results use:

- 4,989 PBMC nuclei
- 3,000 highly variable genes
- 5,000 frequent ATAC peaks
- 17 transferred immune-cell classes
- MultiVI, MOFA+ and Cobolt as integration methods

## Repository contents

- `report_main.md`: main report text in a brief-communication style.
- `supporting_materials.md`: code documentation, supplementary methods and reflection.
- `CMML3_MultiVI_report_and_supporting_materials.docx`: combined Word version of the report and supporting materials.
- `notebooks/01_multivi_benchmark.py`: executable Python workflow for preprocessing, model training, benchmarking and plotting.
- `notebooks/01_multivi_benchmark.ipynb`: lightweight notebook entry point.
- `requirements.txt`: Python dependency list.
- `run_on_cloud_gpu.sh`: convenience script for cloud GPU runs.
- `figures/`: final report figures.
- `tables/method_metrics.csv`: benchmark metrics used in the report.
- `results/report_values.json`: machine-readable summary values.
- `vendor/cobolt-0.0.1.zip`: Cobolt source package for environments that cannot access GitHub directly.

Large local data files and regenerated model checkpoints are intentionally excluded from Git using `.gitignore`.

## Reproducing the workflow

Use Python 3.10 or 3.11 where possible. A GPU is recommended for MultiVI and Cobolt, although small test runs can be done on CPU.

```bash
python -m pip install -r requirements.txt
python -m pip install vendor/cobolt-0.0.1.zip
python notebooks/01_multivi_benchmark.py
```

For a quick cloud-GPU sanity run:

```bash
CMML3_N_CELLS=1000 CMML3_MULTIVI_EPOCHS=20 CMML3_COBOLT_EPOCHS=20 CMML3_MOFA_ITER=200 python notebooks/01_multivi_benchmark.py
```

For the completed report-scale run:

```bash
CMML3_N_CELLS=5000 CMML3_MULTIVI_EPOCHS=100 CMML3_COBOLT_EPOCHS=100 CMML3_MOFA_ITER=1000 python notebooks/01_multivi_benchmark.py
```

The script uses local files in `data/` if present; otherwise it downloads the annotated PBMC Multiome AnnData objects from the Zenodo source referenced in the report.

## Main outputs

- `figures/figure1_umap_methods.png`
- `figures/figure1_umap_methods_revised.png`
- `figures/figure2_metrics.png`
- `figures/figure2_metrics.pdf`
- `tables/method_metrics.csv`
- `results/report_values.json`

## Notes on interpretation

The benchmark evaluates a controlled simulated mosaic-integration task generated from paired multiome data. This provides matched biological origin and transferred cell-type labels, but it may underestimate technical differences present in independently generated scRNA-seq and scATAC-seq datasets. Cell-type labels are treated as an external reference for comparison rather than absolute ground truth.

## Acknowledgement

Code was developed with assistance from OpenAI Codex and follows public documentation for scvi-tools, scanpy, muon/MOFA+, Cobolt, scikit-learn and the PBMC Multiome benchmark data. Public method documentation and packages are cited in the report references.
