# Ship Emissions Analysis: PCA and Clustering

This repository contains an R implementation of a pipeline for analyzing ship emissions using Principal Component Analysis (PCA) and k-means clustering. It reproduces figures and tables for correlation, PCA, and clustering validation.

## Project Structure

```
portfolio/
  ├─ src/                 # R source code
  │   └─ implementation.R # Main analysis script
  ├─ data/                # Input data (not committed); place dataset.xlsx here
  ├─ figures/             # Generated plots (created by the script)
  ├─ results/             # Generated CSV outputs (created by the script)
  ├─ .gitignore
  ├─ LICENSE
  └─ README.md
```

## Requirements

R (≥ 4.0) and the following R packages:

- readxl
- dplyr
- ggplot2
- corrplot
- pheatmap
- gridExtra
- scales
- RColorBrewer
- cluster
- factoextra
- tidyr
- MASS (used if covariance matrix is singular)

Install packages in R:

```r
packages <- c(
  "readxl", "dplyr", "ggplot2", "corrplot", "pheatmap",
  "gridExtra", "scales", "RColorBrewer", "cluster", "factoextra",
  "tidyr", "MASS"
)
install.packages(setdiff(packages, rownames(installed.packages())))
```

## Data

- Place your Excel file as `portfolio/data/dataset.xlsx`.
- The script expects the first two rows in the Excel file to be metadata and skips them, then drops the next two rows as per the original preprocessing.
- Identification columns expected: `IMO Number...1`, `Name...2`, `Ship type`.

## How to Run

From the `portfolio` folder in R or a terminal:

```bash
Rscript src/implementation.R
```

This will generate:

- Figures into `portfolio/figures/`
- CSV outputs into `portfolio/results/`

## Generated Artifacts

- Correlation heatmap: `correlation_heatmap.png`
- Mahalanobis distance plots: `mahalanobis_plots.png`
- PCA (full and reduced) scree plots: `scree_plot_full.png`, `scree_plot.png`
- PCA biplots: `pca_biplot_full.png`, `pca_biplot.png`
- PCA loadings tables: `pca_loadings_table_full.csv`, `pca_loadings_table.csv`
- PCA scores: `pc_scores_full.csv`, `pc_scores.csv`
- Clustering validation plots: `silhouette_validation.png`, `gap_statistic.png`, `clustering_validation.png`
- Cluster visualizations: `clusters_k2.png`, `clusters_k4.png`
- Cluster centers bar chart: `cluster_centers.png`
- Final dataset with cluster labels: `final_dataset_with_clusters.csv`

## Reproducibility Notes

- All outputs are deterministically generated given the input dataset (`set.seed(123)` is used for k-means).
- Large raw data are ignored via `.gitignore`. Commit small, anonymized samples if needed.

## License

Released under the MIT License (see `LICENSE`).

## Citation

If you use this work, please cite this repository and the underlying data source (EU MRV or your specific dataset source) as appropriate.


