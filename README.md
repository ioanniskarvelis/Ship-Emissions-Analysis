# Ship Emissions Analysis: PCA and Clustering

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![R >=4.0](https://img.shields.io/badge/R-%E2%89%A5%204.0-276DC3?logo=r&logoColor=white)
![Language: R](https://img.shields.io/badge/Language-R-276DC3?logo=r&logoColor=white)
[![Data: THETIS-MRV](https://img.shields.io/badge/Data-THETIS--MRV-informational)](https://mrv.emsa.europa.eu/#public/emission-report)
![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen)

This repository contains an R implementation of a pipeline for analyzing ship emissions using Principal Component Analysis (PCA) and k-means clustering. It reproduces figures and tables for correlation, PCA, and clustering validation.

## Project Structure

```
project/
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
- Download the dataset Excel report file for 2024 reporting period.
- Place your Excel file as `data/dataset.xlsx`.
- The script expects the first two rows in the Excel file to be metadata and skips them, then drops the next two rows as per the original preprocessing.
- Identification columns expected: `IMO Number...1`, `Name...2`, `Ship type`.

### Data Source

- Public EU MRV (Monitoring, Reporting and Verification) shipping emissions data from the THETIS-MRV portal (EMSA): `https://mrv.emsa.europa.eu/#public/emission-report`.
- Export your required reporting year(s) from the portal as an Excel file and save it as `data/dataset.xlsx`. The variables used match the MRV public dataset (totals, intensities, and EU MS split emissions).

## Analysis Pipeline

End-to-end steps implemented in `src/implementation.R`:

1. Ingestion
   - Read `dataset.xlsx` skipping the first 2 header rows, then drop the next 2 non-data rows.
   - Keep ID columns (`IMO Number...1`, `Name...2`, `Ship type`) and the following numeric indicators (grouped):
     - Totals: Total fuel/CO₂/CH₄/N₂O/CO₂eq, Time at sea
     - Intensities: Fuel/CO₂ per distance; Fuel/CO₂ per transport work
     - EU MS split: CO₂ from voyages between/departing/to MS jurisdictions

2. Cleaning and typing
   - Replace textual placeholders like `Division by zero!` with `NA`.
   - Coerce all selected indicators to numeric and drop incomplete rows (`complete.cases`).

3. Standardization
   - Standardize numerical columns to z-scores and label them `Z1..Z13`.

4. Outlier detection (Mahalanobis)
   - Compute Mahalanobis distances on standardized data; use generalized inverse if covariance is singular.
   - Flag outliers with distance > 10 and create a reduced dataset excluding them. Save diagnostic plots.

5. PCA (full and reduced)
   - Run PCA on (a) full standardized data and (b) reduced data (outliers removed).
   - Save scree plots, loadings tables (PC1–PC3 with variance explained), biplots, and principal component scores (with ship type and IDs).

6. Clustering and validation
   - Fit k-means on the first 3 PCs (reduced data) with `set.seed(123)`.
   - Validation: Silhouette (k=2..20) and Gap statistic (k=1..20); save plots.
   - Models: k=2, k=4, and k=14; save 2D PC scatterplots colored by cluster and a bar chart of cluster centers for k=4.
   - Export final dataset with cluster assignments (IDs, ship type, PCs, and cluster labels).

7. Optional subtype analysis
   - If both "Bulk carrier" and "Container ship" exist, run separate PCAs and save a loadings comparison figure and table.

Outputs are written under `figures/` and `results/` (see list below).

## How to Run

From the project's folder in R or a terminal:

```bash
Rscript src/implementation.R
```

This will generate:

- Figures into `figures/`
- CSV outputs into `results/`

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


