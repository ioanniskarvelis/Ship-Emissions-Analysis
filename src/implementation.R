# -*- coding: utf-8 -*-
# portfolio/src/implementation.R

## Make paths relative to the portfolio folder
root_dir <- normalizePath(file.path(dirname(sys.frame(1)$ofile %||% "./"), ".."), mustWork = FALSE)
data_dir <- file.path(root_dir, "data")
fig_dir <- file.path(root_dir, "figures")
res_dir <- file.path(root_dir, "results")

dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(res_dir, showWarnings = FALSE, recursive = TRUE)

`%||%` <- function(x, y) if (is.null(x)) y else x

## 0. Environment Setup
library(readxl)
library(dplyr)
library(ggplot2)
library(corrplot)
library(pheatmap)
library(gridExtra)
library(scales)
library(RColorBrewer)
library(MASS)
library(cluster)
library(factoextra)
library(tidyr)

## 0.2 Read Data and Skip First 2 Rows
file_path <- file.path(data_dir, "dataset.xlsx")
if (!file.exists(file_path)) {
  stop("Missing input data. Place dataset.xlsx under portfolio/data/")
}
df <- read_excel(file_path, skip = 2)

# Drop the first two rows which are not part of the data
df <- df[-c(1, 2), ]
rownames(df) <- NULL

## 0.3 Select Only Required Columns
columns_to_keep <- c(
  'IMO Number...1',
  'Name...2',
  'Ship type',
  'Total fuel consumption [m tonnes]',
  'Total CO₂ emissions [m tonnes]',
  'Total CH₄ emissions [m tonnes]',
  'Total N₂O emissions [m tonnes]',
  'Total CO₂eq emissions [m tonnes]',
  'Time spent at sea [hours]',
  'Fuel consumption per distance [kg / n mile]',
  'CO₂ emissions per distance [kg CO₂ / n mile]',
  'Fuel consumption per transport work (mass) [g / m tonnes · n miles]',
  'CO₂ emissions per transport work (mass) [g CO₂ / m tonnes · n miles]',
  'CO₂ emissions from all voyages between ports under a MS jurisdiction [m tonnes]',
  'CO₂ emissions from all voyages which departed from ports under a MS jurisdiction [m tonnes]',
  'CO₂ emissions from all voyages to ports under a MS jurisdiction [m tonnes]'
)

df <- df[, columns_to_keep]

## 1. Data Preprocessing
nan_counts <- colSums(is.na(df))
df[df == "Division by zero!"] <- NA

numeric_cols <- columns_to_keep[!columns_to_keep %in% c('IMO Number...1', 'Name...2', 'Ship type')]
df[numeric_cols] <- lapply(df[numeric_cols], as.numeric)
df <- df[complete.cases(df), ]
rownames(df) <- NULL

id_columns <- c('IMO Number...1', 'Name...2', 'Ship type')
numerical_columns <- setdiff(names(df), id_columns)
X <- df[, numerical_columns]

column_mapping <- c(
  'Total fuel consumption [m tonnes]' = 'X1',
  'Total CO₂ emissions [m tonnes]' = 'X2',
  'Total CH₄ emissions [m tonnes]' = 'X3',
  'Total N₂O emissions [m tonnes]' = 'X4',
  'Total CO₂eq emissions [m tonnes]' = 'X5',
  'Time spent at sea [hours]' = 'X6',
  'Fuel consumption per distance [kg / n mile]' = 'X7',
  'CO₂ emissions per distance [kg CO₂ / n mile]' = 'X8',
  'Fuel consumption per transport work (mass) [g / m tonnes · n miles]' = 'X9',
  'CO₂ emissions per transport work (mass) [g CO₂ / m tonnes · n miles]' = 'X10',
  'CO₂ emissions from all voyages between ports under a MS jurisdiction [m tonnes]' = 'X11',
  'CO₂ emissions from all voyages which departed from ports under a MS jurisdiction [m tonnes]' = 'X12',
  'CO₂ emissions from all voyages to ports under a MS jurisdiction [m tonnes]' = 'X13'
)
names(X) <- column_mapping[names(X)]

X_standardized <- as.data.frame(scale(X))
names(X_standardized) <- paste0("Z", 1:ncol(X_standardized))

## 2. Correlation heatmap
correlation_matrix <- cor(X_standardized)
png(file.path(fig_dir, "correlation_heatmap.png"), width = 800, height = 600)
pheatmap(correlation_matrix,
         display_numbers = TRUE,
         number_format = "%.3f",
         color = colorRampPalette(rev(brewer.pal(11, "RdYlBu")))(100),
         main = "Correlation Matrix of Standardized Variables",
         fontsize_number = 8)
dev.off()

## 2.2 Outlier detection via Mahalanobis
calculate_mahalanobis_distances <- function(data) {
  mean_vector <- colMeans(data)
  cov_matrix <- cov(data)
  inv_cov_matrix <- tryCatch(solve(cov_matrix), error = function(e) MASS::ginv(as.matrix(cov_matrix)))
  mahal_distances <- rep(0, nrow(data))
  for (i in 1:nrow(data)) {
    diff <- as.numeric(data[i, ]) - mean_vector
    md <- sqrt(t(diff) %*% inv_cov_matrix %*% diff)
    mahal_distances[i] <- md
  }
  mahal_distances
}

mahal_distances <- calculate_mahalanobis_distances(X_standardized)

p1 <- ggplot(data.frame(obs = 1:length(mahal_distances), md = mahal_distances), aes(x = obs, y = md)) +
  geom_point(alpha = 0.6, size = 1) +
  geom_hline(yintercept = 10, color = "red", linetype = "dashed") +
  labs(x = "Observation", y = "Mahalanobis Distance",
       title = "Mahalanobis Distance for Outlier Detection") +
  theme_minimal() +
  theme(panel.grid = element_line(color = scales::alpha("grey50", 0.3)))

p2 <- ggplot(data.frame(md = mahal_distances), aes(x = md)) +
  geom_histogram(bins = 50, alpha = 0.7, color = "black", fill = "lightblue") +
  geom_vline(xintercept = 10, color = "red", linetype = "dashed") +
  labs(x = "Mahalanobis Distance", y = "Frequency",
       title = "Distribution of Mahalanobis Distances") +
  theme_minimal() +
  theme(panel.grid = element_line(color = scales::alpha("grey50", 0.3)))

png(file.path(fig_dir, "mahalanobis_plots.png"), width = 1200, height = 600)
grid.arrange(p1, p2, ncol = 2)
dev.off()

threshold <- 10
outliers_mask <- mahal_distances > threshold

df_reduced <- df[!outliers_mask, ]
X_reduced <- X[!outliers_mask, ]
X_standardized_reduced <- X_standardized[!outliers_mask, ]

## 3. PCA (full and reduced)
pca_full <- prcomp(X_standardized, center = FALSE, scale. = FALSE)
variance_explained_full <- (pca_full$sdev^2) / sum(pca_full$sdev^2)

scree_data_full <- data.frame(PC = 1:length(variance_explained_full), Variance = variance_explained_full)
p_scree_full <- ggplot(scree_data_full, aes(x = PC, y = Variance)) +
  geom_line(color = "blue", size = 1) +
  geom_point(color = "blue", size = 3) +
  labs(x = "Principal Component", y = "Proportion of Variance Explained", title = "Scree Plot (Full Dataset)") +
  scale_x_continuous(breaks = 1:length(variance_explained_full)) +
  theme_minimal() +
  theme(panel.grid.minor = element_blank())
ggsave(file.path(fig_dir, "scree_plot_full.png"), p_scree_full, width = 10, height = 6)

pca_loadings_full <- pca_full$rotation[, 1:3]
colnames(pca_loadings_full) <- c("PC1", "PC2", "PC3")
loadings_table_full <- cbind(data.frame(Variable = rownames(pca_loadings_full)), as.data.frame(pca_loadings_full))
variance_row_full <- data.frame(Variable = "Variance Explained",
                                PC1 = variance_explained_full[1], PC2 = variance_explained_full[2], PC3 = variance_explained_full[3])
write.csv(rbind(loadings_table_full, variance_row_full), file.path(res_dir, "pca_loadings_table_full.csv"), row.names = FALSE)

pc_scores_full <- pca_full$x[, 1:3]
biplot_data_full <- data.frame(PC1 = pc_scores_full[, 1], PC2 = pc_scores_full[, 2], Ship_Type = df$`Ship type`)
arrow_data_full <- data.frame(Variable = rownames(pca_loadings_full), PC1 = pca_loadings_full[, "PC1"] * 3, PC2 = pca_loadings_full[, "PC2"] * 3)
p_biplot_full <- ggplot() +
  geom_point(data = biplot_data_full, aes(x = PC1, y = PC2, color = Ship_Type), alpha = 0.6, size = 0.8) +
  geom_segment(data = arrow_data_full, aes(x = 0, y = 0, xend = PC1, yend = PC2), arrow = arrow(length = unit(0.3, "cm")), color = "red", size = 0.8) +
  geom_text(data = arrow_data_full, aes(x = PC1 * 1.1, y = PC2 * 1.1, label = Variable), size = 3, color = "red") +
  theme_minimal() + theme(legend.position = "bottom")
ggsave(file.path(fig_dir, "pca_biplot_full.png"), p_biplot_full, width = 12, height = 8)

pc_scores_full_final <- cbind(df[c('IMO Number...1', 'Name...2', 'Ship type')], data.frame(PC1 = pc_scores_full[,1], PC2 = pc_scores_full[,2], PC3 = pc_scores_full[,3]))
write.csv(pc_scores_full_final, file.path(res_dir, "pc_scores_full.csv"), row.names = FALSE)

pca_result <- prcomp(X_standardized_reduced, center = FALSE, scale. = FALSE)
variance_explained <- (pca_result$sdev^2) / sum(pca_result$sdev^2)
cumulative_variance <- cumsum(variance_explained)

scree_data <- data.frame(PC = 1:length(variance_explained), Variance = variance_explained)
p_scree <- ggplot(scree_data, aes(x = PC, y = Variance)) +
  geom_line(color = "blue", size = 1) +
  geom_point(color = "blue", size = 3) +
  labs(x = "Principal Component", y = "Proportion of Variance Explained", title = "Scree Plot") +
  scale_x_continuous(breaks = 1:length(variance_explained)) +
  theme_minimal() + theme(panel.grid.minor = element_blank())
ggsave(file.path(fig_dir, "scree_plot.png"), p_scree, width = 10, height = 6)

pc_loadings_3 <- pca_result$rotation[, 1:3]
colnames(pc_loadings_3) <- c("PC1", "PC2", "PC3")
loadings_df <- cbind(data.frame(Variable = rownames(pc_loadings_3)), as.data.frame(pc_loadings_3))
variance_row <- data.frame(Variable = "Variance Explained", PC1 = variance_explained[1], PC2 = variance_explained[2], PC3 = variance_explained[3])
write.csv(rbind(loadings_df, variance_row), file.path(res_dir, "pca_loadings_table.csv"), row.names = FALSE)

pc_scores <- pca_result$x[, 1:3]
biplot_data <- data.frame(PC1 = pc_scores[, 1], PC2 = pc_scores[, 2], Ship_Type = df_reduced$`Ship type`)
arrow_data <- data.frame(Variable = rownames(pc_loadings_3), PC1 = pc_loadings_3[, "PC1"] * 3, PC2 = pc_loadings_3[, "PC2"] * 3)
p_biplot <- ggplot() +
  geom_point(data = biplot_data, aes(x = PC1, y = PC2, color = Ship_Type), alpha = 0.6, size = 0.8) +
  geom_segment(data = arrow_data, aes(x = 0, y = 0, xend = PC1, yend = PC2), arrow = arrow(length = unit(0.3, "cm")), color = "red", size = 0.8) +
  geom_text(data = arrow_data, aes(x = PC1 * 1.1, y = PC2 * 1.1, label = Variable), size = 3, color = "red") +
  theme_minimal() + theme(legend.position = "bottom")
ggsave(file.path(fig_dir, "pca_biplot.png"), p_biplot, width = 12, height = 8)

pc_scores_df <- data.frame(PC1 = pc_scores[,1], PC2 = pc_scores[,2], PC3 = pc_scores[,3])
pc_scores_final <- cbind(df_reduced[c('IMO Number...1', 'Name...2', 'Ship type')], pc_scores_df)
write.csv(pc_scores_final, file.path(res_dir, "pc_scores.csv"), row.names = FALSE)

## 4. Clustering and validation
set.seed(123)
pc_data_for_clustering <- pc_scores_df[, 1:3]

# Silhouette
k_range <- 2:20
silhouette_scores <- numeric(length(k_range))
for (i in seq_along(k_range)) {
  k <- k_range[i]
  km <- kmeans(pc_data_for_clustering, centers = k, nstart = 25, iter.max = 100)
  sil <- silhouette(km$cluster, dist(pc_data_for_clustering))
  silhouette_scores[i] <- mean(sil[, 3])
}
silhouette_data <- data.frame(k = k_range, silhouette = silhouette_scores)
p_silhouette <- ggplot(silhouette_data, aes(x = k, y = silhouette)) +
  geom_line(color = "blue", size = 1) + geom_point(color = "blue", size = 3) +
  labs(x = "Number of clusters k", y = "Average silhouette width", title = "Silhouette Analysis") +
  scale_x_continuous(breaks = k_range) + theme_minimal()

# Gap statistic
gap_stat <- clusGap(pc_data_for_clustering, FUN = kmeans, nstart = 25, K.max = 20, B = 50)
gap_data <- data.frame(k = 1:20, gap = gap_stat$Tab[, "gap"], gap_se = gap_stat$Tab[, "SE.sim"])
p_gap <- ggplot(gap_data, aes(x = k, y = gap)) +
  geom_line(color = "blue", size = 1) + geom_point(color = "blue", size = 3) +
  geom_errorbar(aes(ymin = gap - gap_se, ymax = gap + gap_se), width = 0.2, color = "blue") +
  labs(x = "Number of clusters k", y = "Gap statistic", title = "Gap Statistic") +
  scale_x_continuous(breaks = 1:20) + theme_minimal()

ggsave(file.path(fig_dir, "clustering_validation.png"), grid.arrange(p_silhouette, p_gap, ncol = 2), width = 12, height = 6)
ggsave(file.path(fig_dir, "silhouette_validation.png"), p_silhouette, width = 8, height = 5)
ggsave(file.path(fig_dir, "gap_statistic.png"), p_gap, width = 8, height = 5)

# K-means models
kmeans_k2 <- kmeans(pc_data_for_clustering, centers = 2, nstart = 25, iter.max = 100)
kmeans_k4 <- kmeans(pc_data_for_clustering, centers = 4, nstart = 25, iter.max = 100)
kmeans_k14 <- kmeans(pc_data_for_clustering, centers = 14, nstart = 25, iter.max = 100)

pc_scores_with_clusters <- pc_scores_final
pc_scores_with_clusters$Cluster_k2 <- as.factor(kmeans_k2$cluster)
pc_scores_with_clusters$Cluster_k4 <- as.factor(kmeans_k4$cluster)
pc_scores_with_clusters$Cluster_k14 <- as.factor(kmeans_k14$cluster)

# Cluster plots
p_k2_pc1_pc2 <- ggplot(pc_scores_with_clusters, aes(x = PC1, y = PC2, color = Cluster_k2)) +
  geom_point(alpha = 0.7, size = 1) + theme_minimal()
p_k2_pc1_pc3 <- ggplot(pc_scores_with_clusters, aes(x = PC1, y = PC3, color = Cluster_k2)) +
  geom_point(alpha = 0.7, size = 1) + theme_minimal()
p_k2_pc2_pc3 <- ggplot(pc_scores_with_clusters, aes(x = PC2, y = PC3, color = Cluster_k2)) +
  geom_point(alpha = 0.7, size = 1) + theme_minimal()
ggsave(file.path(fig_dir, "clusters_k2.png"), grid.arrange(p_k2_pc1_pc2, p_k2_pc1_pc3, p_k2_pc2_pc3, ncol = 2, nrow = 2), width = 12, height = 10)

p_k4_pc1_pc2 <- ggplot(pc_scores_with_clusters, aes(x = PC1, y = PC2, color = Cluster_k4)) +
  geom_point(alpha = 0.7, size = 1) + theme_minimal()
p_k4_pc1_pc3 <- ggplot(pc_scores_with_clusters, aes(x = PC1, y = PC3, color = Cluster_k4)) +
  geom_point(alpha = 0.7, size = 1) + theme_minimal()
p_k4_pc2_pc3 <- ggplot(pc_scores_with_clusters, aes(x = PC2, y = PC3, color = Cluster_k4)) +
  geom_point(alpha = 0.7, size = 1) + theme_minimal()
ggsave(file.path(fig_dir, "clusters_k4.png"), grid.arrange(p_k4_pc1_pc2, p_k4_pc1_pc3, p_k4_pc2_pc3, ncol = 2, nrow = 2), width = 12, height = 10)

# Cluster centers (k=4)
cluster_centers_k4 <- kmeans_k4$centers
colnames(cluster_centers_k4) <- c("PC1", "PC2", "PC3")
centers_df <- data.frame(Cluster = factor(1:4), PC1 = cluster_centers_k4[, "PC1"], PC2 = cluster_centers_k4[, "PC2"], PC3 = cluster_centers_k4[, "PC3"])
centers_long <- tidyr::pivot_longer(centers_df, cols = c(PC1, PC2, PC3), names_to = "PC", values_to = "Value")
p_centers <- ggplot(centers_long, aes(x = Cluster, y = Value, fill = PC)) + geom_bar(stat = "identity", position = "dodge") + theme_minimal()
ggsave(file.path(fig_dir, "cluster_centers.png"), p_centers, width = 10, height = 6)

# Save final dataset with cluster assignments
write.csv(pc_scores_with_clusters, file.path(res_dir, "final_dataset_with_clusters.csv"), row.names = FALSE)

# Save validation comparison for PCs across ship types (bulk vs container) if present
ship_types_count <- table(df_reduced$`Ship type`)
if ("Bulk carrier" %in% names(ship_types_count) && "Container ship" %in% names(ship_types_count)) {
  bulk_carriers_mask <- df_reduced$`Ship type` == "Bulk carrier"
  container_ships_mask <- df_reduced$`Ship type` == "Container ship"
  X_standardized_bulk_carriers <- X_standardized_reduced[bulk_carriers_mask, ]
  X_standardized_container_ships <- X_standardized_reduced[container_ships_mask, ]
  pca_bulk_carriers <- prcomp(X_standardized_bulk_carriers, center = FALSE, scale. = FALSE)
  pca_container_ships <- prcomp(X_standardized_container_ships, center = FALSE, scale. = FALSE)
  variance_explained_bulk <- (pca_bulk_carriers$sdev^2) / sum(pca_bulk_carriers$sdev^2)
  variance_explained_container <- (pca_container_ships$sdev^2) / sum(pca_container_ships$sdev^2)
  pc_loadings_bulk <- pca_bulk_carriers$rotation[, 1:3]
  pc_loadings_container <- pca_container_ships$rotation[, 1:3]
  loadings_comparison <- data.frame(
    Variable = rownames(pc_loadings_bulk),
    Bulk_PC1 = pc_loadings_bulk[, 1], Container_PC1 = pc_loadings_container[, 1],
    Bulk_PC2 = pc_loadings_bulk[, 2], Container_PC2 = pc_loadings_container[, 2],
    Bulk_PC3 = pc_loadings_bulk[, 3], Container_PC3 = pc_loadings_container[, 3]
  )
  variance_row_cmp <- data.frame(
    Variable = "Variance Explained",
    Bulk_PC1 = variance_explained_bulk[1], Container_PC1 = variance_explained_container[1],
    Bulk_PC2 = variance_explained_bulk[2], Container_PC2 = variance_explained_container[2],
    Bulk_PC3 = variance_explained_bulk[3], Container_PC3 = variance_explained_container[3]
  )
  write.csv(rbind(loadings_comparison, variance_row_cmp), file.path(res_dir, "pca_validation_comparison.csv"), row.names = FALSE)
  # Comparison plot
  loadings_plot_data <- loadings_comparison
  loadings_long <- tidyr::pivot_longer(loadings_plot_data,
                                       cols = c(Bulk_PC1, Container_PC1, Bulk_PC2, Container_PC2, Bulk_PC3, Container_PC3),
                                       names_to = "Analysis", values_to = "Loading") %>%
    tidyr::separate(Analysis, into = c("Ship_Type", "PC"), sep = "_")
  p_loadings_comparison <- ggplot(loadings_long, aes(x = Variable, y = Loading, fill = Ship_Type)) +
    geom_bar(stat = "identity", position = "dodge") + facet_wrap(~PC, scales = "free_y") +
    theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust = 1))
  ggsave(file.path(fig_dir, "pca_loadings_comparison.png"), p_loadings_comparison, width = 14, height = 10)
}

message("Analysis complete. Figures in portfolio/figures, results in portfolio/results.")


