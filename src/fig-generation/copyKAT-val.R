library(devtools)
library(Seurat)
library(parallelDist)
library(RColorBrewer)
library(gplots)
library(copykat)
library(ggplot2)

baseDataPath <- "./data/scRNAseq-data/PCLAcohort"

if (!dir.exists("figures")) {
  dir.create("figures", recursive = TRUE)
}

sampleFolders <- list.dirs(baseDataPath, full.names = FALSE, recursive = FALSE)
sampleFolders <- sampleFolders[grepl("^SMP-", sampleFolders)]

normalize_ids <- function(ids) {
  ids <- gsub("-1$|_1$|\\.1$", "", ids)  
  ids <- gsub("-2$|_2$|\\.2$", "", ids) 
  ids <- gsub("-3$|_3$|\\.3$", "", ids)
  ids <- gsub("-", "", ids)  
  ids <- gsub("_", "", ids)   
  ids <- gsub("\\.", "", ids)
  return(ids)
}

convert_copykat_to_seurat_format <- function(copykat_ids) {
  seurat_format <- gsub("\\.", "-", copykat_ids)
  return(seurat_format)
}

convert_seurat_to_copykat_format <- function(seurat_ids) {
  copykat_format <- gsub("-", ".", seurat_ids)
  return(copykat_format)
}

for (sampleName in sampleFolders) {
  dataPath <- file.path(baseDataPath, sampleName)
  cat("Processing sample:", sampleName, "\n")

  raw <- Read10X(data.dir = dataPath)
  raw <- CreateSeuratObject(counts = raw, project = "copykat.test", min.cells = 3, min.features = 100)

  raw[["percent.mt"]] <- PercentageFeatureSet(raw, pattern = "^MT-")
  raw <- subset(raw, subset = nFeature_RNA > 200 & nFeature_RNA < 5000 & percent.mt < 20)

  exp.rawdata <- as.matrix(GetAssayData(raw, layer = "counts"))
  cat("Matrix dimensions:", nrow(exp.rawdata), "genes x", ncol(exp.rawdata), "cells\n")

  copykat.test <- copykat(
    rawmat = exp.rawdata,
    id.type = "S",
    cell.line = "no",
    ngene.chr = 3,
    LOW.DR = 0.02,
    UP.DR = 0.05,
    win.size = 15,
    norm.cell.names = "",
    KS.cut = 0.15,
    sam.name = sampleName,
    distance = "euclidean",
    output.seg = "FALSE",
    plot.genes = "TRUE",
    genome = "hg20",
    n.cores = 8
  )

  pred.test <- data.frame(copykat.test$prediction)
  pred.test <- pred.test[which(pred.test$copykat.pred %in% c("aneuploid", "diploid")), ]
  CNA.test <- data.frame(copykat.test$CNAmat)

  raw <- NormalizeData(raw, normalization.method = "LogNormalize", scale.factor = 10000)
  raw <- FindVariableFeatures(raw, selection.method = "vst", nfeatures = 2000)
  raw <- ScaleData(raw, features = rownames(raw))
  raw <- RunPCA(raw, features = VariableFeatures(object = raw), npcs = 50)
  ElbowPlot(raw, ndims = 50)
  raw <- FindNeighbors(raw, dims = 1:30, k.param = 20)
  raw <- FindClusters(raw, resolution = 0.3)
  raw <- RunUMAP(raw, dims = 1:30, n.neighbors = 30, min.dist = 0.3)

  copykat_pred <- rep("unknown", ncol(raw))
  names(copykat_pred) <- colnames(raw)

  if (nrow(pred.test) > 0) {
    pred_converted <- convert_copykat_to_seurat_format(pred.test$cell.names)
    exact_matches <- intersect(colnames(raw), pred_converted)
    
    cat("Debug: Found", length(exact_matches), "format-converted matches\n")
    
    if (length(exact_matches) > 0) {
      for (cell in exact_matches) {
        original_idx <- which(pred_converted == cell)
        if (length(original_idx) > 0) {
          copykat_pred[cell] <- pred.test$copykat.pred[original_idx[1]]
        }
      }
    } else {
      exact_matches <- intersect(colnames(raw), pred.test$cell.names)
      cat("Debug: Found", length(exact_matches), "exact matches\n")
      
      if (length(exact_matches) > 0) {
        for (cell in exact_matches) {
          pred_idx <- which(pred.test$cell.names == cell)
          if (length(pred_idx) > 0) {
            copykat_pred[cell] <- pred.test$copykat.pred[pred_idx[1]]
          }
        }
      } else {
        seurat_base <- gsub("-\\d+$", "", colnames(raw))
        pred_base <- gsub("-\\d+$|\\.\\d+$", "", pred.test$cell.names)
        for (i in seq_along(seurat_base)) {
          matches <- which(pred_base == seurat_base[i])
          if (length(matches) > 0) {
            copykat_pred[i] <- pred.test$copykat.pred[matches[1]]
          }
        }
        cat("Debug: Found", sum(copykat_pred != "unknown"), "pattern matches\n")
      }
    }
  }

  raw$copykat_pred <- copykat_pred
  cat("Final cell type distribution:\n")
  print(table(raw$copykat_pred))

  if (sum(copykat_pred != "unknown") > 0) {
    umap_plot1 <- DimPlot(raw, reduction = "umap", group.by = "copykat_pred",
                        cols = c("aneuploid" = "#FF8C00", "diploid" = "#84A970", "unknown" = "#D3D3D3"),
                        pt.size = 0.8) +
    ggtitle(paste0(sampleName, " - CopyKAT Classification")) +
    theme_minimal(base_size = 14) +
    theme(
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        axis.line = element_line(color = "black"),
        plot.title = element_text(hjust = 0.5, size = 16, face = "bold")
    )

    umap_plot2 <- DimPlot(raw, reduction = "umap", group.by = "seurat_clusters",
                        label = TRUE, pt.size = 0.8) +
    ggtitle(paste0(sampleName, " - Seurat Clusters")) +
    theme_minimal(base_size = 14) +
    theme(
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        axis.line = element_line(color = "black"),
        plot.title = element_text(hjust = 0.5, size = 16, face = "bold")
    )

    ggsave(file.path("figures", paste0(sampleName, "-copykat-umap-enhanced.png")),
           plot = umap_plot1 + umap_plot2, width = 16, height = 8, dpi = 600)
    ggsave(file.path("figures", paste0(sampleName, "-copykat-classification-only.png")),
           plot = umap_plot1, width = 10, height = 10, dpi = 600)

    cat("\n=== COPYKAT ANALYSIS SUMMARY FOR", sampleName, "===\n")
    classification_table <- table(raw$copykat_pred)
    print(classification_table)

    cat("\nPercentages:\n")
    total_cells <- sum(classification_table)
    percentages <- round(100 * classification_table / total_cells, 2)
    for (i in names(percentages)) {
      cat(paste0(i, ": ", percentages[i], "%\n"))
    }

    cat("\nCopyKAT predictions by Seurat cluster:\n")
    print(table(raw$seurat_clusters, raw$copykat_pred))
  } else {
    cat("Warning: No successful copyKAT predictions matched to cells. Skipping UMAP plots.\n")
  }

  cat("CNA matrix dimensions:", nrow(CNA.test), "x", ncol(CNA.test), "\n")

  if (ncol(CNA.test) > 4 && nrow(CNA.test) > 0) {
    my_palette <- colorRampPalette(rev(brewer.pal(n = 3, name = "RdBu")))(n = 999)
    
    chr <- as.numeric(CNA.test$chrom) %% 2 + 1
    CHR <- colorRampPalette(c('black', 'grey'))(2)[as.numeric(chr)]
    chr1 <- cbind(CHR, CHR)
    pred <- colorRampPalette(c("#1F78B4", "#E31A1C"))(2)[as.numeric(factor(pred.test$copykat.pred))]
    cells <- rbind(pred, pred)

    col_breaks <- c(seq(-1, -0.4, length = 50), seq(-0.4, -0.2, length = 150),
                    seq(-0.2, 0.2, length = 600), seq(0.2, 0.4, length = 150),
                    seq(0.4, 1, length = 50))

    png(file.path("figures", paste0(sampleName, "-copykat-heatmap-all-enhanced.png")),
        width = 3000, height = 2000, res = 200)
    par(mar = c(8, 8, 4, 8), oma = c(2, 2, 2, 2))
    tryCatch({
      heatmap.3(
        t(CNA.test[, 4:ncol(CNA.test)]),
        dendrogram = "r",
        distfun = function(x) parDist(x, threads = 4, method = "euclidean"),
        hclustfun = function(x) hclust(x, method = "ward.D2"),
        ColSideColors = chr1,
        RowSideColors = cells,
        Colv = NA,
        Rowv = TRUE,
        notecol = "black",
        col = my_palette,
        breaks = col_breaks,
        key = TRUE,
        keysize = 1.0,
        density.info = "none",
        trace = "none",
        cexRow = 0.05,
        cexCol = 0.05,
        margins = c(5, 5),
        main = paste0(sampleName, " - CopyKAT CNV Heatmap")
      )
      legend("topright", paste("", names(table(pred.test$copykat.pred)), sep = ""),
             pch = 15, col = c("#1F78B4", "#E31A1C"), cex = 0.6, bty = "n", title = "Cell Type")
    }, error = function(e) {
      cat("Error creating all-cells heatmap:", e$message, "\n")
    })
    dev.off()
  }

  tumor_cells_raw <- pred.test$cell.names[which(pred.test$copykat.pred == "aneuploid")]
  cna_cell_names <- colnames(CNA.test)[4:ncol(CNA.test)]

  cat("Sample tumor cell format:", head(tumor_cells_raw, 2), "\n")
  cat("Sample CNA cell format:", head(cna_cell_names, 2), "\n")

  matched_cna_cols <- c()

  exact_matches <- intersect(tumor_cells_raw, cna_cell_names)
  if(length(exact_matches) > 0) {
    matched_cna_cols <- exact_matches
    cat("Using exact matches:", length(matched_cna_cols), "\n")
  } else {
    tumor_cells_converted <- convert_seurat_to_copykat_format(tumor_cells_raw)
    direct_matches <- intersect(tumor_cells_converted, cna_cell_names)
    
    if(length(direct_matches) > 0) {
      matched_cna_cols <- direct_matches
      cat("Using format-converted matches:", length(matched_cna_cols), "\n")
    } else {
      tumor_ids_norm <- normalize_ids(tumor_cells_raw)
      cna_ids_norm <- normalize_ids(cna_cell_names)
      norm_matches <- cna_cell_names[cna_ids_norm %in% tumor_ids_norm]
      
      if(length(norm_matches) > 0) {
        matched_cna_cols <- norm_matches
        cat("Using normalized matches:", length(matched_cna_cols), "\n")
      } else {
        cat("No matching strategy worked!\n")
        cat("Tumor format example:", tumor_cells_raw[1], "\n")
        cat("CNA format example:", cna_cell_names[1], "\n")
      }
    }
  }

  cat("Final matched tumor cells in CNA matrix:", length(matched_cna_cols), "\n")

  tumor_mat <- CNA.test[, matched_cna_cols, drop = FALSE]

  if (ncol(tumor_mat) > 10) {
    hcc <- hclust(parDist(t(tumor_mat), threads = 4, method = "euclidean"), method = "ward.D2")
    hc.umap <- cutree(hcc, 2)
    
    tumor_color_palette <- colorRampPalette(brewer.pal(n = 8, name = "Set1")[1:2])(2)
    subpop <- tumor_color_palette[as.numeric(factor(hc.umap))]
    cells_tumor <- rbind(subpop, subpop)

    png(file.path("figures", paste0(sampleName, "-copykat-heatmap-tumor-enhanced.png")),
        width = 2500, height = 1800, res = 200)
    par(mar = c(6, 6, 4, 6), oma = c(2, 2, 2, 2))
    tryCatch({
      heatmap.3(
        t(tumor_mat),
        dendrogram = "r",
        distfun = function(x) parDist(x, threads = 4, method = "euclidean"),
        hclustfun = function(x) hclust(x, method = "ward.D2"),
        ColSideColors = chr1,
        RowSideColors = cells_tumor,
        Colv = NA,
        Rowv = TRUE,
        notecol = "black",
        col = my_palette, 
        breaks = col_breaks,
        key = TRUE,
        keysize = 1.0,
        density.info = "none",
        trace = "none",
        cexRow = 0.1,
        cexCol = 0.1,
        margins = c(5, 5),
        main = paste0(sampleName, " - Tumor Subclusters")
      )
      legend("topright", c("Subclone 1", "Subclone 2"),
             pch = 15, col = brewer.pal(n = 8, name = "Set1")[1:2],
             cex = 0.6, bty = "n")
    }, error = function(e) {
      cat("Error creating tumor heatmap:", e$message, "\n")
    })
    dev.off()
  } else {
    cat("Not enough tumor cells in CNA matrix for", sampleName, "- skipping tumor heatmap\n")
  }

  cat("\n", rep("=", 80), "\n\n", sep = "")
}