library(dplyr)
library(tidyr)
library(GSEABase)
library(jsonlite)

output_file <- "tcga_sample_gene_matrix.csv"

# 1. Build the TCGA sample-gene expression matrix if it doesn't exist
if (!file.exists(output_file)) {
  message("tcga_sample_gene_matrix.csv not found. Processing TCGA data...")
  
  tcga_folder <- "tcga"
  data_list <- list()
  
  folders <- list.dirs(tcga_folder, recursive = FALSE)
  
  for (folder in folders) {
    tsv_file <- list.files(folder, pattern = "\\.tsv$", full.names = TRUE)
    
    if (length(tsv_file) == 1) {
      message(paste("Processing file:", tsv_file))
      tsv_data <- tryCatch({
        read.delim(tsv_file, header = TRUE, sep = "\t", skip = 1, fill = TRUE, quote = "")
      }, error = function(e) {
        message(paste("Error reading file:", tsv_file, "- Skipping."))
        return(NULL)
      })
      
      if (is.null(tsv_data)) next
      
      message("Filtering unwanted rows...")
      tsv_data <- tsv_data %>%
        filter(!gene_id %in% c("N_unmapped", "N_multimapping", "N_noFeature", "N_ambiguous"))
      
      message("Pivoting data...")
      tsv_data_pivot <- tsv_data %>%
        dplyr::select(gene_name, tpm_unstranded) %>%
        pivot_wider(
          names_from = gene_name,
          values_from = tpm_unstranded,
          values_fn = list(tpm_unstranded = mean),
          values_fill = 0
        )
      
      sample_id <- basename(folder)
      tsv_data_pivot <- tsv_data_pivot %>%
        mutate(Sample = sample_id) %>%
        dplyr::select(Sample, everything())
      
      data_list[[sample_id]] <- tsv_data_pivot
    } else {
      message(paste("No .tsv file found in folder:", folder, "- Skipping."))
    }
  }
  
  message("Combining all samples...")
  combined_data <- bind_rows(data_list)
  
  message("Converting columns to numeric...")
  combined_data <- combined_data %>%
    mutate(across(everything(), ~ as.numeric(as.character(.))))
  
  message(paste("Writing combined data to:", output_file))
  write.csv(combined_data, file = output_file, row.names = FALSE)
} else {
  message("tcga_sample_gene_matrix.csv already exists. Skipping TCGA data processing.")
}

# 2. Function to add and aggregate module (pathway) scores from GMT files
add_and_aggregate_module_scores <- function(data_frame, gmt_file, output_file = "sample_x_pathway.csv") {
  # Load gene sets from GMT file
  gene_sets <- tryCatch({
    getGmt(gmt_file)
  }, error = function(e) {
    stop(paste("Error reading GMT file:", gmt_file, "-", e$message))
  })
  
  gene_list <- lapply(gene_sets@.Data, function(x) x@geneIds)
  names(gene_list) <- sapply(gene_sets@.Data, function(x) x@setName)
  
  pathway_scores <- data.frame(Sample = data_frame$Sample)
  
  for (pathway in names(gene_list)) {
    genes_in_data <- gene_list[[pathway]] %in% colnames(data_frame)
    valid_genes <- gene_list[[pathway]][genes_in_data]
    
    if (length(valid_genes) > 0) {
      print(paste("Processing pathway:", pathway))
      
      # Force matrix to numeric
      numeric_data <- as.matrix(data_frame[, valid_genes, drop = FALSE])
      mode(numeric_data) <- "numeric"
      
      # Calculate rowMeans safely
      pathway_score <- rowMeans(numeric_data, na.rm = TRUE)
      pathway_scores[[pathway]] <- pathway_score
    } else {
      warning(paste("No genes found for pathway", pathway, "in the data frame"))
    }
  }
  
  # Write pathway scores to file
  write.csv(pathway_scores, file = output_file, row.names = FALSE)
  message(paste("Pathway scores written to:", output_file))
  
  return(pathway_scores)
}

# 3. Read in combined_data if not already in environment
if (!exists("combined_data")) {
  message("Reading existing tcga_sample_gene_matrix.csv...")
  combined_data <- read.csv(output_file)
}

# 4. For each Hallmark GMT file, compute pathway scores, combine into one data frame
gmt_folder <- "HallmarkPathGMT"
gmt_files <- list.files(gmt_folder, pattern = "\\.Hs\\.gmt$", full.names = TRUE)

master_pathway_scores <- combined_data[, "Sample", drop = FALSE]
for (gmt_file in gmt_files) {
  message(paste("Processing Hallmark GMT file:", gmt_file))
  pathway_scores <- add_and_aggregate_module_scores(combined_data, gmt_file)
  master_pathway_scores <- left_join(master_pathway_scores, pathway_scores, by = "Sample")
}

final_output_file <- "sample_x_hallmark_pathways.csv"
message(paste("Writing final pathway scores to:", final_output_file))
write.csv(master_pathway_scores, file = final_output_file, row.names = FALSE)

message("Hallmark pathway processing completed.")

# 5. Append case_id and clinical staging information

# -- IMPORTANT CHANGE HERE --
# Make sure we parse JSON as a list-of-lists to avoid "$ operator is invalid for atomic vectors":
message("Appending case_id and clinical staging information...")

# Make sure we parse JSON as a list-of-lists:
metadata <- fromJSON("tcga/metadata.repository.2025-01-25.json",
                     flatten = FALSE,
                     simplifyDataFrame = FALSE,
                     simplifyVector = FALSE,
                     simplifyMatrix = FALSE)

clinical <- read.delim("tcga/clinical.cart.2025-01-27/clinical.tsv",
                       header = TRUE, sep = "\t", fill = TRUE, quote = "")

# Keep only what we need
clinical <- clinical %>%
  dplyr::select(case_id, ajcc_pathologic_stage)

# Build (file_id, case_id) mapping from the top-level JSON fields
case_id_mapping_list <- lapply(metadata, function(record) {
  # Check for required fields
  if (!is.null(record$file_id) && 
      is.list(record$associated_entities) && 
      length(record$associated_entities) > 0) {
    
    # The top-level JSON "file_id" you want:
    top_level_file_id <- record$file_id  
    
    # Gather all case_ids from associated_entities
    case_ids <- sapply(record$associated_entities, function(x) x$case_id)
    
    # Build a row per (file_id, case_id)
    df <- expand.grid(file_id = top_level_file_id,
                      case_id = case_ids,
                      stringsAsFactors = FALSE)
    return(df)
  } else {
    return(NULL)
  }
})

# Combine all into one data frame
case_id_mapping <- do.call(rbind, case_id_mapping_list)

case_id_mapping_output <- "case_id_mapping.csv"
write.csv(case_id_mapping, file = case_id_mapping_output, row.names = FALSE)
message("Case ID mapping written to: ", case_id_mapping_output)

# Read the aggregated pathway scores 
# (make sure 'Sample' column = top-level file_id from JSON)
pathway_scores <- read.csv("sample_x_hallmark_pathways.csv", check.names = FALSE)

# Example debug: see what columns we have
message("Columns in 'sample_x_hallmark_pathways.csv': ", 
        paste(colnames(pathway_scores), collapse = ", "))

# Now join by matching "Sample" in pathway_scores to "file_id" in case_id_mapping
pathway_scores <- pathway_scores %>%
  # First join: to get the case_id column
  left_join(case_id_mapping, by = c("Sample" = "file_id")) %>%
  # Next join: to get the staging info from clinical
  left_join(clinical, by = "case_id")

# Write out final CSV with case_id and ajcc_pathologic_stage
final_output <- "sample_x_hallmark_pathways_with_clinical_staging.csv"
write.csv(pathway_scores, file = final_output, row.names = FALSE)

message("Mapping completed. Final output written to: ", final_output)
##############################################################################
# 1. Load Libraries
##############################################################################
library(dplyr)
library(ggplot2)
library(mixOmics)

##############################################################################
# 2. Define Significant Features
##############################################################################
significant_features <- c(
  "Sample",
  "case_id",
  "ajcc_pathologic_stage",
  "ALLOGRAFT_REJECTION",
  "ANDROGEN_RESPONSE",
  "APICAL_JUNCTION",
  "APOPTOSIS",
  "COAGULATION",
  "COMPLEMENT",
  "E2F_TARGETS",
  "ESTROGEN_RESPONSE_EARLY",
  "ESTROGEN_RESPONSE_LATE",
  "FATTY_ACID_METABOLISM",
  "INFLAMMATORY_RESPONSE",
  "INTERFERON_ALPHA_RESPONSE",
  "INTERFERON_GAMMA_RESPONSE",
  "KRAS_SIGNALING_DN",
  "KRAS_SIGNALING_UP",
  "MITOTIC_SPINDLE",
  "MTORC1_SIGNALING",
  "MYC_TARGETS_V1",
  "MYC_TARGETS_V2",
  "MYOGENESIS",
  "OXIDATIVE_PHOSPHORYLATION",
  "P53_PATHWAY",
  "PANCREAS_BETA_CELLS",
  "PEROXISOME",
  "PROTEIN_SECRETION",
  "REACTIVE_OXYGEN_SPECIES_PATHWAY",
  "TGF_BETA_SIGNALING",
  "UNFOLDED_PROTEIN_RESPONSE",
  "UV_RESPONSE_DN",
  "UV_RESPONSE_UP",
  "WNT_BETA_CATENIN_SIGNALING",
  "REACTOME_SIGNALING_BY_EGFR_IN_CANCER"
)

##############################################################################
# 3. Read CSV File
##############################################################################
final_output <- "sample_x_hallmark_pathways_with_clinical_staging.csv"
data <- read.csv(final_output, check.names = FALSE)

##############################################################################
# 4. Subset Columns to Significant Features
##############################################################################
common_cols <- intersect(names(data), significant_features)

data_subset <- data %>%
  dplyr::select(common_cols)

##############################################################################
# 5. Filter Out NA stage, Remove Duplicates, Remove NA Rows
##############################################################################
# (a) Filter rows with a valid AJCC pathologic stage
data_filtered <- data_subset %>%
  filter(!is.na(ajcc_pathologic_stage))

# (b) Keep the first occurrence of each Sample
data_filtered <- data_filtered %>%
  distinct(Sample, .keep_all = TRUE)

# (c) Remove rows that have any NA in the numeric columns
id_cols <- c("Sample", "case_id", "ajcc_pathologic_stage")
numeric_cols <- setdiff(colnames(data_filtered), id_cols)

# rowSums(...) == 0 means "no missing values in these columns"
data_filtered <- data_filtered %>%
  filter(rowSums(is.na(dplyr::select(., all_of(numeric_cols)))) == 0)

##############################################################################
# 6. Prepare Numeric Matrix for PCA / PLS-DA
##############################################################################
# We'll keep only numeric columns for PCA
all_numeric_cols <- setdiff(colnames(data_filtered), id_cols)

numeric_data <- data_filtered %>%
  dplyr::select(all_of(all_numeric_cols))

rownames(numeric_data) <- data_filtered$Sample

##############################################################################
# 7. Run PCA
##############################################################################
set.seed(42)
pca_results <- prcomp(numeric_data, center = TRUE, scale. = TRUE)

##############################################################################
# 8. Build PCA Data Frame and Plot
##############################################################################
pca_df <- data.frame(
  PC1 = pca_results$x[, 1],
  PC2 = pca_results$x[, 2],
  Sample = rownames(pca_results$x),
  ajcc_pathologic_stage = data_filtered$ajcc_pathologic_stage
)

pca_plot <- ggplot(pca_df, aes(x = PC1, y = PC2, color = ajcc_pathologic_stage)) +
  geom_point(size = 2, alpha = 0.8) +
  theme_minimal() +
  labs(
    title = "PCA (Significant Features Only)",
    subtitle = "Colored by AJCC Pathologic Stage",
    x = "PC1",
    y = "PC2",
    color = "Pathologic Stage"
  )

print(pca_plot)

# Save the PCA plot
ggsave(
  filename = "pca_significant_features.png",
  plot = pca_plot,
  width = 8,
  height = 6,
  dpi = 300
)

##############################################################################
# 9. (Optional) Run PLS-DA with mixOmics
##############################################################################
X <- numeric_data
Y <- data_filtered$ajcc_pathologic_stage

plsda_res <- plsda(X, Y, ncomp = 2)
plotIndiv(plsda_res, comp = c(1, 2), group = Y, legend = TRUE)
