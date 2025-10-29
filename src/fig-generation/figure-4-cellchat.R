# --- Libraries ---
library(CellChat)
library(patchwork)
options(stringsAsFactors = FALSE)
library(reticulate)
library(NMF)
library(ggalluvial)
library(anndata)
library(Matrix)
library(ggplot2)

# --- Cell group definitions (you already had these) ---
lymphoid_cells = c("NK cells", "CD4 T cells", "CD8 T cells", "T cells proliferating", "B cells", "Plasma cells", "Plasmacytoid DCs")
myeloid_cells = c("Classical monocytes", "Non classical monocytes", "Alveolar macrophages", "Monocyte derived Mφ", "Alveolar Mφ proliferating", "Alveolar Mφ MT-positive", "Interstitial Mφ perivascular", "Alveolar Mφ CCL3+", "DC1", "DC2", "Migratory DCs", "Mast cells")
fibroblast_cells = c("Adventitial fibroblasts", "Alveolar fibroblasts", "Peribronchial fibroblasts", "Subpleural fibroblasts", "Myofibroblasts", "Fibromyocytes")
endothelial_cells = c("EC venous pulmonary", "EC arterial", "EC venous systemic", "EC general capillary", "EC aerocyte capillary", "Lymphatic EC mature", "Lymphatic EC differentiating", "Lymphatic EC proliferating")
epithelial_cells = c('AT2', 'AT1', 'Suprabasal', 'Basal resting', 'Multiciliated (non-nasal)', 'Goblet (nasal)', 'Club (nasal)', 'Ciliated (nasal)', 'Club (non-nasal)', 'Multiciliated (nasal)', 'Goblet (bronchial)', 'Transitional Club AT2', 'AT2 proliferating', 'Goblet (subsegmental)')

# --- Helper to go from AnnData to CellChat object ---
make_cc <- function(adata_path, group_col = "leiden_res_20.00_celltype") {
  ad <- read_h5ad(adata_path)
  counts <- t(as.matrix(ad$X))
  lib.size <- Matrix::colSums(counts)
  data.input <- as(log1p(Matrix::t(Matrix::t(counts)/lib.size) * 10000), "dgCMatrix")
  meta <- ad$obs
  meta$labels <- meta[[group_col]]
  createCellChat(object = data.input, meta = meta, group.by = group_col)
}

# --- Read and create CellChat objects ---
ear_cc <- make_cc("data/processed_early.subset_projects.h5ad")
nc_cc  <- make_cc("data/processed_nc.subset_projects.h5ad")
adv_cc <- make_cc("data/processed_advanced.subset_projects.h5ad")

# --- DB setup ---
CellChatDB <- CellChatDB.human
CellChatDB.use <- subsetDB(CellChatDB)
ear_cc@DB <- CellChatDB.use
nc_cc@DB  <- CellChatDB.use
adv_cc@DB <- CellChatDB.use

# --- Parallel + memory ---
future::plan("multisession", workers = 4)
options(future.globals.maxSize = 5 * 1024^3)

# --- Minimal pipeline for dot plots (keep only what's needed) ---
prep_for_dot <- function(cc_obj) {
  cc_obj <- subsetData(cc_obj)
  cc_obj <- identifyOverExpressedGenes(cc_obj)
  cc_obj <- identifyOverExpressedInteractions(cc_obj)
  cc_obj <- projectData(cc_obj, PPI.human)
  cc_obj <- computeCommunProb(cc_obj, type = "triMean")
  cc_obj <- filterCommunication(cc_obj, min.cells = 1)
  # For dot plots at pathway/LR-level we keep pathway probs:
  cc_obj <- computeCommunProbPathway(cc_obj)
  cc_obj
}

ear_cc <- prep_for_dot(ear_cc)
nc_cc  <- prep_for_dot(nc_cc)
adv_cc <- prep_for_dot(adv_cc)

# --- Small utility to safely intersect requested cell sets with what's present ---
.intersect_or_warn <- function(wanted, present, label) {
  keep <- intersect(wanted, present)
  if (length(keep) == 0) {
    warning(sprintf("No %s found in this object. Requested: %s ; Present: %s",
                    label, paste(wanted, collapse=", "),
                    paste(sort(unique(present)), collapse=", ")))
  }
  keep
}

make_dotplots <- function(cc_obj, cond_name,
                          myeloid = myeloid_cells,
                          lymphoid = lymphoid_cells,
                          fibro = fibroblast_cells,
                          epith = epithelial_cells,
                          thresh_me = 0.01,
                          thresh_fe = 0.05, 
                          base_width = 10, base_height = 10) {

  present_groups <- levels(cc_obj@idents)

  src_my <- .intersect_or_warn(myeloid, present_groups, "myeloid sources")
  src_ly <- .intersect_or_warn(lymphoid, present_groups, "lymphoid sources")
  src_fb <- .intersect_or_warn(fibro,   present_groups, "fibroblast sources")
  tgt_ep <- .intersect_or_warn(epith,   present_groups, "epithelial targets")

  if (length(tgt_ep) == 0 || (length(src_my) + length(src_ly) + length(src_fb)) == 0) {
    message(sprintf("[%s] Skipping: no overlapping groups found.", cond_name))
    return(invisible(NULL))
  }

  if (length(src_my) > 0) {
    pdf(file = sprintf("%s_dot_myeloid_to_epithelium.pdf", cond_name),
        width = base_width, height = base_height)
    print(
      netVisual_bubble(
        object = cc_obj,
        sources.use = src_my,
        targets.use = tgt_ep,
        remove.isolate = FALSE,
        thresh = thresh_me,
        grid.on = FALSE,
        font.size = 16,
        font.size.title = 20,
        show.legend = TRUE,
        angle.x = 90
      )
    )
    dev.off()
  }

  if (length(src_ly) > 0) {
    pdf(file = sprintf("%s_dot_lymphoid_to_epithelium.pdf", cond_name),
        width = base_width, height = base_height)
    print(
      netVisual_bubble(
        object = cc_obj,
        sources.use = src_ly,
        targets.use = tgt_ep,
        remove.isolate = FALSE,
        thresh = thresh_me,
        grid.on = FALSE,
        font.size = 16,
        font.size.title = 20,
        show.legend = TRUE,
        angle.x = 90
      )
    )
    dev.off()
  }

  if (length(src_fb) > 0) {
    pdf(file = sprintf("%s_dot_fibroblast_to_epithelium.pdf", cond_name),
        width = base_width, height = 20)
    print(
      netVisual_bubble(
        object = cc_obj,
        sources.use = src_fb,
        targets.use = tgt_ep,
        remove.isolate = FALSE,
        thresh = thresh_fe,
        grid.on = FALSE,
        font.size = 16,
        font.size.title = 20,
        show.legend = TRUE,
        angle.x = 90
      )
    )
    dev.off()
  }

  invisible(NULL)
}

make_dotplots(nc_cc,  "noncancer")
make_dotplots(ear_cc, "early")
make_dotplots(adv_cc, "advanced")