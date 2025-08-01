library(CellChat)
library(patchwork)
options(stringsAsFactors = FALSE)
library(reticulate)
library(NMF)
library(ggalluvial)
library(anndata)
library(Matrix)

lymphoid_cells = c("NK cells", "CD4 T cells", "CD8 T cells", "T cells proliferating", "B cells", "Plasma cells", "Plasmacytoid DCs")
myeloid_cells = c("Classical monocytes", "Non classical monocytes", "Alveolar macrophages", "Monocyte derived Mφ", "Alveolar Mφ proliferating", "Alveolar Mφ MT-positive", "Interstitial Mφ perivascular", "Alveolar Mφ CCL3+", "DC1", "DC2", "Migratory DCs", "Mast cells")
fibroblast_cells = c("Adventitial fibroblasts", "Alveolar fibroblasts", "Peribronchial fibroblasts", "Subpleural fibroblasts", "Myofibroblasts", "Fibromyocytes")

ear_ad <- read_h5ad("../data/processed_early.h5ad")
print(ear_ad)
ear_counts <- t(as.matrix(ear_ad$X))
library.size <- Matrix::colSums(ear_counts)
data.input <- as(log1p(Matrix::t(Matrix::t(ear_counts)/library.size) * 10000), "dgCMatrix")
ear_meta <- ear_ad$obs
ear_meta$labels <- ear_meta[["leiden_res_20.00_celltype"]]

ear_cc <- createCellChat(object = data.input, meta = ear_meta, group.by = "leiden_res_20.00_celltype")

print('Created Early CellChat object.')

nc_ad <- read_h5ad("../data/processed_non-cancer.h5ad")
nc_counts <- t(as.matrix(nc_ad$X))
library.size <- Matrix::colSums(nc_counts)
data.input <- as(log1p(Matrix::t(Matrix::t(nc_counts)/library.size) * 10000), "dgCMatrix")
nc_meta <- nc_ad$obs
nc_meta$labels <- nc_meta[["leiden_res_20.00_celltype"]]

nc_cc <- createCellChat(object = data.input, meta = nc_meta, group.by = "leiden_res_20.00_celltype")

print('Created Non-cancer CellChat object.')

adv_ad <- read_h5ad("../data/processed_advanced.h5ad")
adv_counts <- t(as.matrix(adv_ad$X))
library.size <- Matrix::colSums(adv_counts)
data.input <- as(log1p(Matrix::t(Matrix::t(adv_counts)/library.size) * 10000), "dgCMatrix")
adv_meta <- adv_ad$obs
adv_meta$labels <- adv_meta[["leiden_res_20.00_celltype"]]

adv_cc <- createCellChat(object = data.input, meta = adv_meta, group.by = "leiden_res_20.00_celltype")

print('Created Advanced CellChat object.')

CellChatDB <- CellChatDB.human
CellChatDB.use <- subsetDB(CellChatDB)
ear_cc@DB <- CellChatDB.use
nc_cc@DB <- CellChatDB.use
adv_cc@DB <- CellChatDB.use

future::plan("multisession", workers = 4)

ear_cc <- subsetData(ear_cc)
ear_cc <- identifyOverExpressedGenes(ear_cc)
ear_cc <- identifyOverExpressedInteractions(ear_cc)
ear_cc <- projectData(ear_cc, PPI.human)

nc_cc <- subsetData(nc_cc) 
nc_cc <- identifyOverExpressedGenes(nc_cc)
nc_cc <- identifyOverExpressedInteractions(nc_cc)
nc_cc <- projectData(nc_cc, PPI.human)

adv_cc <- subsetData(adv_cc)
adv_cc <- identifyOverExpressedGenes(adv_cc)
adv_cc <- identifyOverExpressedInteractions(adv_cc)
adv_cc <- projectData(adv_cc, PPI.human)

options(future.globals.maxSize = 2 * 1024^3)

ear_cc <- computeCommunProb(ear_cc, type = "triMean")
nc_cc <- computeCommunProb(nc_cc, type = "triMean")
adv_cc <- computeCommunProb(adv_cc, type = "triMean")

ear_cc <- filterCommunication(ear_cc, min.cells = 1)
ear_cc <- computeCommunProbPathway(ear_cc)
ear_cc <- aggregateNet(ear_cc)

groupSize <- as.numeric(table(ear_cc@idents))
par(mfrow = c(1,2), xpd=TRUE)
netVisual_circle(ear_cc@net$count, vertex.weight = groupSize, weight.scale = T, label.edge= F, title.name = "Number of interactions")
netVisual_circle(ear_cc@net$weight, vertex.weight = groupSize, weight.scale = T, label.edge= F, title.name = "Interaction weights/strength")

nc_cc <- filterCommunication(nc_cc, min.cells = 1)
nc_cc <- computeCommunProbPathway(nc_cc)
nc_cc <- aggregateNet(nc_cc)

groupSize <- as.numeric(table(nc_cc@idents))
par(mfrow = c(1,2), xpd=TRUE)
netVisual_circle(nc_cc@net$count, vertex.weight = groupSize, weight.scale = T, label.edge= F, title.name = "Number of interactions")
netVisual_circle(nc_cc@net$weight, vertex.weight = groupSize, weight.scale = T, label.edge= F, title.name = "Interaction weights/strength")

adv_cc <- filterCommunication(adv_cc, min.cells = 1)
adv_cc <- computeCommunProbPathway(adv_cc)
adv_cc <- aggregateNet(adv_cc)

groupSize <- as.numeric(table(adv_cc@idents))
par(mfrow = c(1,2), xpd=TRUE)
netVisual_circle(adv_cc@net$count, vertex.weight = groupSize, weight.scale = T, label.edge= F, title.name = "Number of interactions")
netVisual_circle(adv_cc@net$weight, vertex.weight = groupSize, weight.scale = T, label.edge= F, title.name = "Interaction weights/strength")

pdf(file = "adv_netVisual_chord_gene.pdf", width = 20, height = 20)

netVisual_chord_gene(adv_cc, 
  sources.use = lymphoid_cells, 
  targets.use = myeloid_cells, 
  slot.name = "netP", 
  lab.cex = 1.5,  
  small.gap = 3, 
  big.gap = 15,
  annotationTrackHeight = c(0.03),
  link.visible = TRUE,
  scale = FALSE,
  directional = 1,
  link.target.prop = TRUE,
  reduce = -1,
  transparency = 0.2,
  link.border = NA,
  title.name = "Advanced condition",
  legend.pos.x = 5,
  legend.pos.y = 10,
  show.legend = TRUE,
  thresh = 0.05
)

dev.off()

pdf(file = "early_netVisual_chord_gene.pdf", width = 20, height = 20)

netVisual_chord_gene(ear_cc, 
  sources.use = lymphoid_cells,
  targets.use = myeloid_cells, 
  slot.name = "netP", 
  lab.cex = 1.5,  
  small.gap = 3, 
  big.gap = 15,
  annotationTrackHeight = c(0.03),
  link.visible = TRUE,
  scale = FALSE,
  directional = 1,
  link.target.prop = TRUE,
  reduce = -1,
  transparency = 0.2,
  link.border = NA,
  title.name = "Early condition",
  legend.pos.x = 5,
  legend.pos.y = 10,
  show.legend = TRUE,
  thresh = 0.05
)

dev.off()

pdf(file = "noncancer_netVisual_chord_gene.pdf", width = 20, height = 20)

netVisual_chord_gene(nc_cc, 
  sources.use = lymphoid_cells,
  targets.use = myeloid_cells, 
  slot.name = "netP", 
  lab.cex = 1.5,  
  small.gap = 3, 
  big.gap = 15,
  annotationTrackHeight = c(0.03),
  link.visible = TRUE,
  scale = FALSE,
  directional = 1,
  link.target.prop = TRUE,
  reduce = -1,
  transparency = 0.2,
  link.border = NA,
  title.name = "Non-Cancer condition",
  legend.pos.x = 5,
  legend.pos.y = 10,
  show.legend = TRUE,
  thresh = 0.05
)

dev.off()

pathways.show.all <- nc_cc@netP$pathways
levels(nc_cc@idents)
vertex.receiver = seq(1,4)
for (i in 1:length(pathways.show.all)) {
  netVisual(nc_cc, signaling = pathways.show.all[i], vertex.receiver = vertex.receiver, layout = "hierarchy")
  gg <- netAnalysis_contribution(nc_cc, signaling = pathways.show.all[i])
  ggsave(filename=paste0(pathways.show.all[i], "_L-R_contribution.pdf"), plot=gg, width = 3, height = 2, units = 'in', dpi = 300)
}

pathways.show.all <- ear_cc@netP$pathways
levels(ear_cc@idents)
vertex.receiver = seq(1,4)
for (i in 1:length(pathways.show.all)) {
  netVisual(ear_cc, signaling = pathways.show.all[i], vertex.receiver = vertex.receiver, layout = "hierarchy")
  gg <- netAnalysis_contribution(ear_cc, signaling = pathways.show.all[i])
  ggsave(filename=paste0(pathways.show.all[i], "_L-R_contribution.pdf"), plot=gg, width = 3, height = 2, units = 'in', dpi = 300)
}

pathways.show.all <- adv_cc@netP$pathways
levels(adv_cc@idents)
vertex.receiver = seq(1,4)
for (i in 1:length(pathways.show.all)) {
  netVisual(adv_cc, signaling = pathways.show.all[i], vertex.receiver = vertex.receiver, layout = "hierarchy")
  gg <- netAnalysis_contribution(adv_cc, signaling = pathways.show.all[i])
  ggsave(filename=paste0(pathways.show.all[i], "_L-R_contribution.pdf"), plot=gg, width = 3, height = 2, units = 'in', dpi = 300)
}

pdf(file ="noncancer_netVisual_bubble_tmyeloid.pdf", width = 10, height =10)

netVisual_bubble(
  object = nc_cc,
  sources.use = c('CD4 T cells', 'CD8 T cells', 'T cells proliferating'),
  targets.use = myeloid_cells,
  remove.isolate = FALSE,
  thresh = 0.01,
  grid.on = FALSE,
  font.size = 16,
  font.size.title = 20,
  show.legend = TRUE,
  angle.x = 90
)

dev.off()

pdf(file ="noncancer_netVisual_bubble_fibroblastmyleloid.pdf", width = 10, height =10)

netVisual_bubble(
  object = nc_cc,
  sources.use = c('CD4 T cells', 'CD8 T cells', 'T cells proliferating'),
  targets.use = fibroblast_cells,
  remove.isolate = FALSE,
  thresh = 0.05,
  grid.on = FALSE,
  font.size = 16,
  font.size.title = 20,
  show.legend = TRUE,
  angle.x = 90
)

dev.off()

pdf(file ="early_netVisual_bubble_tmyeloid.pdf", width = 10, height =10)

netVisual_bubble(
  object = ear_cc,
  sources.use = c('CD4 T cells', 'CD8 T cells', 'T cells proliferating'),
  targets.use = myeloid_cells,
  remove.isolate = FALSE,
  thresh = 0.01,
  grid.on = FALSE,
  font.size = 16,
  font.size.title = 20,
  show.legend = TRUE,
  angle.x = 90
)

dev.off()

pdf(file ="early_netVisual_bubble_fibroblastmyleloid.pdf", width = 10, height =10)

netVisual_bubble(
  object = ear_cc,
  sources.use = c('CD4 T cells', 'CD8 T cells', 'T cells proliferating'),
  targets.use = fibroblast_cells,
  remove.isolate = FALSE,
  thresh = 0.05,
  grid.on = FALSE,
  font.size = 16,
  font.size.title = 20,
  show.legend = TRUE,
  angle.x = 90
)

dev.off()

pdf(file ="adv_netVisual_bubble_tmyeloid.pdf", width = 10, height =10)

netVisual_bubble(
  object = adv_cc,
  sources.use = c('CD4 T cells', 'CD8 T cells', 'T cells proliferating'),
  targets.use = myeloid_cells,
  remove.isolate = FALSE,
  thresh = 0.01,
  grid.on = FALSE,
  font.size = 16,
  font.size.title = 20,
  show.legend = TRUE,
  angle.x = 90
)

dev.off()

pdf(file ="adv_netVisual_bubble_fibroblastmyleloid.pdf", width = 10, height =10)

netVisual_bubble(
  object = adv_cc,
  sources.use = c('CD4 T cells', 'CD8 T cells', 'T cells proliferating'),
  targets.use = fibroblast_cells,
  remove.isolate = FALSE,
  thresh = 0.05,
  grid.on = FALSE,
  font.size = 16,
  font.size.title = 20,
  show.legend = TRUE,
  angle.x = 90
)

dev.off()

library(ggplot2)
library(ComplexHeatmap)

nc_cc <- netAnalysis_computeCentrality(nc_cc, slot.name = "netP") 

pdf(file ="nc_netAnalysis_signalingRole_outgoing_heatmap.pdf", width = 10, height =10)

ht1 <- netAnalysis_signalingRole_heatmap(nc_cc, pattern = "outgoing", width = 17, height = 17, font.size = 7 )
draw(ht1)

dev.off()

pdf(file ="nc_netAnalysis_signalingRole_incoming_heatmap.pdf", width = 10, height =10)

ht2 <- netAnalysis_signalingRole_heatmap(nc_cc, pattern = "incoming", width = 17, height = 17, font.size = 7 )
draw(ht2)

dev.off()

ear_cc <- netAnalysis_computeCentrality(ear_cc, slot.name = "netP") 

pdf(file ="ear_netAnalysis_signalingRole_outgoing_heatmap.pdf", width = 10, height =10)

ht1 <- netAnalysis_signalingRole_heatmap(ear_cc, pattern = "outgoing", width = 17, height = 17, font.size = 7 )
draw(ht1)

dev.off()

pdf(file ="ear_netAnalysis_signalingRole_incoming_heatmap.pdf", width = 10, height =10)

ht2 <- netAnalysis_signalingRole_heatmap(ear_cc, pattern = "incoming", width = 17, height = 17, font.size = 7 )
draw(ht2)

dev.off()

adv_cc <- netAnalysis_computeCentrality(adv_cc, slot.name = "netP") 

pdf(file ="adv_netAnalysis_signalingRole_outgoing_heatmap.pdf", width = 10, height =10)

ht1 <- netAnalysis_signalingRole_heatmap(adv_cc, pattern = "outgoing", width = 17, height = 17, font.size = 7 )
draw(ht1)

dev.off()

pdf(file ="adv_netAnalysis_signalingRole_incoming_heatmap.pdf", width = 10, height =10)

ht2 <- netAnalysis_signalingRole_heatmap(adv_cc, pattern = "incoming", width = 17, height = 17, font.size = 7 )
draw(ht2)

dev.off()

selectK(nc_cc, pattern = "outgoing")
selectK(ear_cc, pattern = "outgoing")
selectK(adv_cc, pattern = "outgoing")
nPatterns = 6

pdf(file ="nc_identifyCommunicationPatterns.pdf", width = 10, height =10)

nc_cc <- identifyCommunicationPatterns(nc_cc, pattern = "outgoing", k = nPatterns, height = 15, width = 3, font.size = 7)

dev.off()

pdf(file ="ear_identifyCommunicationPatterns.pdf", width = 10, height =10)

ear_cc <- identifyCommunicationPatterns(ear_cc, pattern = "outgoing", k = nPatterns, height = 15, width = 3, font.size = 7)

dev.off()

pdf(file ="adv_identifyCommunicationPatterns.pdf", width = 10, height =10)

adv_cc <- identifyCommunicationPatterns(adv_cc, pattern = "outgoing", k = nPatterns, height = 15, width = 3, font.size = 7)

dev.off()

netAnalysis_dot(nc_cc, pattern = "outgoing")
netAnalysis_dot(ear_cc, pattern = "outgoing")
netAnalysis_dot(adv_cc, pattern = "outgoing")