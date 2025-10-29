import logging
import os
from importlib import resources
from pathlib import Path
from typing import Dict, Iterable, Optional
import gseapy as gp
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import scipy.sparse as sp
import umap
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import MinMaxScaler
import anndata as ad
import warnings

CONDITIONS = {
    'ADIPOGENESIS': ['EMT and Metastasis', 'Inflammation'],
    'ALLOGRAFT_REJECTION': ['Immune', 'Inflammation'],
    'ANDROGEN_RESPONSE': ['Sensitivity to growth'],
    'ANGIOGENESIS':['Angiogenesis', 'Immune', 'Inflammation'],
    'APICAL_JUNCTION':['EMT and Metastasis', 'Angiogenesis'],
    'APICAL_SURFACE':['EMT and Metastasis', 'Angiogenesis'],
    'APOPTOSIS':['Apoptosis', 'Replication'],
    'BILE_ACID_METABOLISM':['Sensitivity to growth', 'Immune', 'Energetics'],
    'CHOLESTEROL_HOMEOSTASIS':['Sensitivity to growth', 'Immune', 'Energetics'],
    'COAGULATION':['Angiogenesis', 'Immune'],
    'COMPLEMENT':['Immune','Apoptosis'],
    'DNA_REPAIR':['Proliferative signal', 'Genome instability'],
    'E2F_TARGETS':['Proliferative signal'],
    'EPITHELIAL_MESENCHYMAL_TRANSITION':['EMT and Metastasis'],
    'ESTROGEN_RESPONSE_EARLY':['Sensitivity to growth'],
    'FATTY_ACID_METABOLISM':['Sensitivity to growth', 'Immune', 'Energetics'],
    'G2M_CHECKPOINT':['Proliferative signal', 'Genome instability'],
    'GLYCOLYSIS':['Sensitivity to growth'],
    'HEDGEHOG_SIGNALING':['Insensitivity to antigrowth','Immune', 'Sensitivity to growth'],
    'HEME_METABOLISM':['Angiogenesis'],
    'HYPOXIA':['Sensitivity to growth','EMT and Metastasis', 'Energetics'],
    'IL2_STAT5_SIGNALING':['Immune'],
    'IL6_JAK_STAT3_SIGNALING':['Insensitivity to antigrowth'],
    'INFLAMMATORY_RESPONSE':['Inflammation', 'Sensitivity to growth'],
    'INTERFERON_ALPHA_RESPONSE': ['Immune', 'Inflammation'],
    'INTERFERON_GAMMA_RESPONSE':['Immune','Insensitivity to antigrowth'],
    'KRAS_SIGNALING_DN': ['Insensitivity to antigrowth'],
    'KRAS_SIGNALING_UP':['Sensitivity to growth'],
    'MITOTIC_SPINDLE':['Replication', 'Genome instability'],
    'MTORC1_SIGNALING':['Apoptosis','Insensitivity to antigrowth', 'Energetics'],  
    'MYC_TARGETS_V1':['Sensitivity to growth', 'Energetics'],
    'MYC_TARGETS_V2':['Sensitivity to growth', 'Energetics'],
    'MYOGENESIS':['EMT and Metastasis'],
    'NOTCH_SIGNALING':['EMT and Metastasis','Insensitivity to antigrowth', 'Sensitivity to growth' ],
    'OXIDATIVE_PHOSPHORYLATION':['Energetics'],
    'P53_PATHWAY':['Apoptosis', 'Replication'],
    'PANCREAS_BETA_CELLS':['Energetics'],
    'PEROXISOME':['Energetics'],
    'PI3K_AKT_MTOR_SIGNALING':['Apoptosis', 'Insensitivity to antigrowth', 'Energetics'],
    'PROTEIN_SECRETION':['EMT and Metastasis', 'Angiogenesis'],
    'REACTIVE_OXYGEN_SPECIES_PATHWAY':['Genome instability', 'Apoptosis'],
    'SPERMATOGENESIS':['Proliferative signal'],
    'TGF_BETA_SIGNALING':['Insensitivity to antigrowth', 'Immune'],
    'TNFA_SIGNALING_VIA_NFKB':['Immune'],
    'UNFOLDED_PROTEIN_RESPONSE':['EMT and Metastasis', 'Insensitivity to antigrowth', 'sensitivity to growth', 'Genome instability'],
    'UV_RESPONSE_DN':['Genome instability', 'Proliferative signal'],
    'UV_RESPONSE_UP':['Genome instability', 'Apoptosis'],
    'WNT_BETA_CATENIN_SIGNALING':['Replication', 'Energetics', 'Immune', 'sensitivity to growth'],
    'XENOBIOTIC_METABOLISM':['Immune']
}

MARKER_GENES_DEFAULT = {
    "NK cells": ["GZMA", "CD7", "CCL4", "CST7", "NKG7", "GNLY", "CTSW", "CCL5", "GZMB", "PRF1"],
    "AT2": ["SEPP1", "PGC", "NAPSA", "SFTPD", "SLC34A2", "CYB5A", "MUC1", "S100A14", "SFTA2", "SFTA3"],
    "Alveolar Mφ CCL3+": ["MCEMP1", "UPP1", "HLA-DQA1", "CsAR1", "HLA-DMA", "AIF1", "LST1", "LINO1272", "MRC1", "CCL18"],
    "Suprabasal": ["PRDX2", "KRT19", "SFN", "TACSTD2", "KRT5", "LDHB", "KRT17", "KLK11", "S100A2", "SERPINB4"],
    "Basal resting": ["CYR61", "PERP", "IGFBP2", "KRT19", "KRT5", "KRT17", "KRT15", "S100A2", "LAMB3", "BCAM"],
    "EC venous pulmonary": ["VWF", "MGP", "GNG11", "RAMP2", "SPARCL1", "IGFBP7", "IFI27", "CLDN5", "ACKR1", "AQP1"],
    "CD8 T cells": ["CD8A", "CD3E", "CCL4", "CD2", "CXCR4", "GZMA", "NKG7", "IL32", "CD3D", "CCL5"],
    "EC arterial": ["SPARCL1", "SOX17", "IFI27", "TM4SF1", "AZM", "CLEC14A", "GIMAP7", "CRIP2", "CLDN5", "PECAM1"],
    "Peribronchial fibroblasts": ["IGFBP7", "COL1A2", "COL3A1", "AZM", "BGN", "DCN", "MGP", "LUM", "MFAP4", "C1S"],
    "CD4 T cells": ["CORO1A", "KLRB1", "CD3E", "LTB", "CXCR4", "IL7R", "TRAC", "IL32", "CD2", "CD3D"],
    "AT1": ["SFTA2", "CEACAM6", "FXYD3", "CAV1", "TSPAN13", "KRT7", "ADIRF", "HOPX", "AGER", "EMP2"],
    "Multiciliated (non-nasal)": ["SNTN", "FAM229B", "TMEM231", "CSorf49", "C12orf75", "GSTAT", "C11orf97", "RP11-356K23.1", "CD24", "RP11-295M3.4"],
    "Plasma cells": ["ITM2C", "TNFRSF17", "FKBP11", "IGKC", "IGHA1", "IGHG1", "CD79A", "JCHAIN", "MZB1", "ISG20"],
    "Goblet (nasal)": ["KRT7", "MUC1", "MUCSAC", "MSMB", "CP", "LM07", "LCN2", "CEACAM6", "BPIFB1", "PIGR"],
    "Club (nasal)": ["ELF3", "C19orf33", "KRT8", "KRT19", "TACSTD2", "MUC1", "S100A14", "CXCL17", "PSCA", "FAM3D"],
    "SM activated stress response": ["C11orf96", "HES4", "PLAC9", "FLNA", "KANK2", "TPM2", "PLN", "SELM", "GPX3", "LBH"],
    "Classical monocytes": ["LST1", "IL1B", "LYZ", "COTL1", "S100A9", "VCAN", "S100A8", "S100A12", "AIF1", "FCN1"],
    "Monocyte derived Mφ": ["LYZ", "ACP5", "TYROBP", "LGALS1", "CD68", "AIF1", "CTSL", "EMP3", "FCER1G", "LAPTM5"],
    "Alveolar Mφ proliferating": ["H2AFV", "STMN1", "LSM4", "GYPC", "PTTG1", "KIA40101", "FABP4", "CKS1B", "UBE2C", "HMGN2"],
    "Club (non-nasal)": ["SCGB3A1", "CYP2F1", "GSTAT", "HES4", "TSPAN8", "TFF3", "MSMB", "BPIFB1", "SCGB1A1", "PIGR"],
    "SMG serous (bronchial)": ["AZGP1", "ZG16B", "PIGR", "NDRG2", "LPO", "C6orf58", "DMBT1", "PRB3", "FAM3D", "RP11-1143G9.4"],
    "EC venous systemic": ["VWF", "MGP", "GNG11", "PLVAP", "RAMP2", "SPARCL1", "IGFBP7", "AZM", "CLEC14A", "ACKR1"],
    "Non classical monocytes": ["PSAP", "FCGR3A", "FCN1", "CORO1A", "COTL1", "FCER1G", "LAPTM5", "CTSS", "AIF1", "LST1"],
    "EC general capillary": ["EPAS1", "GNG11", "IFI27", "TM4SF1", "EGFL7", "AQP1", "VWF", "FCN3", "SPARCL1", "CLDN5"],
    "Adventitial fibroblasts": ["COL6A2", "SFRP2", "IGFBP7", "IGFBP6", "COL3A1", "C1S", "MMP2", "MGP", "SPARC", "COL1A2"],
    "Lymphatic EC mature": ["PPFIBP1", "GNG11", "RAMP2", "CCL21", "MMRN1", "IGFBP7", "SDPR", "TM4SF1", "CLDN5", "ECSCR"],
    "EC aerocyte capillary": ["EMCN", "HPGD", "IFI27", "CA4", "EGFL7", "AQP1", "IL1RL1", "SPARCL1", "SDPR", "CLDN5"],
    "Smooth muscle": ["PRKCDBP", "NDUF4AL2", "MYL9", "ACTA2", "MGP", "CALD1", "TPM1", "TAGLN", "IGFBP7", "TPM2"],
    "Alveolar fibroblasts": ["LUM", "COL6A1", "CYR61", "C1R", "COL1A2", "MFAP4", "A2M", "C1S", "ADH1B", "GPX3"],
    "Multiciliated (nasal)": ["RP11-356K23.1", "EFHC1", "CAPS", "ROPN1L", "RSPH1", "C9orf116", "TMEM190", "DNAL1", "PIFO", "ODF3B"],
    "Goblet (bronchial)": ["MUC5AC", "MSMB", "PI3", "MDK", "ANKRD36C", "TFF3", "PIGR", "SAA1", "CP", "BPIFB1"],
    "Neuroendocrine": ["UCHL1", "TFF3", "APOA1BP", "CLDN3", "SEC11C", "NGFRAP1", "SCGS", "HIGD1A", "PHGR1", "CD24"],
    "Lymphatic EC differentiating": ["AKAP12", "TFF3", "SDPR", "CLDN5", "TCF4", "TFPI", "TIMP3", "GNG11", "CCL21", "IGFBP7"],
    "DC2": ["ITGB2", "LAPTM5", "HLA-DRB1", "HLA-DPB1", "HLA-DPA1", "HLA-DMB", "HLA-DQB1", "HLA-DQA1", "HLA-DMA", "LST1"],
    "Transitional Club AT2": ["CXCL17", "C16orf89", "RNASE1", "KRT7", "SCGB1A1", "PIGR", "SCGB3A2", "KLK11", "SFT41P", "FOLR1"],
    "DC1": ["HLA-DPA1", "CPNE3", "CORO1A", "CPVL", "C1orf54", "WDFY4", "LSP1", "HLA-DQB1", "HLA-DQA1", "HLA-DMA"],
    "Myofibroblasts": ["CALD1", "CYR61", "TAGLN", "MT1X", "PRELP", "TPM2", "GPX3", "CTGF", "IGFBP5", "SPARCL1"],
    "B cells": ["CD69", "CORO1A", "LIMD2", "BANK1", "LAPTM5", "CXCR4", "LTB", "CD79A", "CD37", "MS4A1"],
    "Mast cells": ["VWASA", "RGS13", "C1orf186", "HPGD5", "CPA3", "GATA2", "MS4A2", "KIT", "TPSAB1", "TPSB2"],
    "Interstitial Mφ perivascular": ["MRC1", "RNASE1", "FGL2", "RNASE6", "HLA-DPA1", "GPR183", "CD14", "HLA-DPB1", "MS4A6A", "AIF1"],
    "SMG mucous": ["FKBP11", "TCN1", "GOLM1", "TFF3", "PIGR", "KLK11", "MARCKSL1", "CRACR2B", "SELM", "MSMB"],
    "AT2 proliferating": ["CDK1", "LSM3", "CKS1B", "EIF1AX", "UBE2C", "MRPL14", "PRC1", "CENPW", "EMP2", "DHFR"],
    "Goblet (subsegmental)": ["MDK", "MUC5B", "SCGB1A1", "CP", "C3", "TSPAN8", "TFF3", "MSMB", "PIGR", "BPIFB1"],
    "Pericytes": ["MYL9", "SPARC", "SPARCL1", "IGFBP7", "COL4A1", "GPX3", "PDGFRB", "CALD1", "COX4I2", "TPM2"],
    "SMG duct": ["PIP", "ZG16B", "PIGR", "SAA1", "MARCKSL1", "ALDH1A3", "SELM", "LTF", "RARRES1", "AZGP1"],
    "Mesothelium": ["CEBPD", "LINCO1133", "MRPL33", "UPK3B", "CFB", "SEPP1", "EID1", "HP", "CUX1", "MRPS21"],
    "SMG serous (nasal)": ["ZG16B", "MUC7", "C6orf58", "PRB3", "LTF", "LYZ", "PRR4", "AZGP1", "PIGR", "RP11-1143G9.4"],
    "Ionocyte": ["FOXI1", "ATP6V1A", "GOLM1", "TMEM61", "SEC11C", "SCNN1B", "ASCL3", "CLCNKB", "HEPACAM2", "CD24"],
    "Alveolar Mφ MT-positive": ["GSTO1", "LGALS1", "CTSZ", "MT2A", "APOC1", "CTSL", "UPP1", "CCL18", "FABP4", "MT1X"],
    "Fibromyocytes": ["NEXN", "ACTG2", "LMOD1", "IGFBP7", "PPP1R14A", "DES", "FLNA", "TPM2", "PLN", "SELM"],
    "Deuterosomal": ["RSPH9", "PIFO", "RUVBL2", "C11orf88", "FAM183A", "MORN2", "SAXO2", "CFAP126", "FAM229B", "C5orf49"],
    "Tuft": ["MUC20", "KHDRBS1", "ZNF428", "BIK", "CRYM", "LRMP", "HES6", "KIT", "AZGP1", "RASSF6"],
    "Plasmacytoid DCs": ["IL3RA", "TCF4", "LTB", "GZMB", "JCHAIN", "ITM2C", "IRF8", "PLD4", "IRF7", "C12orf75"],
    "T cells proliferating": ["TRAC", "HMGN2", "IL32", "CORO1A", "ARHGDIB", "STMN1", "RAC2", "IL2RG", "HMGB2", "CD3D"],
    "Subpleural fibroblasts": ["SERPING1", "C1R", "COL1A2", "NNMT", "COL3A1", "MT1E", "MT1X", "PLA2G2A", "SELM", "MT1M"],
    "Lymphatic EC proliferating": ["S100A16", "TUBB", "HMGN2", "COX20", "LSM2", "HMGN1", "ARPC1A", "ECSCR", "EID1", "MARCKS"],
    "Migratory DCs": ["IL2RG", "HLA-DRBS", "TMEM176A", "BIRC3", "TYMP", "COL22", "SYNGR2", "CD83", "LSP1", "HLA-DOA1"],
    "Alveolar macrophages": ["MS4A7", "C1QA", "HLA-DQB1", "HLA-DMA", "HLA-DPB1", "HLA-DPA1", "ACP5", "C1QC", "CTSS", "HLA-DQA1"],
}

CELLTYPE_MAP = {
    'AT2': 0, 'Goblet (subsegmental)': 1, 'EC general capillary': 2, 'CD8 T cells': 3, 
    'Club (nasal)': 4, 'Club (non-nasal)': 5, 'AT1': 6, 'Plasma cells': 7, 'Pericytes': 8, 
    'Mesothelium': 9, 'Multiciliated (non-nasal)': 10, 'Mast cells': 11, 'Basal resting': 12, 
    'Lymphatic EC differentiating': 13, 'Monocyte derived Mφ': 14, 'Non classical monocytes': 15, 
    'Suprabasal': 16, 'Multiciliated (nasal)': 17, 'Alveolar macrophages': 18, 
    'Transitional Club AT2': 19, 'Peribronchial fibroblasts': 20, 'Goblet (nasal)': 21, 
    'SMG serous (nasal)': 22, 'NK cells': 23, 'Alveolar Mφ MT-positive': 24, 
    'Classical monocytes': 25, 'B cells': 26, 'EC venous pulmonary': 27, 'CD4 T cells': 28,  
    'DC2': 29, 'T cells proliferating': 30, 'Smooth muscle': 31, 'Adventitial fibroblasts': 32, 
    'Plasmacytoid DCs': 33, 'Lymphatic EC proliferating': 34, 'EC aerocyte capillary': 35, 
    'Lymphatic EC mature': 36, 'Subpleural fibroblasts': 37, 'Migratory DCs': 38, 'Alveolar fibroblasts': 39,
    'Alveolar Mφ CCL3+': 40, 'EC arterial': 41, 'SM activated stress response': 42, 'Alveolar Mφ proliferating': 43,
    'SMG serous (bronchial)': 44, 'EC venous systemic': 45, 'Goblet (bronchial)': 46, 'Neuroendocrine': 47, 'DC1': 48,
    'Myofibroblasts': 49, 'Interstitial Mφ perivascular': 50, 'SMG mucous': 51, 'AT2 proliferating': 52, 'SMG duct': 53,
    'Ionocyte': 54, 'Fibromyocytes': 55, 'Deuterosomal': 56, 'Tuft': 57
}

class OncoTerrain:
    def __init__(self, adata=None, *, sample_key: str = "sample", 
                 user_marker_genes: Optional[Dict[str, Iterable[str]]] = None, 
                 user_celltype_vectors: Optional[pd.DataFrame] = None):
        logging.info("Initializing OncoTerrain")
        logging.basicConfig(level=logging.INFO)
        self.adata = adata
        self.sample_key = sample_key
        self.user_marker_genes = user_marker_genes
        self.user_celltype_vectors = user_celltype_vectors
        if self.adata is not None and self.user_celltype_vectors is not None:
            missing = set(self.adata.obs_names) - set(self.user_celltype_vectors.index)
            inter = self.user_celltype_vectors.loc[self.user_celltype_vectors.index.intersection(self.adata.obs_names)].copy()
            inter = inter.apply(pd.to_numeric, errors="coerce").fillna(0.0)
            inter.columns = [f"ctv_{c}" if not str(c).startswith("ctv_") else str(c) for c in inter.columns]
            self.adata.obs = self.adata.obs.join(inter, how="left")
        self.conditions = CONDITIONS

    def _preprocessing(self, min_genes: int = 200, min_cells: int = 10):
        self.adata.obs.index = self.adata.obs.index.astype(str)
        self.adata.var.index = self.adata.var.index.astype(str)
        self.adata.var['mt'] = self.adata.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(self.adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
        sc.pp.filter_cells(self.adata, min_genes=min_genes)
        sc.pp.filter_genes(self.adata, min_cells=min_cells)
        if sp.issparse(self.adata.X):
            if hasattr(self.adata.X, "data"):
                self.adata.X.data = np.nan_to_num(self.adata.X.data)
        else:
            self.adata.X = np.nan_to_num(self.adata.X)
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)
        return self.adata

    def _annotate_clusters_by_markers(self, adata, cluster_key, marker_dict):
        cluster2celltype = {}
        for cluster in adata.obs[cluster_key].unique():
            subset_adata = adata[adata.obs[cluster_key] == cluster]
            if subset_adata.n_obs == 0:
                continue
            best_celltype = None
            best_score = -np.inf
            for celltype, markers in marker_dict.items():
                valid_markers = list(set(markers).intersection(adata.var_names))
                if len(valid_markers) == 0:
                    continue
                avg_expr = subset_adata[:, valid_markers].X.mean()
                if avg_expr > best_score:
                    best_score = avg_expr
                    best_celltype = celltype
            cluster2celltype[cluster] = best_celltype
        return cluster2celltype

    def _ct_annotation(self, *, pca_n_comps=None, neighbors_n_pcs=None, neighbors_k=None, leiden_res=20.00):
        os.makedirs("../figures", exist_ok=True)
        os.makedirs("../data", exist_ok=True)

        max_rank = max(0, min(self.adata.n_obs, self.adata.n_vars) - 1)
        n_comps = int(pca_n_comps) if pca_n_comps is not None else max(2, min(50, max_rank))
        sc.tl.pca(self.adata, n_comps=n_comps, svd_solver='arpack')

        n_pcs_avail = int(self.adata.obsm['X_pca'].shape[1])
        if n_pcs_avail < 2:
            raise ValueError(
                f"Too few PCs after preprocessing (got {n_pcs_avail}). "
                "Consider lowering filtering thresholds or increasing pca_n_comps."
            )

        n_pcs_use = int(neighbors_n_pcs) if neighbors_n_pcs is not None else min(40, n_pcs_avail)
        if n_pcs_use > n_pcs_avail:
            raise ValueError(f"Requested neighbors_n_pcs={n_pcs_use} > available PCs ({n_pcs_avail}).")

        k = int(neighbors_k) if neighbors_k is not None else min(15, max(2, self.adata.n_obs - 1))
        k = max(2, min(k, max(2, self.adata.n_obs - 1)))

        sc.pp.neighbors(self.adata, n_neighbors=k, n_pcs=n_pcs_use, use_rep='X_pca')
        sc.tl.umap(self.adata)

        cluster_key = f"leiden_res_{float(leiden_res):.2f}"
        sc.tl.leiden(self.adata, key_added=cluster_key, resolution=float(leiden_res), flavor="igraph")

        marker_genes = self.user_marker_genes if self.user_marker_genes is not None else MARKER_GENES_DEFAULT
        self.adata.var_names = self.adata.var_names.str.upper()
        self.adata.var_names_make_unique()
        marker_genes = {k: [gene.upper() for gene in v] for k, v in marker_genes.items()}
        filtered_marker_genes = {k: [gene for gene in v if gene in self.adata.var_names] for k, v in marker_genes.items()}

        seaborn_palette = sns.color_palette("husl", 20)
        annotation = self._annotate_clusters_by_markers(self.adata, cluster_key, filtered_marker_genes)
        self.adata.obs[f"{cluster_key}_celltype"] = self.adata.obs[cluster_key].map(annotation)
        celltype_key = f"{cluster_key}_celltype"

        sc.pl.umap(
            self.adata,
            color=cluster_key,
            title=f"Leiden Clusters (res={float(leiden_res):.2f})",
            palette=seaborn_palette,
            show=False,
            save=f"_umap_leiden_res_{float(leiden_res):.2f}.png",
        )
        sc.pl.umap(
            self.adata,
            color=celltype_key,
            title=f"Annotated Cell Types (res={float(leiden_res):.2f})",
            palette=seaborn_palette,
            legend_loc="right margin",
            show=False,
            save=f"_umap_celltype_res_{float(leiden_res):.2f}.png",
        )

        for col in self.adata.obs.select_dtypes(["category"]).columns:
            self.adata.obs[col] = self.adata.obs[col].astype(str)

        self.adata.uns.setdefault("oncoterrain_params", {})
        self.adata.uns["oncoterrain_params"].update(
            dict(
                pca_n_comps=n_comps,
                neighbors_n_pcs=n_pcs_use,
                neighbors_k=k,
                leiden_res=float(leiden_res),
                n_pcs_avail=n_pcs_avail,
                n_obs=int(self.adata.n_obs),
                n_vars=int(self.adata.n_vars),
            )
        )
        return self.adata

    def _add_and_aggregate_module_scores(self, adata, gmt_file):
        gene_sets = gp.get_library(str(gmt_file))
        adata_genes = [gene.upper() for gene in adata.var_names]
        for pathway, genes in gene_sets.items():
            genes = [g.upper() for g in genes]
            genes_in_adata = [gene for gene in genes if gene in adata_genes]
            if len(genes_in_adata) > 0:
                sc.tl.score_genes(adata, genes_in_adata, score_name=pathway)
        return adata

    def _hp_calculation(self):
        gmt_root = resources.files("OncoTerrain") / "HallmarkPathGMT"
        gmt_files = [p for p in gmt_root.iterdir() if p.name.endswith(".gmt")]
        for gmt in gmt_files:
            with resources.as_file(gmt) as p:
                self.adata = self._add_and_aggregate_module_scores(self.adata, Path(p))
        return self.adata

    def _process_one_sample(self, adata_s, outdir, *, pca_n_comps=None, min_cells=3, min_genes=200,
                            neighbors_n_pcs=None, neighbors_k=None, leiden_res=20.00):
        _orig_adata = self.adata
        try:
            self.adata = adata_s.copy()

            self.adata = self._preprocessing(min_cells=min_cells, min_genes=min_genes)
            self.adata = self._ct_annotation(
                pca_n_comps=pca_n_comps,
                neighbors_n_pcs=neighbors_n_pcs,
                neighbors_k=neighbors_k,
                leiden_res=leiden_res,
            )
            self.adata = self._hp_calculation()

            with resources.as_file(resources.files("OncoTerrain") / "OncoTerrain.joblib") as p:
                model_path = Path(p)
            self.model_bundle = joblib.load(model_path)
            self.OncoTerrain = self.model_bundle['model']
            self.model_features = list(self.model_bundle['features'])

            meta_data = self.adata.obs.copy()
            columns = [
                'disease','sample','source','tissue','n_genes','batch',
                'n_genes_by_counts','total_counts','total_counts_mt','pct_counts_mt',
                'leiden_res_0.10','leiden_res_1.00','leiden_res_5.00','leiden_res_10.00','leiden_res_20.00',
                'leiden_res_0.10_celltype','leiden_res_1.00_celltype','leiden_res_5.00_celltype','leiden_res_10.00_celltype',
                'tumor_stage','project'
            ]
            for col in columns:
                if col in meta_data.columns:
                    meta_data = meta_data.drop(col, axis=1)

            meta_data.columns = meta_data.columns.str.replace('^HALLMARK_', '', regex=True)

            ctkey = f'leiden_res_{float(leiden_res):.2f}_celltype'
            if ctkey not in meta_data.columns:
                meta_data[ctkey] = 'Unknown'

            reverse_celltype_map = {v: k for k, v in CELLTYPE_MAP.items()}
            reverse_celltype_map[-1] = 'Unknown'

            mapped = meta_data[ctkey].astype(str).map(CELLTYPE_MAP)
            meta_data[ctkey] = mapped.fillna(-1).astype(int)
            celltype_labels = meta_data[ctkey].map(reverse_celltype_map)

            meta_data_X = meta_data.drop([ctkey], axis=1)
            missing_cols = set(self.model_features) - set(meta_data_X.columns)
            for col in missing_cols:
                meta_data_X[col] = 0.0
            meta_data_X = meta_data_X[self.model_features].astype(float)

            scaler = MinMaxScaler()
            meta_data_X.loc[:, :] = scaler.fit_transform(meta_data_X.values)

            y_val_pred = self.OncoTerrain.predict(meta_data_X.values)
            class_labels = {0: 'Normal-like', 1: 'Pre-malignant', 2: 'Tumor-like'}
            self.adata.obs['oncoterrain_class'] = [class_labels[int(label)] for label in y_val_pred]
            self.adata.obs['celltype_readable'] = celltype_labels.values

            umap_model = umap.UMAP(n_neighbors=50, min_dist=0.05, metric='euclidean', random_state=42)
            X_umap = umap_model.fit_transform(meta_data_X)

            colors = plt.cm.tab20(np.linspace(0, 1, len(pd.unique(self.adata.obs['celltype_readable']))))
            cell_type_color_map = dict(zip(pd.unique(self.adata.obs['celltype_readable']), colors))
            color_palette = {0: "#84A970", 1: "#E4C282", 2: "#FF8C00"}

            plt.figure(figsize=(20, 10))
            plt.subplot(1, 2, 1)
            sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=y_val_pred, palette=color_palette, s=10, legend="brief")
            plt.title('UMAP Projection - Predicted Labels', fontsize=14)

            plt.subplot(1, 2, 2)
            sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=self.adata.obs['celltype_readable'], s=10, legend="brief")
            plt.title('UMAP Projection - Cell Type', fontsize=14)

            for ax in plt.gcf().axes:
                ax.grid(False)
                for side in ('top', 'right', 'bottom', 'left'):
                    ax.spines[side].set_linewidth(2)

            outdir.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(outdir / "umap.png", dpi=300, bbox_inches='tight')
            plt.close()

            self.adata.write_h5ad(outdir / "OncoTerrain_annotated.h5ad")
            return self.adata

        finally:
            self.adata = _orig_adata

    def inferencing(self, save_path, save_adata=True, *, pca_n_comps=None, neighbors_n_pcs=None, 
                    neighbors_k=None, leiden_res=20.00, min_cells=3, min_genes=200):
        """
        Run OncoTerrain across samples, with user-tunable dimensionality and clustering.

        Parameters
        ----------
        save_path : str or Path
            Directory where per-sample outputs and the concatenated .h5ad (if requested) are written.
        save_adata : bool, default True
            Whether to write the concatenated annotated AnnData to disk.
        pca_n_comps : int or None
            Number of PCs to compute. If None, auto = min(50, rank) with lower bound of 2.
        neighbors_n_pcs : int or None
            Number of PCs to use for kNN graph. If None, auto = min(40, available PCs).
        neighbors_k : int or None
            Number of neighbors (k) for the kNN graph. If None, auto = min(15, n_obs-1), bounded to >=2.
        leiden_res : float, default 20.00
            Resolution for Leiden clustering.
        """
        if self.adata is None:
            raise AttributeError("AnnData is None")

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        if self.sample_key not in self.adata.obs.columns:
            raise KeyError(f"'{self.sample_key}' not found in adata.obs")

        col = self.adata.obs[self.sample_key]
        col = col.astype(str).str.strip()
        self.adata.obs[self.sample_key] = col

        vc = col.value_counts(dropna=False)

        annotated_list = []
        skipped = []

        for sample_id in col.unique():
            mask = (self.adata.obs[self.sample_key] == sample_id)

            n = int(mask.sum())
            if n == 0:
                logging.warning(
                    f"[OncoTerrain] Skipping sample '{sample_id}': 0 cells after filtering."
                )
                skipped.append(sample_id)
                continue

            safe_id = (
                str(sample_id)
                .replace(os.sep, "_")
                .replace(os.altsep or "", "_")
                .replace(" ", "_")
            )
            outdir = save_path / safe_id

            adata_s = self.adata[mask, :].copy()

            ann_s = self._process_one_sample(
                adata_s,
                outdir,
                pca_n_comps=pca_n_comps,
                neighbors_n_pcs=neighbors_n_pcs,
                neighbors_k=neighbors_k,
                leiden_res=leiden_res,
                min_cells=min_cells,
                min_genes=min_genes,
            )
            ann_s.obs[self.sample_key] = sample_id
            annotated_list.append(ann_s)

        if not annotated_list:
            raise ValueError(
                "No samples contained any cells after filtering. "
                f"Seen samples (pre-filter): {list(vc.index.astype(str))[:10]} "
                f"→ counts: {list(map(int, vc.values))[:10]}"
            )

        try:
            adata_concat = annotated_list[0].concatenate(
                *annotated_list[1:],
                batch_key="__batch__",
                batch_categories=[str(a.obs[self.sample_key].unique()[0]) for a in annotated_list],
            )
        except Exception:
            adata_concat = ad.concat(
                annotated_list,
                label="__batch__",
                keys=[str(a.obs[self.sample_key].unique()[0]) for a in annotated_list],
            )

        if save_adata:
            adata_concat.write_h5ad(save_path / "OncoTerrain_annotated_all_samples.h5ad")

        if skipped:
            logging.info(f"[OncoTerrain] Skipped {len(skipped)} sample(s) with 0 cells: {skipped}")

        return adata_concat

    def _safe_name(self, s: str) -> str: 
        s = str(s).strip() 
        for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']: 
            s = s.replace(ch, '_') 
        return s or "unnamed"

    def _circ_heatmap_plotter(self, meta_data, save_path, *, groupby_column='project',
                          sample_id=None, scale_range=(0, 100), stage_label=None):
        md = meta_data.copy()
        cond_keys = set(CONDITIONS.keys())

        if groupby_column in md.columns:
            if sample_id is None:
                uniq = md[groupby_column].dropna().unique()
                if len(uniq) != 1:
                    raise ValueError(
                        f"Provide sample_id. Found {len(uniq)} unique values in '{groupby_column}': {list(map(str, uniq))[:10]}..."
                    )
                sample_id = str(uniq[0])
            md = md[md[groupby_column].astype(str) == str(sample_id)]
            if md.empty:
                raise ValueError(f"No rows found for {groupby_column} == '{sample_id}'.")

        print(f"[OncoTerrain] Rendering circular heatmap for {groupby_column} == '{sample_id}'")

        rename_map = {}
        for c in md.columns:
            cu = str(c).upper()
            if cu.startswith('HALLMARK_'):
                rename_map[c] = cu.replace('HALLMARK_', '')
            elif cu in cond_keys:
                rename_map[c] = cu
        if rename_map:
            md = md.rename(columns=rename_map)

        numeric_cols = md.select_dtypes(include=[np.number]).columns.tolist()
        hallmark_cols = [c for c in numeric_cols if c in cond_keys]
        if len(hallmark_cols) == 0:
            raise ValueError(
                "No numeric hallmark columns found after renaming. "
                "Ensure you have HALLMARK_* scores or bare hallmark columns."
            )

        pb = md[hallmark_cols].mean(axis=0).to_frame(name='mean_score').reset_index()
        pb = pb.rename(columns={'index': 'hallmark'})
        pb['Condition'] = pb['hallmark'].map(lambda x: ', '.join(CONDITIONS.get(x, []))).str.lower()

        pb = pb.dropna(subset=['mean_score', 'Condition'])
        pb = pb[pb['Condition'].str.strip().ne('')]
        if pb.empty:
            raise ValueError("No usable hallmarks after dropping NaN mean scores or blank conditions.")

        expanded = pb[['hallmark', 'mean_score', 'Condition']].copy()
        expanded = expanded.assign(Condition=expanded['Condition'].str.split(', ')).explode('Condition')
        expanded = expanded[expanded['Condition'].str.strip().ne('')]
        if expanded.empty:
            raise ValueError("No data to plot after expanding Conditions.")
        expanded = expanded.sort_values(by=["Condition", "hallmark"], kind="stable")

        VALUES = expanded["mean_score"].astype(float).to_numpy()
        LABELS = expanded['hallmark'].astype(str).to_numpy()
        GROUPS = expanded["Condition"].astype(str).to_numpy()

        if np.isnan(VALUES).all():
            raise ValueError("All mean scores are NaN after preprocessing.")

        mask = ~np.isnan(VALUES)
        VALUES = VALUES[mask]
        LABELS = LABELS[mask]
        GROUPS = GROUPS[mask]

        if VALUES.size == 0:
            raise ValueError("No values to plot after removing NaNs.")

        scaler = MinMaxScaler(feature_range=scale_range)
        vmin, vmax = np.nanmin(VALUES), np.nanmax(VALUES)
        if vmax - vmin == 0:
            VALUES = np.full_like(VALUES, (scale_range[0] + scale_range[1]) / 2.0, dtype=float)
        else:
            VALUES = scaler.fit_transform(VALUES.reshape(-1, 1)).ravel()

        if np.isnan(VALUES).any():
            raise RuntimeError("Unexpected NaNs in VALUES after scaling.")

        n = int(VALUES.size)
        ANGLES = np.linspace(0, 2 * np.pi, num=max(1, n), endpoint=False)
        WIDTH = (2 * np.pi) / max(1, n)

        cmap = plt.get_cmap("tab20", 50)
        unique_labels = np.unique(LABELS)
        color_map = {label: cmap(i % 20) for i, label in enumerate(unique_labels)}
        COLORS = [color_map.get(label, cmap(0)) for label in LABELS]

        fig, ax = plt.subplots(figsize=(60, 20), subplot_kw={"projection": "polar"})
        ax.set_theta_offset(0)
        ax.set_ylim(0, float(scale_range[1]))
        ax.set_frame_on(False)
        ax.xaxis.grid(False)
        ax.yaxis.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

        if ANGLES.size != n or len(COLORS) != n or VALUES.size != n:
            raise RuntimeError(
                f"Layout mismatch: angles={ANGLES.size}, colors={len(COLORS)}, values={VALUES.size}"
            )

        ax.bar(ANGLES, VALUES, width=WIDTH, color=COLORS, edgecolor="white", linewidth=2)

        if n > 0:
            from itertools import groupby
            idxs = np.arange(n)
            for cond, run in groupby(zip(idxs, GROUPS), key=lambda t: t[1]):
                run_idxs = [i for (i, _) in run]
                if len(run_idxs) == 0:
                    continue
                theta = np.mean(ANGLES[run_idxs])
                rot = np.rad2deg(theta)
                if 90 < rot < 270:
                    rot += 180
                ax.text(theta, float(scale_range[1]) * 0.95, str(cond), color="#333",
                        fontsize=14, fontweight="bold", ha="center", va="center",
                        rotation=rot)

        if unique_labels.size > 0:
            legend_handles = [mpatches.Patch(color=color_map[label], label=label) for label in unique_labels]
            plt.legend(
                handles=legend_handles,
                loc="center left",
                bbox_to_anchor=(1.02, 0.75),
                fontsize=12,
                title="Pathways",
                frameon=False,
            )

        title_suffix = f" • {stage_label}" if stage_label else ""
        ax.set_title(f"Circular Heatmap — {groupby_column}={sample_id}{title_suffix}",
                    pad=20, fontsize=18, fontweight="bold")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def circular_heatmap(self, meta_data, save_path, *, groupby_column='project',
                        scale_range=(0, 100), stage_label=None):
        if groupby_column not in meta_data.columns:
            raise KeyError(f"'{groupby_column}' not found in meta_data columns")

        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        col = meta_data[groupby_column].astype(str).str.strip()
        sample_ids = [sid for sid in col.unique() if sid and sid.lower() != 'nan']

        generated, skipped = [], []

        for sid in sample_ids:
            out = save_dir / f"{self._safe_name(sid)}_circular_heatmap.png"
            try:
                print(f"[OncoTerrain] Processing sample_id: '{sid}'")
                self._circ_heatmap_plotter(
                    meta_data=meta_data,
                    save_path=str(out),
                    groupby_column=groupby_column,
                    sample_id=sid,
                    scale_range=scale_range,
                    stage_label=stage_label
                )
                generated.append(str(out))
            except Exception as e:
                msg = f"[OncoTerrain] Skipping sample '{sid}': {e}"
                warnings.warn(msg, category=RuntimeWarning, stacklevel=2)
                skipped.append((sid, str(e)))
                continue

        if not generated:
            raise ValueError(
                f"No circular heatmaps were generated. Checked {len(sample_ids)} sample(s); "
                f"all were skipped. First reason: {skipped[0][1] if skipped else 'unknown'}"
            )

        if skipped:
            logging.info(f"[OncoTerrain] Skipped {len(skipped)} sample(s): {skipped}")

        return generated
