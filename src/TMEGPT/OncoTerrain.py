import scanpy as sc
import logging
import os
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import umap
import gseapy as gp
import numpy as np

class OncoTerrain:
    BASE_DIR = Path.cwd()

    def __init__(self, adata=None):
        logging.info(f"Loading trained model from {self.BASE_DIR}")
        logging.basicConfig(level=logging.INFO)

        self.adata = adata
    
    def __preprocessing(self):
        """
        After merging, perform normalization, log1p, and compute QC metrics on the full union.
        """

        self.adata.obs.index = self.adata.obs.index.astype(str)
        self.adata.var.index = self.adata.var.index.astype(str)

        # Compute percent mitochondrial genes
        self.adata.var['mt'] = self.adata.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(
            self.adata,
            qc_vars=['mt'],
            percent_top=None,
            log1p=False,
            inplace=True
        )

        sc.pp.filter_cells(self.adata, min_genes=200)
        sc.pp.filter_genes(self.adata, min_cells=10)

        self.adata.X.data = np.nan_to_num(self.adata.X.data)

        # Normalize total counts per cell
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)


        logging.debug(
            "Post-merge normalization/log1p completed. Shape: %s", 
            self.adata.shape
        )

        logging.info(f"Current AnnData shape: {self.adata.shape}")
        logging.info(f"AnnData obs head:\n{self.adata.obs.head()}")

        return self.adata
    
    def __annotate_clusters_by_markers(self, adata, cluster_key, marker_dict):
        cluster2celltype = {}
        for cluster in adata.obs[cluster_key].unique():
            subset_adata = adata[adata.obs[cluster_key] == cluster]
            if subset_adata.n_obs == 0:
                logging.warning("Cluster %s has no cells. Skipping annotation.", cluster)
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

    def __ct_annotation(self):
        logging.debug("Starting ct_annotation")
        os.makedirs("../figures", exist_ok=True)
        os.makedirs("../data", exist_ok=True)
        sc.tl.pca(self.adata, svd_solver='arpack')
        sc.pp.neighbors(self.adata, n_neighbors=15, n_pcs=40, use_rep='X_pca')
        sc.tl.umap(self.adata)

        cluster_key = f"leiden_res_20.00"
        sc.tl.leiden(self.adata, key_added=cluster_key, resolution=20.00, flavor="igraph")
        marker_genes = {
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
        self.adata.var_names = self.adata.var_names.str.upper()
        self.adata.var_names_make_unique()
        marker_genes = {k: [gene.upper() for gene in v] for k, v in marker_genes.items()}
        missing_genes = [gene for genes in marker_genes.values() for gene in genes if gene not in self.adata.var_names]
        logging.debug("Missing genes: %s", missing_genes)
        filtered_marker_genes = {k: [gene for gene in v if gene in self.adata.var_names] for k, v in marker_genes.items()}
        num_categories = 20
        seaborn_palette = sns.color_palette("husl", num_categories)
        logging.debug(f"Annotating clusters with {cluster_key}")
        annotation = self.__annotate_clusters_by_markers(self.adata, cluster_key, filtered_marker_genes)
        self.adata.obs[f"{cluster_key}_celltype"] = self.adata.obs[cluster_key].map(annotation)

        celltype_key = f"{cluster_key}_celltype"
        sc.pl.umap(
            self.adata,
            color=cluster_key,
            title=f"Leiden Clusters (res=20.00)",
            palette=seaborn_palette,
            show=False,
            save=f"_umap_leiden_res_20.00.png"
        )
        sc.pl.umap(
            self.adata,
            color=celltype_key,
            title=f"Annotated Cell Types (res=20.00)",
            palette=seaborn_palette,
            legend_loc="right margin",
            show=False,
            save=f"_umap_celltype_res_20.00.png"
        )

        for col in self.adata.obs.select_dtypes(["category"]).columns:
            self.adata.obs[col] = self.adata.obs[col].astype(str)
        logging.debug("ct_annotation completed")
        return self.adata

    def ___add_and_aggregate_module_scores(self, adata, gmt_file):
            """
            For the adata object add hallmark pathway supplied by gmt_file to the metadata for each cell.
            """
            gene_sets = gp.get_library(str(gmt_file))
            adata_genes = [gene.upper() for gene in adata.var_names]
            
            for pathway, genes in gene_sets.items():
                genes = [g.upper() for g in genes]
                genes_in_adata = [gene for gene in genes if gene in adata_genes]
                
                if len(genes_in_adata) > 0:
                    logging.info(f"Calculating module score for pathway: {pathway}")
                    logging.info(f"Genes in pathway (after matching): {genes_in_adata}")
                    sc.tl.score_genes(adata, genes_in_adata, score_name=pathway)
                    if pathway in adata.obs.columns:
                        score_col = pathway
                    elif f"{pathway}_score" in adata.obs.columns:
                        score_col = f"{pathway}_score"
                    else:
                        score_col = None
                    if score_col:
                        logging.info(f"Module score for {pathway} added to adata.obs as {score_col}")
                    else:
                        logging.info(f"Module score for {pathway} was not added to adata.obs")
                else:
                    logging.info(f"No genes found for pathway {pathway} in AnnData object")
            return adata

    def __hp_calculation(self):
        """
        Run code to generate hallmark pathway scores.
        """
        logging.debug("starting hp_calculation")
        gmt_dir = self.BASE_DIR / "HallmarkPathGMT"
        gmt_files = list(gmt_dir.glob("*.gmt"))
        for gmt_file in gmt_files:
            logging.debug(f"Processing GMT file: {gmt_file}")
            self.adata = self.___add_and_aggregate_module_scores(self.adata, gmt_file)
        logging.debug("hp_calculation completed")
        return self.adata

    def inferencing(self, save_path, save_adata=True):
        logging.info(f"Preprocessing dataset with shape: {self.adata.shape} and obs names: {self.adata.obs_names[:5]}")
        self.adata = self.__preprocessing()
        logging.info(f"After preprocessing dataset with shape: {self.adata.shape} and obs names: {self.adata.obs_names[:5]}")
        self.adata = self.__ct_annotation()
        logging.info(f"After ct annotation dataset with shape: {self.adata.shape} and obs names: {self.adata.obs_names[:5]}")
        self.adata = self.__hp_calculation()
        logging.info(f"After hp calculation dataset with shape: {self.adata.shape} and obs names: {self.adata.obs_names[:5]}")

        self.model_bundle = joblib.load("OncoTerrain.joblib")
        self.OncoTerrain = self.model_bundle['model']
        self.model_features = self.model_bundle['features']

        meta_data = self.adata.obs.copy()

        columns= ['disease', 'sample', 'source', 'tissue', 'n_genes', 'batch', 'n_genes_by_counts', 'total_counts', 'total_counts_mt', 
                'pct_counts_mt', 'leiden_res_0.10', 'leiden_res_1.00', 'leiden_res_5.00', 'leiden_res_10.00', 'leiden_res_20.00', 
                'leiden_res_0.10_celltype', 'leiden_res_1.00_celltype', 'leiden_res_5.00_celltype', 'leiden_res_10.00_celltype', 'tumor_stage', 'project']
        
        for col in columns:
            if col in meta_data.columns:
                meta_data = meta_data.drop(col, axis=1)
        
        logging.info("Columns left after dropping specified columns: %s", meta_data.columns.tolist())

        meta_data.columns = meta_data.columns.str.replace('^HALLMARK_', '', regex=True)

        celltype_map = {
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

        # Create reverse mapping for human-readable labels
        reverse_celltype_map = {v: k for k, v in celltype_map.items()}
        reverse_celltype_map[-1] = 'Unknown'  # For unmapped cell types

        mapped = meta_data['leiden_res_20.00_celltype'].astype(str).map(celltype_map)

        if mapped.isnull().any():
            missing = meta_data['leiden_res_20.00_celltype'][mapped.isnull()].unique()
            logging.warning(f"The following celltypes were not in the mapping and will be set to -1: {missing}")

        meta_data['leiden_res_20.00_celltype'] = mapped.fillna(-1).astype(int)
        celltype_data = meta_data['leiden_res_20.00_celltype'].copy()
        
        # Convert numeric cell types back to human-readable labels for plotting
        celltype_labels = celltype_data.map(reverse_celltype_map)

        meta_data_X = meta_data.drop(['leiden_res_20.00_celltype'], axis=1)

        missing_cols = set(self.model_features) - set(meta_data_X.columns)
        for col in missing_cols:
            logging.info("Adding missing column: %s", col)
            meta_data_X[col] = 0  # Add missing columns as 0

        meta_data_X = meta_data_X[self.model_features]

        columns_to_exclude = ['leiden_res_20.00_celltype']
        columns_to_scale = [col for col in meta_data_X.columns if col not in columns_to_exclude]

        scaler = MinMaxScaler()
        meta_data_X[columns_to_scale] = scaler.fit_transform(meta_data_X[columns_to_scale])

        meta_data_X = meta_data_X.astype(float)

        logging.info("meta_data_X shape: %s", meta_data_X.shape)
        logging.info("Columns in meta_data_X: %s", meta_data_X.columns.tolist())

        y_val_pred = self.OncoTerrain.predict(meta_data_X.values)
        color_palette = {0: "#84A970", 1: "#E4C282", 2: "#FF8C00"}

        # Use a more manageable color palette - seaborn/matplotlib can handle this automatically
        # or create a palette with distinct colors for the unique cell types present
        unique_celltypes = celltype_labels.unique()
        import matplotlib.pyplot as plt
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_celltypes)))
        cell_type_color_map = dict(zip(unique_celltypes, colors))

        umap_model = umap.UMAP(n_neighbors=50, min_dist=0.05, metric='euclidean', random_state=42)
        X_val_umap = umap_model.fit_transform(meta_data_X)

        plt.figure(figsize=(20, 10))  # Made wider to accommodate legend

        plt.subplot(1, 2, 1)
        sns.scatterplot(x=X_val_umap[:, 0], y=X_val_umap[:, 1], hue=y_val_pred, palette=color_palette, s=10)
        plt.title('UMAP Projection - Predicted Labels', fontsize=14)
        plt.legend(loc='best')

        plt.subplot(1, 2, 2)
        # Use human-readable cell type labels
        sns.scatterplot(x=X_val_umap[:, 0], y=X_val_umap[:, 1], hue=celltype_labels, s=10)
        plt.title('UMAP Projection - Cell Type', fontsize=14)
        
        # Position legend outside the plot to avoid overcrowding
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=15)

        for ax in plt.gcf().axes:
            ax.grid(False)
            ax.spines['top'].set_linewidth(2)
            ax.spines['right'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)
            ax.spines['left'].set_linewidth(2)

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()  # Adjust layout to prevent legend cutoff
        plt.savefig(f"{save_path}/figure.png", dpi=300, bbox_inches='tight')

        class_labels = {0: 'Normal-like', 1: 'Pre-malignant', 2: 'Tumor-like'}
        self.adata.obs['oncoterrain_class'] = [class_labels[label] for label in y_val_pred]
        
        # Also save human-readable cell type labels to adata
        self.adata.obs['celltype_readable'] = celltype_labels.values

        if save_adata:
            self.adata.write_h5ad(save_path / "OncoTerrain_annotated.h5ad")