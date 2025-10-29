import warnings
warnings.filterwarnings("ignore")

import os
import re
import logging
import scanpy as sc
import mygene
import matplotlib.pyplot as plt
import seaborn as sns
import concurrent.futures
import pandas as pd
import numpy as np
import gseapy as gp
from pathlib import Path
from cellphonedb.src.core.methods import cpdb_statistical_analysis_method
import concurrent.futures
from py_monocle import (learn_graph, order_cells, compute_cell_states, regression_analysis, differential_expression_genes)
from matplotlib.colors import LinearSegmentedColormap
from scipy import sparse
import matplotlib.patches as mpatches
from threadpoolctl import threadpool_limits
from scipy.sparse import issparse
from scipy.sparse import triu as sp_triu

sc.settings.n_jobs = 16
BASE_DIR = Path.cwd()

logging.basicConfig(
    level=logging.DEBUG,
    filename=str(BASE_DIR / 'debug_log.txt'),
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class PreprocessingPipeline:
    def __init__(self, data_dir="data"):
        """Initializes the data processing pipeline with the specified data directory.

        Args:
            data_dir (str): Path to the base directory containing SDxx subfolders or 
                `.h5ad` files. If a relative path is provided, it is resolved to an 
                absolute path based on the current working directory.

        Returns:
            None

        Raises:
            FileNotFoundError: If the specified data directory does not exist.
            NotADirectoryError: If the resolved data_dir path is not a directory.
        """
        data_dir = Path(data_dir)
        if not data_dir.is_absolute():
            data_dir = Path.cwd() / data_dir
        self.data_dir = data_dir
        sc.settings.n_jobs = -1
        self.adata = None

        self.sample_dirs = [entry.path for entry in os.scandir(self.data_dir) if entry.is_dir()]

        logging.info(f"Initialized pipeline with data_dir: {self.data_dir}")
        logging.info(f"Found {len(self.sample_dirs)} sample directories")
    
    @staticmethod
    def _load_h5ad_file(fpath):
        """
        Loads a .h5ad file into memory and standardizes its metadata.

        Adds or sets default values for key observation metadata fields such as 
        'tumor_stage', 'sample', 'source', 'project', and converts 'batch' to string 
        if present.

        Args:
            fpath (str): Path to the `.h5ad` file to be loaded.

        Returns:
            AnnData or None: The loaded AnnData object with updated `.obs` metadata, 
            or `None` if an error occurred during loading.

        Raises:
            None: All exceptions are caught internally and logged; no exceptions are propagated.
        """
        logging.info(f"Loading h5ad file: {fpath}")
        try:
            ad = sc.read_h5ad(fpath).to_memory()
            sample_id = os.path.splitext(os.path.basename(fpath))[0]

            # --- special-case: force normal / non-cancer for the control file
            control_uuid = "b351804c-293e-4aeb-9c4c-043db67f4540"
            if control_uuid in os.path.normpath(fpath):
                ad.obs["disease"] = "normal"
                ad.obs["tumor_stage"] = "non-cancer"
                logging.info("Annotated control file as disease=normal, tumor_stage=non-cancer")

            if 'tumor_stage' not in ad.obs.columns:
                ad.obs['tumor_stage'] = np.nan

            # If disease not present, leave it; above block sets it for the control file
            ad.obs['sample']  = ad.obs.get('sample', sample_id)
            ad.obs['source']  = fpath
            ad.obs['project'] = ad.obs.get('project', sample_id)

            if 'batch' in ad.obs.columns:
                ad.obs['batch'] = ad.obs['batch'].astype(str)

            # Ensure string indices
            ad.obs.index = ad.obs.index.astype(str)

            return ad

        except Exception as e:
            logging.info(f"  Error loading {fpath}: {e}")
            return None

    @staticmethod
    def _load_sample_dir(sample_dir):
        """
        Loads and processes all single-cell data files within a sample directory.

        This function walks through the given directory, loading `.h5ad` or 10× matrix 
        files. It standardizes key metadata fields (`tumor_stage`, `disease`, `sample`, 
        `project`, `source`, etc.), inferring tumor stage and disease type when missing 
        based on the SDxx project naming convention. All loaded AnnData objects are 
        concatenated into a single dataset representing one project.

        Args:
            sample_dir (str): Path to the sample directory containing `.h5ad` files or 
                10× matrix data to load.

        Returns:
            List[AnnData]: A list containing a single merged `AnnData` object with all 
            loaded and standardized data. Returns an empty list if no valid data is found.

        Raises:
            None: All exceptions during file loading are caught and logged; function 
            does not raise errors directly.
        """
        logging.info(f"Loading sample directory as one project: {sample_dir}")
        sd_numbers    = [2, 3, 4, 6, 7, 8, 9, 10, 12, 14, 15, 16]
        cancer_stages = ["insitu", "benign", "benign", "benign", "invasive",
                         "insitu", "insitu", "insitu", "insitu", "invasive",
                         "insitu", "invasive"]
        sd_to_stage   = {f"SD{n}": s for n, s in zip(sd_numbers, cancer_stages)}
        stage_to_ts   = {'benign':'non-cancer', 'insitu':'early', 'invasive':'advanced'}

        project_name = os.path.basename(sample_dir)
        collected = []

        for root, dirs, files in os.walk(sample_dir):
            h5ad_files = [f for f in files if f.lower().endswith(".h5ad")]
            if h5ad_files:
                for fname in h5ad_files:
                    fpath = os.path.join(root, fname)
                    logging.info(f"  → Loading nested h5ad: {fpath}")
                    try:
                        ad = sc.read_h5ad(fpath).to_memory()

                        if 'tumor_stage' not in ad.obs.columns:
                            m = re.search(r"SD(\d+)", project_name)
                            key = f"SD{m.group(1)}" if m else None
                            inferred = stage_to_ts.get(sd_to_stage.get(key, ''), np.nan)
                            ad.obs['tumor_stage'] = inferred

                        if 'disease' not in ad.obs.columns and key in sd_to_stage:
                            ad.obs['disease'] = (
                                "normal" if sd_to_stage[key]=="benign"
                                else "lung adenocarcinoma"
                            )

                        sample_id = os.path.splitext(fname)[0]
                        ad.obs['sample']  = ad.obs.get('sample', sample_id)
                        ad.obs['project'] = project_name
                        ad.obs['source']  = root

                        if 'batch' in ad.obs.columns:
                            ad.obs['batch'] = ad.obs['batch'].astype(str)

                        ad.obs.index = ad.obs.index.astype(str)
                        collected.append(ad)
                    except Exception as e:
                        logging.info(f"    Error loading {fpath}: {e}")
                continue

            if any(f.lower().endswith((".mtx", ".mtx.gz")) for f in files):
                logging.info(f"  → Loading 10× directory: {root}")
                try:
                    ad = sc.read_10x_mtx(root, var_names="gene_symbols", cache=True).to_memory()

                    if 'tumor_stage' not in ad.obs.columns:
                        m = re.search(r"SD(\d+)", project_name)
                        key = f"SD{m.group(1)}" if m else None
                        ad.obs['tumor_stage'] = stage_to_ts.get(sd_to_stage.get(key, ''), np.nan)

                    if 'disease' not in ad.obs.columns and key in sd_to_stage:
                        ad.obs['disease'] = (
                            "normal" if sd_to_stage[key]=="benign"
                            else "lung adenocarcinoma"
                        )

                    ad.obs['sample']  = project_name
                    ad.obs['project'] = project_name
                    ad.obs['source']  = root
                    ad.obs['tissue']  = "lung"
                    ad.obs.index = ad.obs.index.astype(str)

                    if 'batch' in ad.obs.columns:
                        ad.obs['batch'] = ad.obs['batch'].astype(str)

                    collected.append(ad)
                except Exception as e:
                    logging.info(f"    Failed to load 10× data from {root}: {e}")

        if not collected:
            logging.info(f"No valid data found under {sample_dir}")
            return []

        logging.info(f"  → Concatenating {len(collected)} pieces for project {project_name}")
        merged = sc.concat(
            collected,
            join="outer",
            index_unique="-",
            label="subbatch"
        )

        if sparse.issparse(merged.X):
            merged.X.data = np.nan_to_num(merged.X.data)
        else:
            merged.X = np.nan_to_num(merged.X)

        merged.obs.index = merged.obs.index.astype(str)
        merged.var.index = merged.var.index.astype(str)
        logging.info(
            f"    → Project {project_name} merged: "
            f"{merged.n_obs} cells, {merged.n_vars} genes"
        )
        return [merged]

    @staticmethod
    def _map_gene_symbols(adata) -> sc.AnnData:
        """
        Maps Ensembl gene IDs to gene symbols in an AnnData object.

        This method uses the MyGene.info service to convert Ensembl gene identifiers 
        in `adata.var_names` to human gene symbols. The resulting gene symbols replace 
        `var_names`, which are also made unique.

        Args:
            adata (AnnData): An AnnData object with Ensembl gene IDs as `var_names`.

        Returns:
            AnnData: The updated AnnData object with `var_names` mapped to gene symbols 
            and made unique.

        Raises:
            None: All mapping failures are handled gracefully by falling back to the 
            original Ensembl ID.
        """
        logging.info("Mapping gene symbols for one AnnData")
        mg = mygene.MyGeneInfo()
        ensg_ids = adata.var_names.tolist()
        query_results = mg.querymany(ensg_ids, scopes="ensembl.gene", fields="symbol", species="human")
        mapping = {res["query"]: str(res.get("symbol", res["query"])) for res in query_results}
        adata.var["gene_symbol"] = [mapping.get(g, str(g)) for g in adata.var_names]
        adata.var["gene_symbol"] = adata.var["gene_symbol"].astype(str)
        adata.var_names = adata.var["gene_symbol"]
        adata.var_names_make_unique()
        adata.var.drop(columns="gene_symbol", inplace=True)
        adata.var.index = adata.var.index.astype(str)
        return adata

    def load_all_data(self) -> sc.AnnData:
        """
        Loads and preprocesses all single-cell data from the provided data directory.

        This method performs parallel loading of `.h5ad` files and 10× Genomics-formatted 
        directories found in the root of `data_dir` or its subdirectories. It standardizes 
        metadata, filters low-quality cells and genes, maps gene symbols, reindexes each 
        dataset to a shared union of genes, and concatenates them into a single 
        `AnnData` object stored in `self.adata`.

        Args:
            None

        Returns:
            AnnData: A single concatenated and preprocessed `AnnData` object containing 
            all valid data loaded from the root `.h5ad` files and subdirectories.

        Raises:
            ValueError: If no valid data could be loaded or processed successfully.
        """
        logging.info("Starting parallel data loading")

        root_h5ad_files = [
            entry.path
            for entry in os.scandir(self.data_dir)
            if entry.is_file() and entry.name.lower().endswith(".h5ad")
        ]

        all_adata = []
        failed_samples = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
            futures = []
            for fpath in root_h5ad_files:
                futures.append(executor.submit(PreprocessingPipeline._load_h5ad_file, fpath))
            for sample_dir in self.sample_dirs:
                futures.append(executor.submit(PreprocessingPipeline._load_sample_dir, sample_dir))

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result is None:
                        continue
                    if isinstance(result, list):
                        all_adata.extend([ad for ad in result if ad is not None])
                    else:
                        all_adata.append(result)
                except Exception as e:
                    logging.info(f"Error retrieving future result: {e}")
                    failed_samples.append(str(e))

        if not all_adata:
            errmsg = "No valid data loaded. Errors: " + ", ".join(failed_samples[:5])
            if len(failed_samples) > 5:
                errmsg += f" (+{len(failed_samples)-5} more errors)"
            logging.info(errmsg)
            raise ValueError(errmsg)

        logging.info(f"Loaded {len(all_adata)} AnnData objects. Failed: {len(failed_samples)}")

        for ad in all_adata:
            ad.raw = None
            if "project" not in ad.obs.columns:
                ad.obs["project"] = os.path.basename(ad.obs["source"].iloc[0])
                logging.info(f"Filled missing project for an AnnData: {ad.obs['project'].iloc[0]}")

        for i in range(len(all_adata)):
            all_adata[i] = PreprocessingPipeline._map_gene_symbols(all_adata[i])

        for i, ad in enumerate(all_adata):
            logging.info(f"QC sample {i+1}/{len(all_adata)}: {ad.n_obs} cells, {ad.n_vars} genes")
            sc.pp.filter_cells(ad, min_genes=200)
            logging.info(f"  After filter_cells: {ad.n_obs} cells")
            sc.pp.filter_genes(ad, min_cells=10)
            logging.info(f"  After filter_genes: {ad.n_vars} genes")

        union_genes = sorted({g for ad in all_adata for g in ad.var_names})
        logging.info(f"Union of genes after QC: {len(union_genes)} genes")

        gene_to_idx = {g: idx for idx, g in enumerate(union_genes)}
        reindexed = []
        for ad in all_adata:
            orig_genes = list(ad.var_names)
            n_obs = ad.n_obs
            n_union = len(union_genes)

            new_X = sparse.lil_matrix((n_obs, n_union), dtype=ad.X.dtype)
            orig_indices = [gene_to_idx[g] for g in orig_genes]
            new_X[:, orig_indices] = ad.X

            new_adata = sc.AnnData(
                X = new_X.tocsr(),
                obs = ad.obs.copy(),
                var = pd.DataFrame(index=union_genes)
            )
            new_adata.var.index = new_adata.var.index.astype(str)
            new_adata.var_names = union_genes
            logging.info(f"Reindexed sample: {new_adata.n_obs} cells, {new_adata.n_vars} genes")
            reindexed.append(new_adata)

        keys = [f"{ad.obs['project'].iloc[0]}_{i}" for i, ad in enumerate(reindexed)]
        self.adata = sc.concat(
            reindexed,
            label="batch",
            keys=keys,
            join="inner",
            index_unique=None
        )
        self.adata.obs.index = self.adata.obs.index.astype(str)
        self.adata.var.index = self.adata.var.index.astype(str)
        logging.info(
            f"Concatenated {len(reindexed)} samples → "
            f"{self.adata.n_obs} total cells, {self.adata.n_vars} genes"
        )

        if sparse.issparse(self.adata.X):
            self.adata.X.data = np.nan_to_num(self.adata.X.data)
        else:
            self.adata.X = np.nan_to_num(self.adata.X)

        return self.adata

    def preprocessing(self):
        """
        Applies standard preprocessing to the loaded single-cell dataset.

        This includes mitochondrial gene annotation, QC metric calculation, total-count 
        normalization (to 10,000 reads per cell), and log1p transformation.

        Args:
            None

        Returns:
            AnnData: The processed `AnnData` object with QC metrics, normalized, and 
            log-transformed expression data.

        Raises:
            ValueError: If `self.adata` is not loaded (i.e., `load_all_data()` was not run).
        """
        if self.adata is None:
            logging.error("Data not loaded. Run load_all_data() first.")
            raise ValueError("Data not loaded. Run load_all_data() first.")

        self.adata.obs.index = self.adata.obs.index.astype(str)
        self.adata.var.index = self.adata.var.index.astype(str)

        if 'disease' in self.adata.obs.columns:
            disease_raw = self.adata.obs['disease'].astype(str).str.strip().str.lower()

            normalize_map = {
                'luad': 'lung adenocarcinoma',
                'lung_adenocarcinoma': 'lung adenocarcinoma',
                'lung adenocarcinoma': 'lung adenocarcinoma',
                'adenocarcinoma': 'lung adenocarcinoma',
                'normal lung': 'normal',
                'healthy': 'normal',
                'control': 'normal',
                'normal': 'normal',
            }
            disease_norm = disease_raw.map(lambda x: normalize_map.get(x, x))
            keep_labels = {'normal', 'lung adenocarcinoma'}
            mask = disease_norm.isin(keep_labels)

            n_before = int(self.adata.n_obs)
            if mask.any():
                self.adata = self.adata[mask, :].copy()
                n_after = int(self.adata.n_obs)
                logging.info(
                    f"Subset to disease in {keep_labels}: kept {n_after}/{n_before} cells "
                    f"({n_before - n_after} removed)."
                )
            else:
                logging.warning(
                    "No cells matched disease in {'normal', 'lung adenocarcinoma'}; "
                    "proceeding without subsetting."
                )
        else:
            logging.warning(
                "Column 'disease' not found in adata.obs; proceeding without subsetting."
            )
        
        if "tumor_stage" in self.adata.obs.columns:
            if not self.adata.obs_names.is_unique:
                self.adata.obs_names_make_unique()
            s = self.adata.obs["tumor_stage"].astype("string").str.strip()

            bad = s.isna() | s.str.casefold().eq("nan")
            n_before = int(self.adata.n_obs)
            self.adata.obs["tumor_stage"] = s

            keep = (~bad).to_numpy(dtype=bool)
            self.adata = self.adata[keep, :].copy()

            n_after = int(self.adata.n_obs)
            logging.info(
                f"Removed cells without usable tumor_stage: kept {n_after}/{n_before} cells "
                f"({n_before - n_after} removed)."
            )
        else:
            logging.warning("Column 'tumor_stage' not found in adata.obs; proceeding without filtering.")

        self.adata.var['mt'] = self.adata.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(
            self.adata,
            qc_vars=['mt'],
            percent_top=None,
            log1p=False,
            inplace=True
        )

        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)

        logging.debug(
            "Post-merge normalization/log1p completed. Shape: %s", 
            self.adata.shape
        )
        return self.adata

    def ct_annotation(self):
        """
        Performs cell type annotation on the preprocessed single-cell dataset.

        This method runs dimensionality reduction (PCA), batch correction (BBKNN), 
        UMAP embedding, and Leiden clustering at multiple resolutions. It then annotates 
        clusters using predefined marker gene sets for various cell types. UMAP plots 
        are saved to disk for both raw clusters and annotated cell types.

        Args:
            None

        Returns:
            AnnData: The updated `AnnData` object with cluster and cell type annotations 
            added to `.obs`. The result is also saved as `data/processed_data.h5ad`.

        Raises:
            ValueError: If `self.adata` is not loaded and preprocessed (i.e., 
            `preprocessing()` has not been run).
        """

        logging.debug("Starting ct_annotation")
        if self.adata is None:
            logging.error("Data not preprocessed. Run preprocessing() first.")
            raise ValueError("Data not preprocessed. Run preprocessing() first.")
        os.makedirs("../figures", exist_ok=True)
        os.makedirs("../data", exist_ok=True)

        with threadpool_limits(limits= 16):
            logging.info("Running PCA")
            sc.tl.pca(self.adata, svd_solver='arpack')

            logging.info("Running BBKNN integration")
            sc.external.pp.bbknn(self.adata, batch_key='batch')

            logging.info("Computing UMAP")
            sc.tl.umap(self.adata)

        resolutions = [0.1, 0.5, 1, 2, 5, 10, 20]
        for res in resolutions:
            key = f"leiden_res_{res:4.2f}"
            sc.tl.leiden(self.adata, key_added=key, resolution=res, flavor="igraph")

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
        for res in resolutions:
            cluster_key = f"leiden_res_{res:4.2f}"
            logging.debug(f"Annotating clusters with {cluster_key}")
            annotation = self._annotate_clusters_by_markers(self.adata, cluster_key, filtered_marker_genes)
            self.adata.obs[f"{cluster_key}_celltype"] = self.adata.obs[cluster_key].map(annotation)
        for res in resolutions:
            cluster_key = f"leiden_res_{res:4.2f}"
            celltype_key = f"{cluster_key}_celltype"
            sc.pl.umap(
                self.adata,
                color=cluster_key,
                title=f"Leiden Clusters (res={res:4.2f})",
                palette=seaborn_palette,
                show=False,
                save=f"_umap_leiden_res_{res:4.2f}.png"
            )
            sc.pl.umap(
                self.adata,
                color=celltype_key,
                title=f"Annotated Cell Types (res={res:4.2f})",
                palette=seaborn_palette,
                legend_loc="right margin",
                show=False,
                save=f"_umap_celltype_res_{res:4.2f}.png"
            )
        for col in self.adata.obs.select_dtypes(["category"]).columns:
            self.adata.obs[col] = self.adata.obs[col].astype(str)
        self.adata.write("data/processed_data.h5ad")
        logging.debug("ct_annotation completed")
        return self.adata

    @staticmethod
    def _annotate_clusters_by_markers(adata, cluster_key, marker_dict):
        """
        Assigns cell type labels to clusters based on average marker gene expression.

        For each cluster defined by `cluster_key`, this method computes the mean 
        expression of marker genes across all cells in the cluster. The cell type 
        with the highest average expression of its marker genes is assigned to the cluster.

        Args:
            adata (AnnData): The AnnData object containing the expression data and cluster assignments.
            cluster_key (str): The name of the `.obs` column that contains cluster labels (e.g., "leiden_res_1.00").
            marker_dict (dict): Dictionary mapping cell type names (str) to lists of marker gene names (List[str]).

        Returns:
            dict: A mapping from cluster label (str or int) to inferred cell type name (str).

        Raises:
            None: All edge cases, including empty clusters or missing markers, are handled gracefully.
        """

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

    def _add_and_aggregate_module_scores(self, adata, gmt_file):
        """
        Adds module scores to the AnnData object based on gene sets defined in a GMT file.

        This function parses gene sets from a GMT file, matches them to genes in the provided
        AnnData object, and computes module scores using `scanpy.tl.score_genes` for each
        gene set found.

        Args:
            adata (AnnData): An AnnData object containing single-cell data.
            gmt_file (Path or str): Path to a GMT file containing gene sets.

        Returns:
            AnnData: The input AnnData object with new module scores added to `adata.obs`.

        Raises:
            FileNotFoundError: If the specified GMT file does not exist.
            ValueError: If `adata` is not a valid AnnData object.
        """
        gene_sets = gp.get_library(str(gmt_file))
        adata_genes = [gene.upper() for gene in adata.var_names]
        
        for pathway, genes in gene_sets.items():
            genes = [g.upper() for g in genes]
            genes_in_adata = [gene for gene in genes if gene in adata_genes]
            
            if len(genes_in_adata) > 0:
                print(f"Calculating module score for pathway: {pathway}")
                print(f"Genes in pathway (after matching): {genes_in_adata}")
                sc.tl.score_genes(adata, genes_in_adata, score_name=pathway)
                if pathway in adata.obs.columns:
                    score_col = pathway
                elif f"{pathway}_score" in adata.obs.columns:
                    score_col = f"{pathway}_score"
                else:
                    score_col = None
                if score_col:
                    print(f"Module score for {pathway} added to adata.obs as {score_col}")
                else:
                    print(f"Module score for {pathway} was not added to adata.obs")
            else:
                print(f"No genes found for pathway {pathway} in AnnData object")
        return adata

    def hp_calculation(self):
        """
        Performs hallmark pathway module score calculation on single-cell data.

        This function loads a preprocessed AnnData object, iterates through all GMT
        files in a predefined directory, computes module scores for each gene set
        using `_add_and_aggregate_module_scores`, and writes the updated object back to disk.

        Returns:
            AnnData: The updated AnnData object with all module scores added.

        Raises:
            FileNotFoundError: If the processed AnnData file does not exist.
            Exception: If GMT file processing or AnnData read/write fails.
        """
        if os.path.exists(BASE_DIR / "data/processed_data.h5ad"):
            self.adata = sc.read_h5ad(BASE_DIR / "data/processed_data.h5ad")
            self.adata.obs_names_make_unique()
        logging.debug("starting hp_calculation")
        gmt_dir = BASE_DIR / "HallmarkPathGMT"
        gmt_files = list(gmt_dir.glob("*.gmt"))
        for gmt_file in gmt_files:
            logging.debug(f"Processing GMT file: {gmt_file}")
            self.adata = self._add_and_aggregate_module_scores(self.adata, gmt_file)
        self.adata.write(str(BASE_DIR / "data/processed_data.h5ad"))
        logging.debug("hp_calculation completed")
        return self.adata

    def monocle_per_celltype(self):
        """
        Performs pseudotime trajectory analysis per cell type using UMAP and principal graph learning.

        Now also:
        - Stores per–cell-type MST in `adata_ct.obsp['mst']` and centroids in `adata_ct.uns['mst_centroids']`.
        - Exports per–cell-type MST edge list CSV and centroid/node CSV.
        - Saves an additional centroid-only MST figure.

        Returns:
            Dict[str, AnnData]: maps cell type -> AnnData with `.obs['pseudotime']`,
            `.obsp['mst']`, and `.uns['mst_centroids']`.
        """

        logging.debug("=== START monocle_per_celltype (with MST export) ===")

        if os.path.exists(BASE_DIR / "data/processed_data.h5ad"):
            logging.debug("Reloading processed_data.h5ad for monocle_per_celltype")
            self.adata = sc.read_h5ad(BASE_DIR / "data/processed_data.h5ad")
            self.adata.obs_names_make_unique()

        if self.adata is None:
            logging.error("Data not preprocessed. Run preprocessing() first.")
            raise ValueError("Data not preprocessed. Run preprocessing() first.")

        if 'tumor_stage' not in self.adata.obs.columns:
            logging.error("Column 'tumor_stage' not found in AnnData.obs")
            raise KeyError("Column 'tumor_stage' not found in AnnData.obs."
                        " Please annotate tumor_stage when loading your data.")

        self.adata.obs['tumor_stage'] = self.adata.obs['tumor_stage'].replace(['nan', 'NaN', 'NAN'], np.nan)

        dist_pre = self.adata.obs['tumor_stage'].value_counts(dropna=False)
        logging.debug(f"tumor_stage distribution before drop (incl NaN/strings):\n{dist_pre}")

        mask = self.adata.obs['tumor_stage'].notna()
        n_missing = (~mask).sum()
        logging.debug(f"Found {n_missing} cells with missing or string 'nan' tumor_stage")
        if n_missing > 0:
            logging.debug(f"Dropping {n_missing} cells with missing tumor_stage")
            self.adata = self.adata[mask, :].copy()

        dist_post = self.adata.obs['tumor_stage'].value_counts(dropna=False)
        logging.debug(f"tumor_stage distribution after drop (incl NaN):\n{dist_post}")

        trajectory_dir = BASE_DIR / "figures/trajectory_plots"
        os.makedirs(trajectory_dir, exist_ok=True)
        logging.debug(f"Ensured trajectory directory exists: {trajectory_dir}")

        cmap_custom = LinearSegmentedColormap.from_list('gray_emerald', ['gray', '#00a4ef'])

        results = {}

        for cell_type in self.adata.obs['leiden_res_20.00_celltype'].unique():
            logging.debug(f"--- Processing cell_type = '{cell_type}' ---")

            adata_ct = self.adata[self.adata.obs['leiden_res_20.00_celltype'] == cell_type, :].copy()
            logging.debug(f"  Subset adata_ct: {adata_ct.n_obs} cells × {adata_ct.n_vars} genes")

            if adata_ct.n_obs < 50:
                logging.warning(f"  Skipping '{cell_type}' ({adata_ct.n_obs} cells < 50)")
                continue

            logging.debug("  Computing top 5000 HVGs within this cell_type subset")
            sc.pp.highly_variable_genes(
                adata_ct,
                n_top_genes=5000,
                flavor='seurat_v3',
                subset=False,
                n_bins=20,
                batch_key=None
            )
            num_hvg_ct = int(adata_ct.var['highly_variable'].sum()) if 'highly_variable' in adata_ct.var else 0
            logging.debug(f"    → {num_hvg_ct} HVGs found (expect ~5000)")

            adata_ct = adata_ct[:, adata_ct.var['highly_variable']].copy()
            logging.debug(f"  After HVG subset: {adata_ct.n_obs} cells × {adata_ct.n_vars} genes")

            if 'X_umap' not in adata_ct.obsm:
                logging.debug("  Computing neighbors + UMAP")
                sc.pp.neighbors(adata_ct)
                sc.tl.umap(adata_ct)
            else:
                logging.debug("  UMAP already present, skipping neighbor/umap computation")

            umap = adata_ct.obsm['X_umap']

            if 'leiden' not in adata_ct.obs:
                logging.debug("  Computing Leiden clustering")
                sc.tl.leiden(adata_ct)
            try:
                louvain = adata_ct.obs['leiden'].astype(int).values
            except ValueError:
                logging.warning("  Found NaN in leiden; coercing to int with NaN→-1")
                louvain = pd.to_numeric(adata_ct.obs['leiden'], errors='coerce').fillna(-1).astype(int).values

            logging.debug("  Learning principal graph via learn_graph(...)")
            projected_points, mst, centroids = learn_graph(matrix=umap, clusters=louvain)

            logging.debug("  Ordering cells in pseudotime via order_cells(...)")
            pseudotime = order_cells(
                umap, centroids,
                mst=mst,
                projected_points=projected_points,
                root_cells=0
            )
            adata_ct.obs['pseudotime'] = pseudotime

            # ===== Persist MST & centroids (cluster-level, not cell-level) =====
            # Store in .uns to avoid obs×obs shape requirement
            adata_ct.uns['mst'] = mst                 # sparse (n_clusters × n_clusters)
            adata_ct.uns['mst_centroids'] = centroids # (n_clusters × 2) in UMAP coords
            # keep the cluster ids used to build MST
            adata_ct.obs['leiden_int'] = louvain

            # ===== Export MST edge list + centroid table =====
            mst_ut = sp_triu(mst, k=1, format='coo') if issparse(mst) else None
            if mst_ut is None:
                raise RuntimeError("Expected sparse MST matrix from learn_graph.")

            def _euclid(a, b):
                dx = a[0] - b[0]
                dy = a[1] - b[1]
                return float(np.sqrt(dx*dx + dy*dy))

            edge_rows = []
            for i, j in zip(mst_ut.row, mst_ut.col):
                length = _euclid(centroids[i], centroids[j])
                edge_rows.append((int(i), int(j), float(length)))

            edges_df = pd.DataFrame(edge_rows, columns=['source_cluster', 'target_cluster', 'euclidean_length'])

            nodes_df = pd.DataFrame(
                {
                    'cluster': np.arange(len(centroids), dtype=int),
                    'centroid_umap1': centroids[:, 0],
                    'centroid_umap2': centroids[:, 1],
                }
            )

            safe_ct = str(cell_type).replace(' ', '_')
            edges_path = trajectory_dir / f"mst_edges_{safe_ct}.csv"
            nodes_path = trajectory_dir / f"mst_nodes_{safe_ct}.csv"
            edges_df.to_csv(edges_path, index=False)
            nodes_df.to_csv(nodes_path, index=False)
            logging.debug(f"  Saved MST edges to {edges_path}")
            logging.debug(f"  Saved MST nodes to {nodes_path}")

            # ===== Doc-style principal graph plot (clusters + MST) =====
            plt.figure(1, (8, 6))
            plt.clf()
            plt.title("Principal graph")
            plt.scatter(umap[:, 0], umap[:, 1], c=louvain, s=1, cmap="nipy_spectral")

            edges = np.array(mst.nonzero()).T
            for edge in edges:
                # edge is a pair [i, j]; use advanced indexing like in the docs
                plt.plot(centroids[edge, 0], centroids[edge, 1], c="black", linewidth=1)

            plt.xticks([]); plt.yticks([])
            figpath_pg = trajectory_dir / f"principal_graph_{safe_ct}.png"
            plt.savefig(figpath_pg, dpi=300, bbox_inches='tight')
            plt.close()
            logging.debug(f"  Saved principal graph figure to {figpath_pg}")

            # ===== Your existing pseudotime overlay with MST & sample-stage centroids =====
            plt.figure(figsize=(8, 6))
            plt.title(f"Principal graph + sample‐centroids ({cell_type})")

            scatter_plot = plt.scatter(
                umap[:, 0], umap[:, 1], c=pseudotime,
                s=1, cmap=cmap_custom, rasterized=True
            )

            for edge in edges:
                x0, y0 = centroids[edge[0], 0], centroids[edge[0], 1]
                x1, y1 = centroids[edge[1], 0], centroids[edge[1], 1]
                plt.plot([x0, x1], [y0, y1], c="black", linewidth=1)

            df_umap = pd.DataFrame(umap, columns=['UMAP1', 'UMAP2'], index=adata_ct.obs_names)
            df_umap['sample'] = adata_ct.obs['sample'].values
            df_umap['tumor_stage'] = adata_ct.obs['tumor_stage'].values
            df_umap = df_umap.dropna(subset=['tumor_stage'])

            def mode_or_first(x):
                m = x.mode()
                return m.iloc[0] if len(m) > 0 else x.iloc[0]

            centroid_df = (
                df_umap.groupby('sample')
                .agg({'UMAP1': 'mean', 'UMAP2': 'mean', 'tumor_stage': mode_or_first})
                .reset_index()
            )

            stage_colors = {
                'non-cancer': '#84A970',
                'early': '#E4C282',
                'advanced': '#FF8C00'
            }

            for _, row in centroid_df.iterrows():
                stg = row['tumor_stage']
                plt.scatter(
                    row['UMAP1'], row['UMAP2'],
                    c=stage_colors.get(stg, 'black'), s=30,
                    edgecolors='none', marker='o', zorder=10
                )

            legend_patches = [mpatches.Patch(color=col, label=stg) for stg, col in stage_colors.items()]
            plt.legend(handles=legend_patches, title='Tumor stage', loc='best', frameon=True)

            plt.xticks([]); plt.yticks([])
            cbar = plt.colorbar(scatter_plot); cbar.set_label('Pseudotime')

            figpath = trajectory_dir / f"trajectory_{safe_ct}_with_centroids.png"
            plt.savefig(figpath, dpi=300, bbox_inches='tight')
            plt.close()

            # ===== Optional: centroid-only MST diagnostic (clean) =====
            plt.figure(figsize=(6, 5))
            plt.title(f"MST (centroids) — {cell_type}")
            for _, r in edges_df.iterrows():
                i, j = int(r['source_cluster']), int(r['target_cluster'])
                plt.plot([centroids[i, 0], centroids[j, 0]],
                        [centroids[i, 1], centroids[j, 1]],
                        linewidth=1)
            plt.scatter(centroids[:, 0], centroids[:, 1], s=40)
            for k, (x, y) in enumerate(centroids):
                plt.text(x, y, str(k), fontsize=8, ha='center', va='center')
            plt.xticks([]); plt.yticks([])
            clean_mst_fig = trajectory_dir / f"mst_{safe_ct}_centroids_only.png"
            plt.savefig(clean_mst_fig, dpi=300, bbox_inches='tight')
            plt.close()
            logging.debug(f"  Saved centroid-only MST figure to {clean_mst_fig}")