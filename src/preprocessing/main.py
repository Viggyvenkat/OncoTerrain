import os
import sys
import numpy as np
import scanpy as sc
import pandas as pd
import logging
import gc
from preprocessing import PreprocessingPipeline 
from pathlib import Path
from scipy import sparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    pipeline = PreprocessingPipeline(data_dir="data/scRNAseq-data")
    logger.info("Running full pipeline")
    adata = pipeline.load_all_data()
    adata = pipeline.preprocessing()
    adata = pipeline.ct_annotation()
    adata = pipeline.hp_calculation()
    adata = pipeline.lr_interactions_graph_representation()    
    logger.info("Starting monocle analysis")
    monocle_output = pipeline.monocle_per_celltype()
    logger.info("Monocle analysis completed")
    logger.info("Pipeline completed successfully!")
