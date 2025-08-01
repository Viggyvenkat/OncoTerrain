import scanpy as sc
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import gseapy
import pandas as pd 
from pathlib import Path
import matplotlib.colors as mcolors
from cellphonedb.src.core.methods import cpdb_statistical_analysis_method
import os
import sys
import hashlib
import colorsys
from graphviz import Digraph
from adjustText import adjust_text
import matplotlib.patheffects as path_effects
import random

sys.path.append('../src/TMEGPT')
from Graph import Node, Graph, Edge
import pickle
from scipy.stats import ttest_ind
import networkx as nx
import xlrd
import time
import requests
import mygene

supplementary_table_1_nature = pd.read_excel('../data/NIHMS982390-supplement-Table_1 (1).xls')
supplementary_table_1_nature.head()

sample_types = supplementary_table_1_nature.iloc[0, 1:]  

data = supplementary_table_1_nature.iloc[1:, :]
data.columns = ['Gene'] + sample_types.tolist()
data = data.set_index('Gene')

data = data.apply(pd.to_numeric)

averaged_df = data.groupby(level=0, axis=1).mean()

averaged_df = averaged_df.reset_index()
mouse_genes = averaged_df['Gene'].tolist()  

mg = mygene.MyGeneInfo()
orthologs = mg.querymany(mouse_genes, 
                        scopes='ensembl.gene',
                        fields='symbol,homologene',
                        species='mouse')

gene_mapping = {}
no_ortholog = []

for item in orthologs:
    mouse_id = item['query']
    
    if 'homologene' in item and item['homologene']:
        homologene_data = item['homologene']
        if isinstance(homologene_data, dict) and 'genes' in homologene_data:
            genes = homologene_data['genes']
            
            for gene_info in genes:
                if isinstance(gene_info, dict) and gene_info.get('taxid') == 9606:
                    gene_mapping[mouse_id] = gene_info.get('symbol', 'Unknown')
                    break
        
        if mouse_id not in gene_mapping and 'symbol' in item:
            gene_mapping[mouse_id] = item['symbol']
    
    if mouse_id not in gene_mapping:
        no_ortholog.append(mouse_id)

print(f"Successfully mapped: {len(gene_mapping)} genes")
print(f"No mapping found for: {len(no_ortholog)} genes")

averaged_df['Human_Ortholog'] = averaged_df['Gene'].map(gene_mapping)
averaged_df = averaged_df.dropna(subset=['Human_Ortholog'])

averaged_df.head()

averaged_df.to_csv('../data/averaged_gene_expression_nature_mice_supp_1.csv', index=False)