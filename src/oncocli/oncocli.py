#!/usr/bin/env python
# oncoterrain_cli.py

import logging
from pathlib import Path

import click
import scanpy as sc

from oncocli.OncoTerrain import OncoTerrain

@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Enable debug logging (very chatty)."
)
@click.pass_context
def cli(ctx, verbose):
    """OncoTerrain: single-cell oncogenic terrain analysis."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s: %(message)s")
    ctx.obj = {"log_level": level}


@cli.command()
@click.argument(
    "input_path",
    type=click.Path(exists=True, path_type=Path)
)
@click.option(
    "-o", "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("results"),
    show_default=True,
    help="Where to save figures and outputs."
)
@click.option(
    "--no-save-adata",
    is_flag=True,
    help="Don't write the annotated AnnData back to disk."
)
@click.pass_context
def infer(ctx, input_path, output_dir, no_save_adata):
    """
    Run full OncoTerrain pipeline on INPUT_PATH and dump outputs into OUTPUT_DIR.
    
    INPUT_PATH can be either:
      - a 10x mtx folder (will use scanpy.read_10x_mtx)
      - an .h5ad file (will use scanpy.read_h5ad)
    """
    path = Path(input_path)
    if path.is_dir():
        logging.info(f"Reading 10x mtx from {path}")
        adata = sc.read_10x_mtx(path, var_names='gene_symbols')
    else:
        logging.info(f"Reading AnnData from {path}")
        adata = sc.read_h5ad(path)

    ot = OncoTerrain(adata)
    ot.inferencing(
        save_path=output_dir,
        save_adata=not no_save_adata
    )
    logging.info("Done! All files in %s", output_dir)


@cli.command(name="batch")
@click.argument(
    "input_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "-o", "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("figures"),
    show_default=True,
    help="Base output directory for all samples."
)
@click.option(
    "--no-save-adata",
    is_flag=True,
    help="Don't write the annotated AnnData back to disk for each sample."
)
@click.pass_context
def batch(ctx, input_dir, output_dir, no_save_adata):
    """
    Process all 10x-style sample folders under INPUT_DIR, writing outputs to OUTPUT_DIR.

    Each subdirectory of INPUT_DIR that contains a 10x mtx is processed in turn.
    """
    base_out = Path(output_dir)
    for sample_dir in Path(input_dir).iterdir():
        if not sample_dir.is_dir():
            continue
        try:
            logging.info(f"--- Processing sample {sample_dir.name} ---")
            adata = sc.read_10x_mtx(sample_dir, var_names='gene_symbols')
        except Exception as e:
            logging.warning(f"Skipping {sample_dir.name}: not a valid 10x folder ({e})")
            continue

        ot = OncoTerrain(adata)
        out_subdir = base_out / f"{sample_dir.name}_oncoterrain"
        ot.inferencing(
            save_path=out_subdir,
            save_adata=not no_save_adata
        )
        logging.info(f"Finished {sample_dir.name}, outputs in {out_subdir}\n")


if __name__ == "__main__":
    cli()
