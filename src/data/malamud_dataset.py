import polars as pl
import itertools
import argparse
import logging
import glob
from time import perf_counter

logger = logging.getLogger(__name__)

class MalamudDataset:
    """Iterator class for loading malamud general index data from a postgres database

    **Assumes that the data from a single parquet file fits in memory**
    
    Attributes:
        parq_path: String path to a parquet file to load, in form 'path/to/file_<X>.parquet'
        num_files: Integer indicating the number of files to load, will replace <X> in parq_path
        column: String indicating column to use within the selected table
    """
    def __init__(self, parquet_path: str, column: str):
        
        # Save params
        self.parquet_path = parquet_path
        self.column = column

    def __iter__(self):
        """Yield a value from each dataframe"""

        # Load parquets file into memory one at a time
        for parquet_file in glob.glob(self.parquet_path):

            # Read the parquet into memory
            logger.info(f"Reading {parquet_file} into memory...")
            df = pl.scan_parquet(parquet_file).select(self.column).collect()
            logger.info(f"Finished reading {parquet_file} into memory. Iterating...")

            # Yield each row
            for row in df.iter_rows():
                yield row[0]
            logger.info(f"Finished iterating {parquet_file}")

if __name__ == "__main__":
    """
    If calling this file directly from the command line, print time to read all files

    Example usage: "python src/data/malamud_dataset.py --parquet_path "*.parquet" --column ngram_lc"
    
    Benchmarks:
        For 650k row csv split into 16 chunks and converted to parquet:
            - Read when applying preprocessing function: ~4.2s
            - Read when not applying preprocessing function: ~0.15s
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_path", type=str, required=True)
    parser.add_argument("--column", type=str, required=True)
    parser.add_argument("--print", action="store_true", default=False)
    args = parser.parse_args()
    t = perf_counter()
    dataset = MalamudDataset(args.parquet_path, args.column)
    for r in dataset:
        if args.print:
            print(r)
    print(f"Time to read: {(perf_counter() - t):.4f}s")
