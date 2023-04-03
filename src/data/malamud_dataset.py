import polars as pl
import itertools
import argparse
import glob
from time import perf_counter

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

        # Load each df into memory
        self.dfs = []
        for parquet_file in glob.glob(parquet_path):
            # Get the next keyword file name
            self.dfs.append(pl.scan_parquet(parquet_file).select(column).collect())

    def __iter__(self):
        """Yield a value from each dataframe"""
        empty = {}
        df_iters = map(pl.DataFrame.iter_rows, self.dfs)
        for values in itertools.zip_longest(*df_iters, fillvalue=empty):
            for value in values:
                if value != empty:
                    yield value[0]

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
