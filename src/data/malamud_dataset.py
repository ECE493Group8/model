import pandas as pd
import polars as pl
import argparse
import gensim
import itertools
from time import perf_counter

def word2med_preprocess(phrase: str):
    return phrase.split()

class MalamudDataset:
    """Iterator class for loading malamud general index data from a postgres database

    **Assumes that the data from a single parquet file fits in memory**
    
    Attributes:
        parq_path: String path to a parquet file to load, in form 'path/to/file_<X>.parquet'
        num_files: Integer indicating the number of files to load, will replace <X> in parq_path
        column: String indicating column to use within the selected table
    """
    def __init__(self, parq_base_path: str, num_files: int, column: str):
        
        # Save params
        self.parq_base_path = parq_base_path
        self.num_files = num_files
        self.column = column

        # List that will hold the iterators for each df
        self.df_iters = []

        # Load each df
        for i in range(self.num_files):

            # Get the next keyword file name
            parq_path = self.parq_base_path.replace('<X>', str(i))
            self.df_iters.append(pl.scan_parquet(parq_path).select(column).collect().iter_rows())

    def __iter__(self):
        """Yield a value from each dataframe"""
        empty = {}
        for values in itertools.zip_longest(*self.df_iters, fillvalue=empty):
            for value in values:
                if value != empty:
                    yield value[0]

if __name__ == "__main__":
    """
    If calling this file directly from the command line, print time to read all files

    Example usage: "python src/data/malamud_dataset_polars_multi_files.py --parq_base_path "docs.doc_ngram_0_<X>.parquet" --num_files 16 --column ngram_lc"
    
    Benchmarks:
        For 650k row csv split into 16 chunks and converted to parquet:
            - Read when applying preprocessing function: ~4.2s
            - Read when not applying preprocessing function: ~0.15s
    """

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--parq_base_path", type=str, required=True)
    # parser.add_argument("--num_files", type=int, required=True)
    # parser.add_argument("--column", type=str, required=True)
    # parser.add_argument("--print", action="store_true", default=False)
    # args = parser.parse_args()
    t = perf_counter()
    dataset = MalamudDataset("/home/justin/uni/ece493/model/docs.doc_ngram_0_<X>.parquet", 16, "ngram_lc")
    # dataset = MalamudDataset(args.parq_base_path, args.num_files, args.column)
    for r in dataset:
        pass
        # if args.print:
            # print(r)
    print(f"Time to read: {(perf_counter() - t):.4f}s")