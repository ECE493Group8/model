import pandas as pd
import polars as pl
import argparse
import gensim
from time import perf_counter

def word2med_preprocess(phrase: str):
    return phrase

class MalamudDataset:
    """Iterator class for loading malamud general index data from a postgres database
    
    Attributes:
        parq_path: String path to a parquet file to load
        chunk_size: Integer indicating the number of rows in each dataframe chunk
        column: String indicating column to use within the selected table
        rows: Number of rows to train with, starting from 0
            ** Note: rows should be a multiple of chunk_size
    """
    def __init__(self, parq_path: str, chunk_size: int, column: str, rows: int=None):
        
        # Save params
        self.chunk_size = chunk_size
        self.rows = rows
        self.column = column

        # Initialize query
        self.q = (
            pl.scan_parquet(parq_path)
            .select(pl.col(column).apply(lambda e: word2med_preprocess(e)))
        )

    def __iter__(self):

        # Reset tracking variables
        self.chunk = None
        self.offset_factor = 0
        self.current_row = 0

        while self.chunk is None or (self.chunk.height == self.chunk_size and self.current_row < self.rows):

            # Get another chunk of data from the LazyFrame
            self.chunk = self.q.slice(self.offset_factor*self.chunk_size, self.chunk_size).collect()
            self.offset_factor += 1
            self.current_row += self.chunk_size
            
            for row in self.chunk.iter_rows():
                yield row[0]

if __name__ == "__main__":
    """
    If calling this file directly from the command line, print time to iterate through a dataframe with the given parameters

    In general, a larger chunk size is fastest.
    """

    # TODO: Map each sentence to list, see warning in training.log

    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, required=True)
    parser.add_argument("--column", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--print", action="store_true", default=False)
    args = parser.parse_args()
    t = perf_counter()
    dataset = MalamudDataset(args.filepath, args.chunk_size, args.column, rows=args.rows)
    for r in dataset:
        if args.print:
            print(r)
    print(f"Time to read: {(perf_counter() - t):.4f}s")