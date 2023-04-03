import gensim
import itertools
import pandas as pd
from streampq import streampq_connect

class MalamudDataset:
    """Iterator class for loading malamud general index data from a postgres database
    
    Attributes:
        postgres_conn_params: Tuple of tuples, each containing a key value pair with postgres connection details
            e.g.
                (
                    ('host', 'localhost'),
                    ('port', '5432'),
                    ('dbname', 'my_db'),
                    ('user', 'my_user'),
                    ('password', 'my_password'),
                )
        chunk_size: Integer indicating the number of rows in each dataframe chunk
        table: String indicating table to use within the malamud database
        column: String indicating column to use within the selected table
        rows: Number of rows to train with, starting from 0
    """
    def __init__(self, postgres_conn_params: tuple, chunk_size: int, table: str, column: str, rows: int=None):
        self.postgres_conn_params = postgres_conn_params
        self.chunk_size = chunk_size
        self.table = table
        self.column = column
        self.pg_command = f"SELECT * FROM {self.table}{f' limit {rows}' if rows != float('inf') else ''}"

    def __iter__(self):
        conn = streampq_connect(self.postgres_conn_params)
        with conn as query:
            for chunked_dfs in self.query_chunked_dfs(query, self.pg_command, chunk_size=self.chunk_size):  # TODO: Experiment with different chunk sizes
                for df in chunked_dfs:
                    for ngram in df:
                        yield ngram

    # Adapted from example in https://github.com/uktrade/streampq#chunked-pandas-dataframes-of-sql-query-results
    def query_chunked_dfs(self, query, sql, chunk_size):

        def _chunked_df(columns, rows):
            it = iter(rows)
            while True:
                df = pd.DataFrame(itertools.islice(it, chunk_size), columns=columns)[self.column].apply(gensim.utils.simple_preprocess)
                if len(df) == 0:
                    break
                yield df

        for columns, rows in query(sql):
            yield _chunked_df(columns, rows)