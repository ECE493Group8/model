import gensim
import connectorx as cx

import itertools
import pandas as pd
from streampq import streampq_connect

PG_QUERY = "SELECT * FROM docs.doc_ngrams_0"

class MalamudDataset:
    """
    
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
    """
    def __init__(self, postgres_conn_params: tuple, chunk_size: int=10000):
        self.conn = streampq_connect(postgres_conn_params)
        self.chunk_size = chunk_size

    def __iter__(self):
        with self.conn as query:
            for chunked_dfs in self.query_chunked_dfs(query, PG_QUERY, chunk_size=self.chunk_size):  # TODO: Experiment with different chunk sizes
                for df in chunked_dfs:
                    for ngram in df:
                        yield ngram

    # Adapted from example in https://github.com/uktrade/streampq#chunked-pandas-dataframes-of-sql-query-results
    def query_chunked_dfs(self, query, sql, chunk_size):

        def _chunked_df(columns, rows):
            it = iter(rows)
            while True:
                df = pd.DataFrame(itertools.islice(it, chunk_size), columns=columns).ngram_lc.apply(gensim.utils.simple_preprocess)
                if len(df) == 0:
                    break
                yield df

        for columns, rows in query(sql):
            yield _chunked_df(columns, rows)