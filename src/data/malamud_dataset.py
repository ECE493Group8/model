import gensim
import connectorx as cx

class MalamudDataset:
    def __init__(self, postgres_uri: str):
        self.df = cx.read_sql(postgres_uri, "SELECT * FROM docs.doc_ngrams_0")
        self.ngrams = self.df.ngram_lc.apply(gensim.utils.simple_preprocess)

    def __iter__(self):
        for ngram in self.ngrams:
            yield ngram