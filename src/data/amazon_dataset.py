import gensim
import pandas as pd


class AmazonDataset:
    def __init__(self, path: str):
        self.df = pd.read_json(path, lines=True)
        self.reviews = self.df.reviewText.apply(gensim.utils.simple_preprocess)

    def __iter__(self):
        for review in self.reviews:
            yield review
