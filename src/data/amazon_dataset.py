import gensim
import pandas as pd


class AmazonDataset:
    def __init__(self, path: str):
        self.df = pd.read_json(path, lines=True)
        self.reviews = self.df.reviewText.apply(gensim.utils.simple_preprocess)
    
    def __iter__(self):
        for review in self.reviews:
            yield review


if __name__ == "__main__":
    print("Creating dataset...")
    dataset = AmazonDataset("./amazon_product_reviews.json")
    model = gensim.models.Word2Vec(window=10, min_count=2, workers=4)
    print("Building vocabulary...")
    model.build_vocab(dataset, progress_per=1000)
    print("Training...")
    model.train(dataset, total_examples=model.corpus_count, epochs=model.epochs)
    print("Done!")
    model.save("word2vec_amazon.model")
