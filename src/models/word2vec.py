from gensim.models import Word2Vec as GensimWord2Vec
from gensim.test.utils import common_texts

from utils.callbacks import Word2VecEpochCallback


class Word2Vec:
    def __init__(self):
        self.epoch_callback = Word2VecEpochCallback()
        self.model = GensimWord2Vec(sentences=common_texts,
                                    corpus_file=None,
                                    vector_size=16,
                                    window=5,
                                    min_count=1,  # Ignore all words with total frequency lower than this
                                    workers=1,
                                    sg=1,  # 0 for skip-gram, 1 for CBOW
                                    hs=0,  # 0 for negative sampling, 1 for hierarchical softmax
                                    negative=5,  # Negative sampling used if > 0, specifies how many noise words should be drawn (usually between 5 and 20), 0 if no noise words
                                    ns_exponent=0.75,  # Exponent used to shape negative sampling distribution (popular default of 0.75)
                                    cbow_mean=0,  # 0 to use sum of context word vectors, 1 to use mean (only applies when CBOW is used)
                                    alpha=0.025,  # Initial learning rate
                                    seed=0,  # Seed for RNG
                                    max_vocab_size=None,  # Limits RAM during vocabulary building (None for no limit)
                                    max_final_vocab=None,  # Limits the vocab to a target vocab size
                                    sample=0.001,  # TODO: This might not be a good value
                                    hashfxn=None,
                                    epochs=1000,
                                    trim_rule=None,
                                    sorted_vocab=0,  # If 1, sort the vocabulary by descending frequency before assigning word indexes
                                    batch_words=10000,  # Target size (in words) for batches of examples passed to worker threads
                                    compute_loss=True,
                                    callbacks=[self.epoch_callback],
                                    shrink_windows=None)
        
        self.model.build_vocab()

        self.model.save("word2vec.model")

    def train(self):
        pass

    # TODO
    # @classmethod
    # def load(cls, path: str):
    #     self.model = GensimWord2Vec.load(path)

    def save(self, path: str):
        pass


if __name__ == "__main__":
    for text in common_texts:
        print(text)
    w2v = Word2Vec()

