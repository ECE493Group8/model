import unittest

from gensim.test.utils import common_texts

from models.word2vec import MalamudWord2Vec


class Word2VecTest(unittest.TestCase):
    def test_malamud_word2vec_train(self):
        # NOTE: MalamudWord2Vec is essentially just a type alias for Gensim's
        #  Word2Vec model. As a result, we don't test it extensively, but just
        #  ensure that each argument passed to the methods of MalamudWord2Vec
        #  work as intended.
        m = MalamudWord2Vec(workers=1, epochs=1, vector_size=16)
        m.build_vocab(corpus_iterable=common_texts)
        m.train(dataset=common_texts, callbacks=[], report_delay=None)

    def test_malamud_word2vec_invalid_workers(self):
        with self.assertRaises(AssertionError):
            MalamudWord2Vec(workers=0, epochs=1, vector_size=16)

    def test_malamud_word2vec_invalid_epochs(self):
        with self.assertRaises(AssertionError):
            MalamudWord2Vec(workers=1, epochs=0, vector_size=16)

    def test_malamud_word2vec_invalid_vector_size(self):
        with self.assertRaises(AssertionError):
            MalamudWord2Vec(workers=1, epochs=1, vector_size=0)        

    def test_malamud_word2vec_train_invalid_dataset(self):
        with self.assertRaises(TypeError):
            m = MalamudWord2Vec(workers=1, epochs=1, vector_size=16)
            m.build_vocab(corpus_iterable=common_texts)
            m.train(dataset=None, callbacks=[], report_delay=None)

    def test_malamud_word2vec_train_invalid_callbacks(self):
        with self.assertRaises(TypeError):
            m = MalamudWord2Vec(workers=1, epochs=1, vector_size=16)
            m.build_vocab(corpus_iterable=common_texts)
            m.train(dataset=common_texts, callbacks=None, report_delay=None)


if __name__ == "__main__":
    unittest.main()
