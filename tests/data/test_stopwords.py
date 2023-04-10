import unittest

from data.stopwords import STOPWORDS


class StopwordsTest(unittest.TestCase):
    def test_stopwords_type(self):
        # Ensure the stopwords is a set for constant-time look-up.
        self.assertIsInstance(STOPWORDS, set)

    def test_stopwords_length(self):
        # Ensure there are stopwords in the set.
        self.assertGreater(len(STOPWORDS), 0)


if __name__ == "__main__":
    unittest.main()
