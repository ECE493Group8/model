import logging
import os
import shutil
import unittest

from gensim.models import Word2Vec
from gensim.test.utils import common_texts

from utils.callbacks import EpochLogger, EpochSaver


class CallbacksTest(unittest.TestCase):
    TEST_DIRECTORY = "./tests/callbacks_test"

    def setUp(self) -> None:
        os.makedirs(CallbacksTest.TEST_DIRECTORY)

    def test_epoch_callback(self):
        logging.basicConfig(
                filename=os.path.join(CallbacksTest.TEST_DIRECTORY, "test.log"),
                level=logging.INFO)

        epoch_saver = EpochSaver(CallbacksTest.TEST_DIRECTORY, 1)
        epoch_logger = EpochLogger()

        model = Word2Vec(window=5,
                         min_count=1,
                         workers=4,
                         epochs=10,
                         compute_loss=True,
                         seed=1)  # Runnning with seed=1 ensures that a best model will be saved.
        model.build_vocab(common_texts, progress_per=1)
        model.train(
                common_texts, total_examples=model.corpus_count,
                epochs=model.epochs, compute_loss=True,
                callbacks=[epoch_logger, epoch_saver])

        self.assertTrue(os.path.exists(CallbacksTest.TEST_DIRECTORY))
        self.assertTrue(os.path.exists(os.path.join(CallbacksTest.TEST_DIRECTORY, f"start.model")))
        self.assertTrue(os.path.exists(os.path.join(CallbacksTest.TEST_DIRECTORY, f"best.model")))
        for i in range(1, 10 + 1):
            self.assertTrue(os.path.exists(os.path.join(CallbacksTest.TEST_DIRECTORY, f"model_{i}.model")))

    def tearDown(self) -> None:
        shutil.rmtree(CallbacksTest.TEST_DIRECTORY)


if __name__ == "__main__":
    unittest.main()
