import os
import unittest
import unittest.mock

from gensim.models import Word2Vec

from utils.callbacks import EpochLogger, EpochSaver


class EpochSaverTest(unittest.TestCase):
    TEST_DIRECTORY = "./tests/epoch_saver_test"
    TRAINING_LOSS = 0.5
    SAVE_FREQUENCY = 2

    def setUp(self) -> None:
        if not os.path.exists(EpochSaverTest.TEST_DIRECTORY):
            os.makedirs(EpochSaverTest.TEST_DIRECTORY)

        self.epoch_saver = EpochSaver(EpochSaverTest.TEST_DIRECTORY,
                                      EpochSaverTest.SAVE_FREQUENCY)

        self.model_mock = Word2Vec()
        self.model_mock.save = unittest.mock.MagicMock()
        self.model_mock.get_latest_training_loss = \
            unittest.mock.MagicMock(return_value=EpochSaverTest.TRAINING_LOSS)

    def test_epoch_saver_on_train_begin(self):
        self.epoch_saver.on_train_begin(self.model_mock)
        self.model_mock.save.assert_called_once_with(
            os.path.join(EpochSaverTest.TEST_DIRECTORY, "start.model"))

    def test_epoch_saver_on_epoch_begin(self):
        last_epoch = self.epoch_saver.epoch
        self.epoch_saver.on_epoch_begin(self.model_mock)
        this_epoch = self.epoch_saver.epoch
        self.assertEqual(this_epoch, last_epoch + 1)

    def test_epoch_saver_on_epoch_end(self):
        self.assertEqual(self.epoch_saver.best_loss, float("inf"))
        self.assertEqual(self.epoch_saver.loss, 0.)

        self.epoch_saver.on_epoch_end(self.model_mock)
        self.model_mock.save.has_calls([
            unittest.mock.call(os.path.join(EpochSaverTest.TEST_DIRECTORY, "model_0.epoch")),
            unittest.mock.call(os.path.join(EpochSaverTest.TEST_DIRECTORY, "best.model")),
        ])
        self.epoch_saver.on_epoch_end(self.model_mock)  # No new best loss and no frequent save.
        self.model_mock.save.has_calls([
            unittest.mock.call(os.path.join(EpochSaverTest.TEST_DIRECTORY, "model_0.epoch")),
            unittest.mock.call(os.path.join(EpochSaverTest.TEST_DIRECTORY, "best.model")),
        ])
        self.epoch_saver.on_epoch_end(self.model_mock)  # A frequent save.
        self.model_mock.save.has_calls([
            unittest.mock.call(os.path.join(EpochSaverTest.TEST_DIRECTORY, "model_0.epoch")),
            unittest.mock.call(os.path.join(EpochSaverTest.TEST_DIRECTORY, "best.model")),
            unittest.mock.call(os.path.join(EpochSaverTest.TEST_DIRECTORY, "model_2.epoch")),
        ])


class EpochLoggerTest(unittest.TestCase):
    TRAINING_LOSS = 0.5

    def setUp(self) -> None:
        self.epoch_logger = EpochLogger()

        self.model_mock = Word2Vec()
        self.model_mock.save = unittest.mock.MagicMock()
        self.model_mock.get_latest_training_loss = \
            unittest.mock.MagicMock(return_value=EpochLoggerTest.TRAINING_LOSS)

    def test_epoch_logger_on_epoch_begin(self):
        last_epoch = self.epoch_logger.epoch
        self.epoch_logger.on_epoch_begin(self.model_mock)
        this_epoch = self.epoch_logger.epoch
        self.assertEqual(this_epoch, last_epoch + 1)

    def test_epoch_logger_on_epoch_end(self):
        self.assertEqual(self.epoch_logger.loss, 0.)
        self.epoch_logger.on_epoch_end(self.model_mock)
        self.model_mock.get_latest_training_loss.assert_called_once()
        self.assertEqual(self.epoch_logger.loss, EpochLoggerTest.TRAINING_LOSS)


if __name__ == "__main__":
    unittest.main()
