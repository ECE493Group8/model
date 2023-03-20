import logging
import os

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

logger = logging.getLogger(__name__)


class EpochSaver(CallbackAny2Vec):
    def __init__(self, directory: str, frequency: int):
        self.epoch = 0
        self.loss = 0.
        self.best_loss = float("inf")
        self.directory = directory
        self.frequency = frequency

    def on_epoch_begin(self, model: Word2Vec):
        self.epoch += 1

    def on_epoch_end(self, model: Word2Vec):
        # This get_latest_training_loss if the accumulated loss.
        self.loss = model.get_latest_training_loss() - self.loss
        if self.loss < self.best_loss:
            # Save the best model.
            logger.info(f"EPOCH {self.epoch}: saving new best model")
            model.save(os.path.join(self.directory, "best.model"))
            self.best_loss = self.loss
        if self.epoch % self.frequency == 0:
            # Save at the given frequency.
            logger.info(f"EPOCH {self.epoch}: saving model")
            model.save(os.path.join(self.directory, f"model_{self.epoch}.model"))


class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.loss = 0.

    def on_epoch_begin(self, model: Word2Vec):
        self.epoch += 1

    def on_epoch_end(self, model: Word2Vec):
        # This get_latest_training_loss if the accumulated loss.
        self.loss = model.get_latest_training_loss() - self.loss
        logger.info(f"EPOCH {self.epoch}: loss = {self.loss}")
