from typing import List, Iterable

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec


class MalamudWord2Vec(Word2Vec):
    MAX_NGRAM_SIZE = 5
    MIN_NGRAM_SIZE = 1

    def __init__(self,
                 workers: int,
                 epochs: int,
                 vector_size: int):
        super().__init__(vector_size=vector_size,
                         window=MalamudWord2Vec.MAX_NGRAM_SIZE,
                         min_count=MalamudWord2Vec.MIN_NGRAM_SIZE,  # Ignore all words with total frequency lower than this
                         workers=workers,
                         sg=1,  # 0 for skip-gram, 1 for CBOW
                         hs=0,  # 0 for negative sampling, 1 for hierarchical softmax
                         negative=5,  # Negative sampling used if > 0, specifies how many noise words should be drawn (usually between 5 and 20), 0 if no noise words
                         ns_exponent=0.75,  # Exponent used to shape negative sampling distribution (popular default of 0.75)
                         cbow_mean=0,  # 0 to use sum of context word vectors, 1 to use mean (only applies when CBOW is used)
                         alpha=0.025,  # Initial learning rate
                         min_alpha=0.0001,
                         max_vocab_size=None,  # Limits RAM during vocabulary building (None for no limit)
                         max_final_vocab=10_000_000,  # Limits the vocab to a target vocab size
                         sample=1e-3,
                         hashfxn=hash,
                         epochs=epochs,
                         trim_rule=None,  # This specifies whether certain words should remain in the vocabulary, be trimmed away, or handled using the default (None => discarded if word count < min_count).
                         sorted_vocab=0,  # If 1, sort the vocabulary by descending frequency before assigning word indexes
                         shrink_windows=True,
                         null_word=0,
                         seed=1)

    def train(self,
              dataset: Iterable,
              callbacks: List[CallbackAny2Vec],
              report_delay: float):
            #   save_frequency: float):
        # self.save_counter = 0
        # self.save_every = int(save_frequency // report_delay)
        # assert self.save_every >= 1
        super().train(corpus_iterable=dataset,
                      epochs=self.epochs,
                      total_examples=self.corpus_count,
                      compute_loss=True,
                      callbacks=callbacks,
                      report_delay=report_delay)

    def _log_progress(self, job_queue, progress_queue, cur_epoch, example_count,
                      total_examples, raw_word_count, total_words,
                      trained_word_count, elapsed):
        # self.save_counter += 1
        # if self.save_counter % self.save_every == 0:
        #     self.save()
        super()._log_progress(job_queue, progress_queue, cur_epoch,
                              example_count, total_examples, raw_word_count,
                              total_words, trained_word_count, elapsed)
