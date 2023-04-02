import argparse
import glob
import logging
import multiprocessing as mp
import os
import time
from typing import Iterable, List, Tuple

from gensim.models.phrases import FrozenPhrases, Phraser, Phrases
from gensim.parsing.preprocessing import remove_stopword_tokens
from gensim.utils import simple_preprocess, to_unicode
import polars as pl

from stopwords import STOPWORDS

logger = logging.getLogger(__name__)

FROZEN_PHRASES_FILE_PREFIX = "frozen_phrases"
PHRASES_MIN_COUNT = 5
PHRASES_THRESHOLD = 10
PHRASES_MAX_VOCAB_SIZE = 100000000
PHRASES_PROGRESS_PER = 10000


def merge_phrases(save_path: str) -> FrozenPhrases:
    frozen_phrases_files = glob.glob(
            os.path.join(save_path, f"{FROZEN_PHRASES_FILE_PREFIX}*"))
    
    frozen_phrases = FrozenPhrases.load(frozen_phrases_files[0])
    for i, frozen_phrase_file in enumerate(frozen_phrases_files):
        logger.warn(f"loading frozen phrase mode {i}")
        temp_frozen_phrases = FrozenPhrases.load(frozen_phrase_file)
        frozen_phrases.phrasegrams.update(temp_frozen_phrases.phrasegrams)

    return frozen_phrases


def create_phrases_process(
    rank: int,
    skip_rows: int,
    rows_in_mem: int,
    save_path: str,
    data_path: str,
):
    frozen_phrases_file = os.path.join(
            save_path, f"{FROZEN_PHRASES_FILE_PREFIX}_{rank}.model")

    phrases = Phrases(sentences=None,
                      min_count=PHRASES_MIN_COUNT,
                      threshold=PHRASES_THRESHOLD,
                      max_vocab_size=PHRASES_MAX_VOCAB_SIZE,
                      progress_per=PHRASES_PROGRESS_PER)

    def register_sentence(sentences: Tuple[List[str]]):
        phrases.add_vocab(sentences)
        return sentences

    _ = (
        pl.read_csv(
            source=data_path,
            separator='\t',
            new_columns=[
                "dkey",
                "ngram",
                "ngram_lc",
                "ngram_tokens",
                "ngram_count",
                "term_freq",
                "doc_count",
                "insert_date",
            ],
            dtypes={
                "dkey": pl.Utf8,
                "ngram": pl.Utf8,
                "ngram_lc": pl.Utf8,
                "ngram_tokens": pl.UInt32,
                "ngram_count": pl.UInt32,
                "term_freq": pl.Float64,
                "doc_count": pl.UInt32,
                "insert_date": pl.Utf8,
            },
            skip_rows=(skip_rows + rank * rows_in_mem),
            n_rows=rows_in_mem,
            quote_char=None,
            encoding='utf8'
        )
        .head(n=rows_in_mem)
        .filter(pl.col("ngram_tokens") > 1)
        .with_columns(pl.col("ngram_lc").str.split(" ").alias("ngram_lc_split"))
        .select("ngram_lc_split")
        .apply(register_sentence, return_dtype=pl.List)
    )

    frozen_phrases = Phraser(phrases)
    frozen_phrases.save(frozen_phrases_file)


def create_phrases(
    n_processes: int,
    skip_rows: int,
    n_rows: int,
    rows_in_mem: int,
    save_path: str,
    data_path: str,
    verbose: bool = False,
):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        filename=os.path.join(save_path, "create_phrases.log"),
                        level=logging.INFO if verbose else logging.WARN,
                        datefmt="%Y-%m-%d %H:%M:%S")

    # Create the phrases in parallel by reading different chunks of the file at
    # a time.
    process_pool = mp.Pool(processes=n_processes)
    start_time = time.time()
    process_pool.starmap(create_phrases_process,
                         [
                            (i, skip_rows, rows_in_mem, save_path, data_path)
                            for i in range(n_rows // rows_in_mem)
                         ])
    end_time = time.time()
    logger.warning(f"time to create phrases = {end_time - start_time}")

    # Merge all the frozen phrases into one.
    start_time = time.time()
    frozen_phrases_file = \
            os.path.join(save_path, f"{FROZEN_PHRASES_FILE_PREFIX}_merged.model")
    frozen_phrases = merge_phrases(save_path)
    frozen_phrases.save(frozen_phrases_file)
    end_time = time.time()
    logger.warning(f"time to merge phrases = {end_time - start_time}")


rows_preprocessed = 0
phraser = FrozenPhrases.load("./phraser/frozen_phrases_merged.model")


def custom_remove_words(s, stopwords):
    s = to_unicode(s)
    s = remove_stopword_tokens(s.split(), stopwords)
    return s


def preprocess(df: pl.DataFrame) -> List[str]:
    def _preprocess(str_itr: Iterable[str]) -> Iterable[str]:
        preprocessed_str = phraser[str_itr]
        preprocessed_str = " ".join(preprocessed_str)
        preprocessed_str = simple_preprocess(preprocessed_str)
        preprocessed_str = " ".join(preprocessed_str)
        preprocessed_str = custom_remove_words(preprocessed_str, STOPWORDS)
        if len(preprocessed_str):
            return preprocessed_str
        return [""]

    # Log the progress.
    global rows_preprocessed
    rows_preprocessed_before = rows_preprocessed
    rows_preprocessed += df.shape[0]
    logger.warning(f"preprocessing rows {rows_preprocessed_before} to {rows_preprocessed}")

    # Perform the preprocessing.
    df = df.select(pl.col("keywords_lc").apply(_preprocess, return_dtype=pl.List(pl.Utf8)))
    df = df.filter(~pl.col("keywords_lc").arr.contains(""))

    return df


def write_parquet(
    skip_rows: int,
    n_rows: int,
    save_path: str,
    save_tag: str,
    data_path: str,
    verbose: bool = False,
):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        filename=os.path.join(save_path, f"{save_tag}.log"),
                        level=logging.INFO if verbose else logging.WARN,
                        datefmt="%Y-%m-%d %H:%M:%S")

    logger.warning("beginning writing parquet")
    start_time = time.time()
    _ = (
        pl.scan_csv(
            source=data_path,
            separator='\t',
            new_columns=[
                "dkey",
                "keywords",
                "keywords_lc",
                "keyword_tokens",
                "keyword_score",
                "doc_count",
                "insert_date",
            ],
            dtypes={
                "dkey": pl.Utf8,
                "keywords": pl.Utf8,
                "keywords_lc": pl.Utf8,
                "keyword_tokens": pl.UInt32,
                "keyword_score": pl.UInt32,
                "doc_count": pl.UInt32,
                "insert_date": pl.Utf8,
            },
            skip_rows=skip_rows,
            n_rows=n_rows,
            quote_char=None,
            encoding='utf8'
        )
        .head(n=n_rows)

        # <initial-splitting>
        .filter(~pl.col("keywords_lc").is_null())  # Remove null values.
        .select(pl.col("keywords_lc").str.split(" "))
        # </initial-splitting>

        # <preprocessing>
        .map(lambda x: preprocess(x), streamable=True)
        # </preprocessing>

        .sink_parquet(os.path.join(save_path, f"{save_tag}.parquet"))
    )
    end_time = time.time()
    logger.warning(f"time to write parquet: {end_time - start_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    phrase_parser = subparsers.add_parser("phrases")
    phrase_parser.add_argument("--n-processes", type=int, required=True)
    phrase_parser.add_argument("--skip-rows", type=int, required=True)
    phrase_parser.add_argument("--n-rows", type=int, required=True)
    phrase_parser.add_argument("--rows-in-mem", type=int, required=True)
    phrase_parser.add_argument("--save-path", type=str, required=True)
    phrase_parser.add_argument("--data-path", type=str, required=True)

    parquet_parser = subparsers.add_parser("parquet")
    parquet_parser.add_argument("--skip-rows", type=int, required=True)
    parquet_parser.add_argument("--n-rows", type=int, required=True)
    parquet_parser.add_argument("--save-path", type=str, required=True)
    parquet_parser.add_argument("--save-tag", type=str, required=True)
    parquet_parser.add_argument("--data-path", type=str, required=True)

    args = parser.parse_args()

    if args.command == "phrases":
        create_phrases(n_processes=args.n_processes,
                       skip_rows=args.skip_rows,
                       n_rows=args.n_rows,
                       rows_in_mem=args.rows_in_mem,
                       save_path=args.save_path,
                       data_path=args.data_path)
    elif args.command == "parquet":
        write_parquet(skip_rows=args.skip_rows,
                      n_rows=args.n_rows,
                      save_path=args.save_path,
                      save_tag=args.save_tag,
                      data_path=args.data_path)
    else:
        raise ValueError(f"must use a command: {list(subparsers.choices.keys())}")
