import argparse
import glob
import logging
from io import StringIO
import multiprocessing as mp
import os
import time
from timeit import default_timer as timer
from typing import List, Tuple

# from dask.distributed import Client
# import dask.dataframe as dd
# from dotenv import load_dotenv
from gensim.models.phrases import FrozenPhrases, Phraser, Phrases
from gensim.models.word2vec import Text8Corpus
from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import simple_preprocess
import polars as pl
# import psycopg2

from data.amazon_dataset import AmazonDataset
from data.small_dataset import LineSentence
from stopwords import STOPWORDS

# load_dotenv()

# POSTGRES_URI = os.getenv("PG_URI")

logger = logging.getLogger(__name__)


class Preprocessor:
    # TODO: Change query.
    # TODO: Get those with count > 0?
    # QUERY = (
    #     """
    #     SELECT ngram_lc FROM docs.doc_ngrams_0 WHERE ngram_lc LIKE '%back pain%';
    #     """
    # )
    QUERY = (
        """
        SELECT ngram_lc FROM docs.doc_ngrams_0 WHERE ngram_tokens > 1;
        """
    )

    QUERY_LIMITED = (
        """
        SELECT ngram_lc FROM docs.doc_ngrams_0 WHERE ngram_lc LIKE '%back pain%';
        """
    )
    QUERY_OFFSET_LIMITED = (
        """
        SELECT ngram_lc FROM docs.doc_ngrams_0 WHERE ngram_lc LIKE '%%back pain%%' LIMIT %s OFFSET %s;
        """
    )
    # TODO: Change query.
    QUERY_OFFSET_LIMITED_ROWNUM = (
        """
        SELECT ngram_lc FROM (
            SELECT ngram_lc, row_number() over() AS row_num FROM docs.doc_ngrams_0
        ) t2 WHERE ngram_lc LIKE '%%back pain%%' AND row_num %% %s = %s;
        """
    )
    # TODO
    INSERT = (
        """
        """
    )
    # TODO: Change name of table.
    TABLE = (
        """
        CREATE TABLE ngram_data (
            ngram text
        );
        """
    )

    def __init__(self,
                 user: str,
                 password: str,
                 host: str,
                 database: str,
                 min_count: int,
                 cursor_itersize: int,
                 cursor_fetchsize: int,
                 phrases_progress_per: int = 100000):
        self.db_connection = psycopg2.connect(
            host=host, database=database, user=user, password=password)
        self.min_count = min_count
        self.cursor_itersize = cursor_itersize
        self.cursor_fetchsize = cursor_fetchsize
        self.phrases_progress_per = phrases_progress_per

    def preprocess(self):
        logger.info("creating table...")
        start_time = timer()
        self._create_data_table()
        logger.info(f"creating table took {timer() - start_time}")

        logger.info("generating phrases...")
        start_time = timer()
        frozen_phrases = self._create_frozen_phrases()
        logger.info(f"generating phrases took {timer() - start_time}")

        logger.info("writing phrases to table...")
        start_time = timer()
        self._write_phrases(frozen_phrases)
        logger.info(f"writing phrases to table took {timer() - start_time}")

    def _create_data_table(self):
        cursor = self.db_connection.cursor()
        cursor.execute(Preprocessor.TABLE)
        self.db_connection.commit()
        cursor.close()

    def _create_frozen_phrases(self) -> FrozenPhrases:
        cursor = self.db_connection.cursor("cursor-create-frozen-phrases")
        cursor.itersize = self.cursor_itersize
        cursor.execute(Preprocessor.QUERY)

        # TODO: Parameters
        # TODO: Change max vocab size for larger
        # for 100M, we got around 10M vocab
        phrases = Phrases(sentences=None,
                          min_count=5,  # Parameter from previous group's work.
                          threshold=10,  # Parameter from previous group's work.
                          progress_per=self.phrases_progress_per)

        while True:
            rows = cursor.fetchmany(self.cursor_fetchsize)
            if not rows:
                break
            rows = [row[0].split(" ") for row in rows]
            phrases.add_vocab(rows)
            # for row in rows:
            #     row_str = row[0]
            #     phrases.add_vocab([row_str.split(" ")])
        cursor.close()

        return Phraser(phrases)

    def _write_phrases(self, frozen_phrases: FrozenPhrases):
        # write_cursor = self.db_connection.cursor()
        # read_cursor = self.db_connection.cursor("cursor-read-phrases")
        # read_cursor.itersize = self.cursor_itersize
        # read_cursor.execute(Preprocessor.QUERY)

        N_PROCESSES = 4
        processes = [
            mp.Process(
                target=self._write_phrases_parallel,
                args=(i,
                      N_PROCESSES,  # TODO: Number of processes.
                      self.cursor_itersize,
                      self.cursor_fetchsize,
                      100000000,  # TODO: Make variable.
                      frozen_phrases))
            for i in range(N_PROCESSES)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join()

        # while True:
        #     rows = read_cursor.fetchmany(self.cursor_fetchsize)
        #     if not rows:
        #         break
        #     memory_file = StringIO()
        #     insert_values = []
        #     for row in rows:
        #         row_str = row[0]
        #         phrased = " ".join(frozen_phrases[row_str.split(" ")])
        #         phrased = " ".join(simple_preprocess(phrased))
        #         phrased = remove_stopwords(phrased, STOPWORDS)
        #         if len(phrased) > 0:
        #             insert_values.append(f"{phrased}\n")
        #     # https://stackoverflow.com/questions/47116877/efficiently-insert-massive-amount-of-rows-in-psycopg2
        #     memory_file.writelines(insert_values)
        #     memory_file.seek(0)
        #     write_cursor.copy_from(memory_file, "ngram_data", columns=("ngram",))

        # read_cursor.close()
        # write_cursor.close()
        # self.db_connection.commit()

    def _write_phrases_parallel(self,
                                id: int,
                                n_processes: int,
                                itersize: int,
                                fetchsize: int,
                                query_size: int,
                                frozen_phrases: FrozenPhrases):
        start_time = timer()

        # start_index = id * (query_size // n_processes)
        # if id < n_processes - 1:
        #     end_index = (id + 1) * (query_size // n_processes) - 1
        # else:
        #     end_index = query_size - 1
        # current_index = 0

        # query_limit = query_size // n_processes
        # query_offset = id * (query_limit)

        local_db_connection = psycopg2.connect(
            host="localhost", database="malamud", user="chris", password="12345")
        write_cursor = local_db_connection.cursor()
        read_cursor = local_db_connection.cursor(f"cursor-read-phrases-{id}")
        read_cursor.itersize = itersize
        # read_cursor.execute(Preprocessor.QUERY_OFFSET_LIMITED, (query_limit, query_offset))
        # read_cursor.execute(Preprocessor.QUERY_OFFSET_LIMITED_ROWNUM, (n_processes, id))
        read_cursor.execute(Preprocessor.QUERY)

        while True:
            rows = read_cursor.fetchmany(fetchsize)
            logger.info(f"process {id} fetched rows")
            if not rows:
                break
            memory_file = StringIO()
            insert_values = []
            for row in rows[id::n_processes]:
                row_str = row[0]
                phrased = " ".join(frozen_phrases[row_str.split(" ")])
                phrased = " ".join(simple_preprocess(phrased))
                phrased = remove_stopwords(phrased, STOPWORDS)
                if phrased.count(" ") >= self.min_count - 1:
                    # insert_values.append(f"{phrased}\n")
                    write_cursor.execute("INSERT INTO ngram_data (ngram) VALUES (%s);", (phrased,))
            # https://stackoverflow.com/questions/47116877/efficiently-insert-massive-amount-of-rows-in-psycopg2
            # memory_file.writelines(insert_values)
            # memory_file.seek(0)
            # write_cursor.copy_from(memory_file, "ngram_data", columns=("ngram",))

        # TODO: commit inside loop, not here?
        read_cursor.close()
        write_cursor.close()
        local_db_connection.commit()

        logger.info(f"process {id} took {timer() - start_time}")


FROZEN_PHRASES_FILE_PREFIX = "frozen_phrases"
PHRASES_MIN_COUNT = 5
PHRASES_THRESHOLD = 10
PHRASES_MAX_VOCAB_SIZE = 100000000
PHRASES_PROGRESS_PER = 10000


def merge_phrases(save_path: str) -> FrozenPhrases:
    frozen_phrases_files = glob.glob(
            os.path.join(save_path, f"{FROZEN_PHRASES_FILE_PREFIX}*"))
    
    frozen_phrases = FrozenPhrases.load(frozen_phrases_files[0])
    for frozen_phrase_file in frozen_phrases_files:
        temp_frozen_phrases = FrozenPhrases.load(frozen_phrase_file)
        frozen_phrases.phrasegrams.update(temp_frozen_phrases.phrasegrams)

    return frozen_phrases


def create_phrases_process(
    rank: int,
    # n_processes: int,
    skip_rows: int,
    # n_rows: int,
    rows_in_mem: int,
    save_path: str,
    data_path: str,
):    
    # chunk_size = n_rows // n_processes  # Number of lines to read in the file.
    # assert chunk_size % rows_in_mem == 0  # TODO: Explain why this should be the case.
    # disk_reads = chunk_size // rows_in_mem

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
            # skip_rows=(skip_rows + rank * chunk_size + disk_read * rows_in_mem),  # TODO
            skip_rows=(skip_rows + rank * rows_in_mem),  # TODO
            n_rows=rows_in_mem,
            quote_char=None,
            encoding='utf8'
        )
        .head(n=rows_in_mem)  # TODO
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

    # processes = [
    #     mp.Process(
    #         target=create_phrases_process,
    #         args=(
    #             i,
    #             n_processes,
    #             skip_rows,
    #             n_rows,
    #             rows_in_mem,
    #             save_path,
    #             data_path,
    #         )
    #     )
    #     for i in range(n_processes)
    # ]
    # 
    # start_time = time.time()
    # for process in processes:
    #     process.start()
    # for process in processes:
    #     process.join()
    # end_time = time.time()
    # logger.warning(f"time = {end_time - start_time}")

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

    print(frozen_phrases[["new", "york"]])
    print(frozen_phrases[["back", "pain"]])
    print(frozen_phrases[["san", "francisco"]])


PARQUET_PREFIX = "doc_ngrams_0.parquet"


def write_parquet(
    skip_rows: int,
    n_rows: int,
    save_path: str,
    data_path: str,
    verbose: bool = False,
):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        filename=os.path.join(save_path, "write_parquet.log"),
                        level=logging.INFO if verbose else logging.WARN,
                        datefmt="%Y-%m-%d %H:%M:%S")

    start_time = time.time()
    _ = (
        pl.scan_csv(
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
            skip_rows=skip_rows,
            n_rows=n_rows,
            quote_char=None,
            encoding='utf8'
        )
        .head(n=n_rows)
        .filter(pl.col("ngram_tokens") > 1)
        .with_columns(pl.col("ngram_lc").str.split(" ").alias("ngram_lc_split"))
        .select("ngram_lc_split")
        .sink_parquet(os.path.join(save_path, f"{PARQUET_PREFIX}"))
    )
    end_time = time.time()
    logger.warning(f"time to write parquet: {end_time - start_time}")


def read_parquet(save_path: str):
    df = pl.read_parquet(os.path.join(save_path, PARQUET_PREFIX))
    logger.warning(df)


if __name__ == "__main__":
    # TODO: Change filename.
    # logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    #                     filename="./run_preprocessor_1",
    #                     level=logging.INFO,
    #                     datefmt="%Y-%m-%d %H:%M:%S")

    # PARQUET_DIRECTORY = "./experiment_results"
    # # DATASET_PATH = "./experiment_data/test_malamud_small.data"
    # # DATASET_PATH = "./experiment_data/test_malamud.data"
    # DATASET_PATH = "/storage8TB/chris/malamud_100m.data"

    # if not os.path.exists(PARQUET_DIRECTORY):
    #     os.makedirs(PARQUET_DIRECTORY)

    # write_parquet(
    #     45,
    #     100000000,
    #     PARQUET_DIRECTORY,
    #     DATASET_PATH,
    # )
    # read_parquet(PARQUET_DIRECTORY)
    # exit()

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
                      data_path=args.data_path)
        # read_parquet(args.save_path)
    else:
        raise ValueError(f"must use a command: {list(subparsers.choices.keys())}")

    exit()

    # DATASET_PATH = "./test_malamud_small"
    # DATASET_PATH = "./test_malamud"
    DATASET_PATH = "/mnt/malamud_100m.data"

    # TODO: Set max vocab.
    phrases = Phrases(sentences=None,
                        min_count=5,  # Parameter from previous group's work.  TODO
                        threshold=10,  # Parameter from previous group's work.  TODO
                        progress_per=1000)  # TODO


    # DASK

    def func(s: str):
        print(s)
        return s

    # # TODO: Set workers?
    # client = Client(memory_limit="32GB")
    # df = dd.read_csv(DATASET_PATH,
    #                 sep='\t',
    #                 # dtype={
    #                 #     "dkey": str,
    #                 #     "ngram": str,
    #                 #     "ngram_lc": str,
    #                 #     "ngram_tokens": int,
    #                 #     "ngram_count": int,
    #                 #     "term_freq": float,
    #                 #     "doc_count": int,
    #                 #     "insert_date": str,
    #                 # },
    #                 dtype={
    #                     "dkey": object,
    #                     "ngram": object,
    #                     "ngram_lc": object,
    #                     "ngram_tokens": object,
    #                     "ngram_count": object,
    #                     "term_freq": object,
    #                     "doc_count": object,
    #                     "insert_date": object,
    #                 },
    #                 skiprows=46,
    #                 on_bad_lines='warn',
    #                 keep_default_na=False).head(n=1000)
    # # df.apply(func, axis=1)
    # # df.to_parquet("./test_dask_parquet")
    # # dfto_parquet("./dask_test_parquet")
    # # print(df["ngram_lc"])
    # # results = data.drop
    # print(df)
    # exit()

    # POLARS

    def phrase(x: str):
        # # for s in x:
        # #     for c in s:
        # #         print("iteration")
        # #         print(c)
        # #         print(type(c))
        # # phrases.add_vocab(x)
        # return
        print(x)
        # print(x.rows())
        # print(type(x))
        # phrases.add_vocab(x["ngram_lc_split"].tolist())
        phrases.add_vocab([item[0] for item in x.rows()])
        print("asdf")
        return x

    def preprocess(s: str):
        print(s)
        s = simple_preprocess(s[0])
        return s

    def phrase_str_list(l: List[str]):
        # print(l)
        phrases.add_vocab(l)
        return l

    def func(s: str):
        print(s)
        return s

    def func2(s: str):
        if s:
            phrases.add_vocab(s.split(" "))
        return s

    ROWS = 100000
    df = (
        pl.read_csv(source=DATASET_PATH,
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
                    # null_values="\"",
                    # comment_char='-',
                    # skip_rows=45,
                    skip_rows=1000000,
                    n_rows=ROWS,
                    quote_char=None,
                    encoding='utf8')
        .head(n=ROWS)
        # .filter(pl.col("ngram_lc").str.contains("back|pain"))
        # TODO: Select only those with ngram_tokens > 1?
        .filter(pl.col("ngram_tokens") > 1)
        .select("ngram_lc")
        # .apply(preprocess, return_dtype=pl.Utf8)
        .with_columns(pl.col("ngram_lc").str.split(" ").alias("ngram_lc_split"))
        .select("ngram_lc_split")
        .apply(phrase_str_list, return_dtype=pl.List)
        # .apply(func, pl.List)
    )
    print(df)
    print(df.shape)
    print("done loading dataframe")
    # df.apply(phrase_str_list, return_dtype=pl.List)
    print("done applying function")
    frozen_phrases = Phraser(phrases)
    frozen_phrases.save("./frozen_phrases.model")
    print(f"'back pain' = '{' '.join(frozen_phrases[['back', 'pain']])}'")

    exit()

    # TODO: Remove end of file.
    df = (
        pl.scan_csv(source=DATASET_PATH,
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
                        "ngram_count": pl.Float64,
                        "term_freq": pl.Float64,
                        "doc_count": pl.UInt32,
                        "insert_date": pl.Utf8,
                    },
                    null_values="\\N",
                    comment_char='-',
                    skip_rows=45,
                    encoding='utf8-lossy',
                    ignore_errors=True)
        .filter(pl.col("ngram_tokens") > 1)
        .select("ngram_lc")
        .collect(streaming=True)
        # .apply(lambda t: func(t[0]), return_dtype=pl.Utf8)
        .apply(lambda t: " ".join(simple_preprocess(t[0])))
        # .apply(lambda t: func(t[0]))
        # .apply(lambda t: func2(t[0]))



        # .with_columns(pl.col("ngram_lc").str.split(" ").alias("ngram_lc_split"))
        # .select("ngram_lc_split")
        # # .map(phrase)
        # # .with_columns([
        # #     pl.col("ngram_lc_split").map(lambda s: phrases.add_vocab(s.to_numpy())).alias("none")
        # # ])
        # .collect(streaming=True)
    )
    print(df)
    frozen_phrases = Phraser(phrases)
    print(f"back pain = {' '.join(frozen_phrases[['back', 'pain']])}")
    # print(df.select(pl.count()).collect(streaming=True));
    exit()

    df = (
        pl.scan_csv(source=DATASET_PATH,
                    has_header=False,
                    separator='\t',
                    comment_char='-',
                    skip_rows=46,
                    new_columns=["dkey",
                                 "ngram",
                                 "ngram_lc",
                                 "ngram_tokens",
                                 "ngram_count",
                                 "term_freq",
                                 "doc_count",
                                 "insert_date"],
                    encoding='utf8-lossy')
        .filter(pl.col("ngram_tokens") > 1)
        # .select("ngram_lc")
        .with_columns(pl.col("ngram_lc").str.split(" ").alias("ngram_lc_split"))
        .select("ngram_lc_split")
        .map(phrase)
        .collect(streaming=True)
    )
    # print(df["ngram_lc"])

    frozen_phrases = Phraser(phrases)
    print(f"back pain = {' '.join(frozen_phrases[['back', 'pain']])}")

    exit()


    print(POSTGRES_URI)

    user = "chris"
    password = "12345"
    host = "localhost"
    database = "malamud"

    p = Preprocessor(user, password, host, database, 2, 10000, 10000)
    p.preprocess()

    # Load data using connectorx
    # logger.info("start")
    # df = cx.read_sql(POSTGRES_URI, "SELECT * FROM docs.doc_ngrams_0 LIMIT 1000000", partition_num=1)
    # logger.info("end")

    # Load data using psycopg2
    # TODO: Get those with count > 0?
    # QUERY = (
    #     """
    #     SELECT ngram_lc FROM docs.doc_ngrams_0 WHERE ngram_lc LIKE '%back pain%';
    #     """
    # )
    # db_connection = psycopg2.connect(
    #     host=host, database=database, user=user, password=password)
    # cursor = db_connection.cursor("cursor1")
    # cursor.itersize = 1000000
    # cursor.execute(QUERY)

    # phrases = Phrases(sentences=None,
    #                   min_count=20,  # Parameter from previous group's work.
    #                   threshold=5,  # Parameter from previous group's work.
    #                   progress_per=100)

    # while True:
    #     rows = cursor.fetchmany(10000)
    #     if not rows:
    #         break
    #     for row in rows:
    #         row_str = row[0]
    #         phrases.add_vocab([row_str.split(" ")])
    #         print(row[0])
    # cursor.close()

    # frozen_phrases = Phraser(phrases)

    # cursor = db_connection.cursor("cursor1")
    # cursor.itersize = 1000000
    # cursor.execute(QUERY)
    # with open("./phrases.txt", "w") as file:
    #     while True:
    #         rows = cursor.fetchmany(10000)
    #         if not rows:
    #             break
    #         for row in rows:
    #             row_str = row[0]
    #             phrased = " ".join(frozen_phrases[row_str.split(" ")])
    #             file.write(f"{phrased}\n")
    # cursor.close()


    exit()


    phrases = Phrases(sentences=None,
                      min_count=1,  # TODO: Change parameter
                      threshold=0.1,  # TODO: Change parameter
                      progress_per=10000)
    print("loading amazon dataset...")
    # dataset = AmazonDataset("./amazon_product_reviews.json")
    dataset = LineSentence("./test_dataset.txt")

    print("adding to vocabulary...")
    for ngram in dataset:
        phrases.add_vocab(ngram)

    frozen_phrases = Phraser(phrases)

    print("writing to text file...")
    with open("./amazon_frozen.txt", "w") as file:
        for ngram in dataset:
            processed_ngram = " ".join(frozen_phrases[ngram])
            file.write(f"{processed_ngram}\n")

    print("done!")
