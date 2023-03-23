import logging
from io import StringIO
import multiprocessing as mp
import os
from timeit import default_timer as timer

from dotenv import load_dotenv
from gensim.models.phrases import FrozenPhrases, Phraser, Phrases
from gensim.models.word2vec import Text8Corpus
from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import simple_preprocess
import psycopg2

from data.amazon_dataset import AmazonDataset
from data.small_dataset import LineSentence
from stopwords import STOPWORDS

load_dotenv()

POSTGRES_URI = os.getenv("PG_URI")

logger = logging.getLogger(__name__)


class Preprocessor:
    # TODO: Change query.
    # TODO: Get those with count > 0?
    QUERY = (
        """
        SELECT ngram_lc FROM docs.doc_ngrams_0 WHERE ngram_lc LIKE '%back pain%';
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
        phrases = Phrases(sentences=None,
                          min_count=1,  # Parameter from previous group's work.
                          threshold=1,  # Parameter from previous group's work.
                          progress_per=self.phrases_progress_per)

        while True:
            rows = cursor.fetchmany(self.cursor_fetchsize)
            if not rows:
                break
            for row in rows:
                row_str = row[0]
                phrases.add_vocab([row_str.split(" ")])
                # print(row[0])  # TODO: Remove
        cursor.close()

        return Phraser(phrases)

    def _write_phrases(self, frozen_phrases: FrozenPhrases):
        # write_cursor = self.db_connection.cursor()
        # read_cursor = self.db_connection.cursor("cursor-read-phrases")
        # read_cursor.itersize = self.cursor_itersize
        # read_cursor.execute(Preprocessor.QUERY)

        N_PROCESSES = 16
        processes = [
            mp.Process(
                target=self._write_phrases_parallel,
                args=(i,
                      N_PROCESSES,  # TODO: Number of processes.
                      self.cursor_itersize,
                      self.cursor_fetchsize,
                      798,  # TODO: Make variable.
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

        start_index = id * (query_size // n_processes)
        if id < n_processes - 1:
            end_index = (id + 1) * (query_size // n_processes) - 1
        else:
            end_index = query_size - 1
        current_index = 0

        query_limit = query_size // n_processes
        query_offset = id * (query_limit)

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
                if len(phrased) > 0:
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


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        level=logging.INFO,
                        datefmt="%Y-%m-%d %H:%M:%S")

    print(POSTGRES_URI)

    user = "chris"
    password = "12345"
    host = "localhost"
    database = "malamud"

    p = Preprocessor(user, password, host, database, 1, 10000, 1000)
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
