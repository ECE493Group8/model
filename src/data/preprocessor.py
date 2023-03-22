import logging
import os

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
    # TODO
    INSERT = (
        """
        """
    )
    TABLE = (
        """
        CREATE TABLE IF NOT EXISTS %s (
            ngram text,
        );
        """
    )

    def __init__(self,
                 user: str,
                 password: str,
                 host: str,
                 database: str,
                 table_name: str,
                 min_count: int,
                 cursor_itersize: int,
                 cursor_fetchsize: int,
                 phrases_progress_per: int = 100000):
        self.db_connection = psycopg2.connect(
            host=host, database=database, user=user, password=password)
        self.table_name = table_name
        self.min_count = min_count
        self.cursor_itersize = cursor_itersize
        self.cursor_fetchsize = cursor_fetchsize
        self.phrases_progress_per = phrases_progress_per

    def preprocess(self):
        cursor = self.db_connection.cursor("cursor-create-table")
        cursor.execute(Preprocessor.TABLE, (self.table_name,))

        frozen_phrases = self._create_frozen_phrases()
        self._write_phrases(frozen_phrases)

    def _create_frozen_phrases(self) -> FrozenPhrases:
        cursor = self.db_connection.cursor("cursor-create-frozen-phrases")
        cursor.itersize = self.cursor_itersize
        cursor.execute(Preprocessor.QUERY, (self.table_name,))

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
                print(row[0])  # TODO: Remove
        cursor.close()

        return Phraser(phrases)

    def _write_phrases(self, frozen_phrases: FrozenPhrases):
        read_cursor = self.db_connection.cursor("cursor-write-phrases")
        read_cursor.itersize = self.cursor_itersize
        read_cursor.execute(Preprocessor.QUERY)
        with open("./phrases.txt", "w") as file:
            while True:
                rows = read_cursor.fetchmany(self.cursor_fetchsize)
                if not rows:
                    break
                for row in rows:
                    row_str = row[0]
                    phrased = " ".join(frozen_phrases[row_str.split(" ")])
                    phrased = " ".join(simple_preprocess(phrased))
                    phrased = remove_stopwords(phrased, STOPWORDS)
                    if len(phrased) > 0:
                        file.write(f"{phrased}\n")
        read_cursor.close()




if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        level=logging.INFO,
                        datefmt="%Y-%m-%d %H:%M:%S")

    print(POSTGRES_URI)

    user = "chris"
    password = "12345"
    host = "localhost"
    database = "malamud"

    p = Preprocessor(user, password, host, database, "my_table", 1, 10000, 1000)
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
