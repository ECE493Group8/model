import argparse
import logging
import os

from gensim.models.word2vec import LineSentence
from dotenv import load_dotenv

from models.word2vec import MalamudWord2Vec
from data.amazon_dataset import AmazonDataset
from data.malamud_dataset import MalamudDataset
from utils.callbacks import EpochLogger, EpochSaver

# Load database connection variables (set in .env file)
load_dotenv()
POSTGRES_HOST = os.getenv('PG_HOST', 'localhost')
POSTGRES_PORT = os.getenv('PG_PORT', '5432')
POSTGRES_DBNAME = os.getenv('PG_DBNAME', 'malamud')
POSTGRES_USER = os.getenv('PG_USER')
POSTGRES_PASSWORD = os.getenv('PG_PASS')
POSTGRES_CONNECTION = (
    ('host', POSTGRES_HOST),
    ('port', POSTGRES_PORT),
    ('dbname', POSTGRES_DBNAME),
    ('user', POSTGRES_USER),
    ('password', POSTGRES_PASSWORD),
)

def main(args):
    _setup_directory(args)
    _configure_logging(args)

    # Train on a smaller dataset.
    if args.test:
        epoch_logger = EpochLogger()
        epoch_saver = EpochSaver(args.directory, frequency=1)
        dataset = MalamudDataset(
            postgres_conn_params=POSTGRES_CONNECTION,
            chunk_size=args.chunk_size,
            table=args.table,
            column=args.column,
            rows=args.rows
        )
        # dataset = AmazonDataset("./amazon_product_reviews.json")
        # dataset = LineSentence("./test_dataset.txt")  # For if you want to try an extremely small dataset.
        model = MalamudWord2Vec(workers=args.workers,
                                epochs=args.epochs,
                                vector_size=args.vector_size)
        model.build_vocab(dataset, progress_per=100000)
        model.train(dataset,
                    callbacks=[epoch_logger, epoch_saver],
                    report_delay=5.0)


def _setup_directory(args):
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)


def _configure_logging(args):
    log_file = os.path.join(args.directory, "training.log")
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        filename=log_file,
                        level=logging.INFO,
                        datefmt="%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--vector_size", type=int, required=True)
    parser.add_argument("--chunk_size", type=int, default=100000)
    parser.add_argument("--table", type=str, default='docs.doc_ngrams_0')
    parser.add_argument("--column", type=str, default='ngram_lc')
    parser.add_argument("--rows", type=int, default=float('inf')),
    parser.add_argument("--test", action="store_true")  # Used for testing.
    args = parser.parse_args()

    main(args)