import argparse
import logging
import os

from gensim.models.word2vec import LineSentence
from dotenv import load_dotenv

from models.word2vec import MalamudWord2Vec
from data.amazon_dataset import AmazonDataset
from data.malamud_dataset import MalamudDataset
from utils.callbacks import EpochLogger, EpochSaver

# Load database connection string (set in .env file)
load_dotenv()
POSTGRES_URI = os.getenv('PG_URI')

def main(args):
    _setup_directory(args)
    _configure_logging(args)

    # Train on a smaller dataset.
    if args.test:
        epoch_logger = EpochLogger()
        epoch_saver = EpochSaver(args.directory, frequency=1)
        dataset = MalamudDataset(postgres_uri=POSTGRES_URI)
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
    parser.add_argument("--test", action="store_true")  # Used for testing.
    args = parser.parse_args()

    main(args)