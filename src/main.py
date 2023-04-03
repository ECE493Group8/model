import argparse
import logging
import os

from models.word2vec import MalamudWord2Vec
from data.malamud_dataset import MalamudDataset
from utils.callbacks import EpochLogger, EpochSaver


def main(args):
    _setup_directory(args)
    _configure_logging(args)

    # Train on a smaller dataset.
    if args.test:
        epoch_logger = EpochLogger()
        epoch_saver = EpochSaver(args.directory, frequency=1)
        dataset = MalamudDataset(
            parq_base_path=args.parq_base_path,
            num_files=args.num_files,
            column=args.column
        )
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
    parser.add_argument("--parq_base_path", type=str, default='/storage8TB/malamud-download/doc_keywords_parquets/doc_keywords_<X>.parquet')
    parser.add_argument("--num_files", type=int, default=16)
    parser.add_argument("--column", type=str, default='keywords_lc')
    parser.add_argument("--test", action="store_true")  # Used for testing.
    args = parser.parse_args()

    main(args)