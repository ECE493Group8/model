import argparse
import os
import shutil
import unittest

import polars as pl

from train import main


class TrainTest(unittest.TestCase):
    TEST_DIRECTORY = "./tests/train_test"

    def setUp(self) -> None:
        self.valid_column = "column1"
        if not os.path.exists(TrainTest.TEST_DIRECTORY):
            os.makedirs(TrainTest.TEST_DIRECTORY)

        self.df1 = pl.DataFrame(
            {self.valid_column: [["when", "he", "was"], ["thirteen", "my"], ["brother", "jem"]]})
        self.df2 = pl.DataFrame(
            {self.valid_column: [["got", "his"], ["arm"]]})

        self.df1.write_parquet(
            os.path.join(TrainTest.TEST_DIRECTORY, "df1.parquet"))
        self.df2.write_parquet(
            os.path.join(TrainTest.TEST_DIRECTORY, "df2.parquet"))

    def test_train_smoke(self):
        args = argparse.Namespace()
        args.directory = os.path.join(TrainTest.TEST_DIRECTORY, "train")
        args.workers = 1
        args.epochs = 2
        args.vector_size = 16
        args.parquet_path = os.path.join(TrainTest.TEST_DIRECTORY, "*.parquet")
        args.column = self.valid_column
        args.build_vocab_progress_per = 10

        main(args)

        self.assertTrue(os.path.exists(os.path.join(TrainTest.TEST_DIRECTORY, "train", "start.model")))
        self.assertTrue(os.path.exists(os.path.join(TrainTest.TEST_DIRECTORY, "train", "model_1.model")))
        self.assertTrue(os.path.exists(os.path.join(TrainTest.TEST_DIRECTORY, "train", "model_2.model")))

    def tearDown(self) -> None:
        shutil.rmtree(TrainTest.TEST_DIRECTORY)


if __name__ == "__main__":
    unittest.main()
