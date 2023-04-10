import os
import shutil
import unittest

import polars as pl

from data.malamud_dataset import MalamudDataset


class MalamudDatasetTest(unittest.TestCase):
    TEST_DIRECTORY = "./tests/malamud_dataset_test"

    def setUp(self) -> None:
        self.valid_parquet_path = \
            os.path.join(MalamudDatasetTest.TEST_DIRECTORY, "*.parquet")
        self.valid_column = "column1"
        if not os.path.exists(MalamudDatasetTest.TEST_DIRECTORY):
            os.makedirs(MalamudDatasetTest.TEST_DIRECTORY)

        self.df1 = pl.DataFrame(
            {self.valid_column: ["df1_data1", "df1_data2", "df1_data3"]})
        self.df2 = pl.DataFrame(
            {self.valid_column: ["df2_data1", "df2_data2"]})

        self.df1.write_parquet(
            os.path.join(MalamudDatasetTest.TEST_DIRECTORY, "df1.parquet"))
        self.df2.write_parquet(
            os.path.join(MalamudDatasetTest.TEST_DIRECTORY, "df2.parquet"))

    def test_malamud_dataset_invalid_column(self):
        m = MalamudDataset(self.valid_parquet_path, "invalid_column")
        with self.assertRaises(pl.ColumnNotFoundError):
            for _ in m:
                pass

    def test_malamud_dataset_invalid_path(self):
        m = MalamudDataset(MalamudDatasetTest.TEST_DIRECTORY, self.valid_column)
        with self.assertRaises(IsADirectoryError):
            for _ in m:
                pass

    def test_malamud_dataset_no_files(self):
        os.remove(os.path.join(MalamudDatasetTest.TEST_DIRECTORY, "df1.parquet"))
        os.remove(os.path.join(MalamudDatasetTest.TEST_DIRECTORY, "df2.parquet"))
        m = MalamudDataset(self.valid_parquet_path, self.valid_column)
        i = 0
        for _ in m:
            i += 1
        self.assertEqual(0, i)

    def test_malamud_dataset_one_file(self):
        os.remove(os.path.join(MalamudDatasetTest.TEST_DIRECTORY, "df2.parquet"))
        m = MalamudDataset(self.valid_parquet_path, self.valid_column)
        i = 0
        for d in m:
            self.assertEqual(self.df1[self.valid_column][i], d)
            i += 1
        self.assertEqual(len(self.df1[self.valid_column]), i)

    def test_malamud_dataset_two_files(self):
        m = MalamudDataset(self.valid_parquet_path, self.valid_column)
        i = 0
        for _ in m:
            i += 1
        expected = len(self.df1[self.valid_column]) + len(self.df2[self.valid_column])
        self.assertEqual(expected, i)

    def test_malmaud_dataset_multiple_iterations(self):
        m = MalamudDataset(self.valid_parquet_path, self.valid_column)
        i = 0
        for _ in m:
            i += 1
        for _ in m:
            i += 1
        expected = 2 * (len(self.df1[self.valid_column]) + len(self.df2[self.valid_column]))
        self.assertEqual(expected, i)

    def tearDown(self) -> None:
        shutil.rmtree(MalamudDatasetTest.TEST_DIRECTORY)


if __name__ == "__main__":
    unittest.main()
