import unittest
import filecmp
import shutil

# pylint: disable=import-error
import constants as const
import module_2.preparations.gz_to_json_files as gz_json_files
import module_2.preparations.json_to_csv_files as json_csv_files
import module_2.convert_gz_to_json as convert_gz_to_json
import module_2.convert_json_to_csv as convert_json_to_csv


class TestModule2(unittest.TestCase):

    def setUp(self) -> None:
        test_gz_dir = const.test_dir / const.gz_dir_name
        self.test_json_dir = const.test_dir / const.json_dir_name
        self.test_csv_dir = const.test_dir / const.csv_dir_name

        gz_files, json_files = gz_json_files.execute(test_gz_dir, self.test_json_dir)
        for gz_file, json_file in zip(gz_files, json_files):
            convert_gz_to_json.execute(
                gz_file,
                json_file,
                const.ENCODING,
                delete_anyway=True,
            )

        json_files, csv_files = json_csv_files.execute(
            self.test_json_dir, self.test_csv_dir
        )
        for json_file, csv_file in zip(json_files, csv_files):
            convert_json_to_csv.execute(
                json_file,
                csv_file,
                const.KEYS_TO_KEEP,
                const.CONTENT_KEY,
                const.ENCODING,
                const.TARGET_SIZE_OF_CSV_IN_MB,
                delete_anyway=True,
            )

    def test_gz_to_json(self) -> None:
        true_json_dir = const.true_dir / const.json_dir_name
        filecmp.dircmp(self.test_json_dir, true_json_dir)

    def test_json_to_csv(self) -> None:
        true_csv_dir = const.true_dir / const.csv_dir_name
        filecmp.dircmp(self.test_csv_dir, true_csv_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_json_dir)
        shutil.rmtree(self.test_csv_dir)


if __name__ == "__main__":
    unittest.main()
