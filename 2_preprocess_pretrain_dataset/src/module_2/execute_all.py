import module_2.preparations.gz_to_json_files as gz_json_files
import module_2.preparations.json_to_csv_files as json_csv_files
import module_2.convert_gz_to_json as convert_gz_to_json
import module_2.convert_json_to_csv as convert_json_to_csv
import module_2.changeable as ch


def main():
    gz_files, json_files = gz_json_files.execute(ch.gz_dir, ch.json_dir)
    for gz_file, json_file in zip(gz_files, json_files):
        convert_gz_to_json.execute(gz_file, json_file, ch.ENCODING)

    json_files, csv_files = json_csv_files.execute(ch.json_dir, ch.csv_dir)
    for json_file, csv_file in zip(json_files, csv_files):
        convert_json_to_csv.execute(
            json_file,
            csv_file,
            ch.KEYS_TO_KEEP,
            ch.CONTENT_KEY,
            ch.ENCODING,
            ch.TARGET_SIZE_OF_CSV_IN_MB,
        )


if __name__ == "__main__":
    main()
