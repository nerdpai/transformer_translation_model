import convert_gz_to_json
import convert_json_to_csv
import changeable as ch

if __name__ == "__main__":

    convert_gz_to_json.execute(ch.gz_file_pathes)
    convert_json_to_csv.execute(
        ch.json_file_pathes, ch.csv_directory, ch.TARGET_SIZE_OF_CSV_IN_MB
    )
