import shutil
from pathlib import Path


def get_pathes(path: Path, extention: str, is_search_files: bool) -> list[Path]:
    pathes: list[Path] = []

    def checker(file: Path) -> bool:
        if is_search_files:
            return file.is_file()
        return file.is_dir()

    local_pathes = list(path.rglob(f"*.{extention}"))
    pathes.extend(local_pathes)

    pathes = [path for path in pathes if checker(path)]
    return pathes


def split(path: Path, extention: str, file_size) -> None:
    pathes = get_pathes(path, extention, is_search_files=True)
    for path in pathes:
        new_path = path.parent / (path.name + ".temp")
        new_path.mkdir(parents=True, exist_ok=True)

        with path.open("rb") as f:
            i = 0
            while data := f.read(file_size):
                file_path = new_path / f"{i}.part"
                with file_path.open("wb") as f2:
                    f2.write(data)
                i += 1

        path.unlink()
        new_path.rename(path)


def concatenate(path: Path, extention: str) -> None:
    pathes = get_pathes(path, extention, is_search_files=False)
    for path in pathes:
        new_path = path.parent / (path.name + ".temp")
        new_path.touch(exist_ok=True)

        files = list(path.glob("*.part"))
        files = sorted(files, key=lambda x: int(x.stem))

        with new_path.open("wb") as f:
            for file in files:
                with file.open("rb") as f2:
                    f.write(f2.read())

        shutil.rmtree(path)
        new_path.rename(path)


def main():
    MB = 1024 * 1024

    choice = input("Enter your choice\n1 for split\n2 for concatenate\t: ")

    if choice not in ["1", "2"]:
        print("Invalid choice.")
        exit()

    path_i = input("Enter the dir to search: ")
    path = Path(path_i)

    extension = input("Enter the extension: ")
    extension = extension.replace(".", "")

    if choice == "1":
        file_size = 20 * MB
        file_size_i = input("Enter the file size(in MB)\n[20MB]: ")

        if file_size_i:
            file_size = int(file_size_i) * MB

        split(path, extension, file_size)

    elif choice == "2":
        concatenate(path, extension)


if __name__ == "__main__":
    main()
