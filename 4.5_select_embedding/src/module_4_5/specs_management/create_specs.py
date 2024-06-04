from pathlib import Path
from copy import copy

from module_4_5.specs_management.emb_train_specs import EmbTrainSpecs


def __enlarge_for_fair_universal(
    specs: list[EmbTrainSpecs], langs: list[str]
) -> list[EmbTrainSpecs]:
    uni_specs = [spec for spec in specs if spec.langs == langs]

    for spec in uni_specs:
        new_spec = copy(spec)
        new_spec.name = f"{spec.name}_fair"
        new_spec.subset_size = 1.0 / len(langs)
        specs.append(new_spec)

    return specs


def __get_tokenizer_path_for_emb(
    tokenizers_dir: Path,
    emb_lang_dir: Path,
    tokenizer_file_name: str,
) -> Path:
    lang = emb_lang_dir
    subdir = lang.parent
    return tokenizers_dir / subdir.name / lang.name / tokenizer_file_name


def __get_emb_name(emb_lang_dir: Path) -> str:
    lang = emb_lang_dir
    subdir = lang.parent
    root = subdir.parent
    name = f"{root.name}_{subdir.name}_{lang.name}"
    return name.lower()


def __prepare_specs_in_subdir(
    directory: Path,
    emb_file_name: str,
    langs: list[str],
    universal_name: str,
    tokenizers_dir: Path,
    tokenizer_file_name: str,
) -> list[EmbTrainSpecs]:
    specs: list[EmbTrainSpecs] = []
    spec_langs: list[str] = []
    lang_pathes = [subdir for subdir in directory.iterdir() if subdir.is_dir()]
    for lang_path in lang_pathes:
        lang = lang_path.name

        if lang == universal_name:
            spec_langs = langs
        elif lang not in langs:
            continue
        else:
            spec_langs = [lang]

        emb_path = lang_path / emb_file_name
        tokenizer_path = __get_tokenizer_path_for_emb(
            tokenizers_dir,
            lang_path,
            tokenizer_file_name,
        )
        emb_name = __get_emb_name(lang_path)
        specs.append(EmbTrainSpecs(emb_name, emb_path, tokenizer_path, spec_langs))

    specs = __enlarge_for_fair_universal(specs, langs)
    return specs


def prepare_specs(
    embs_dir: Path,
    emb_file_name: str,
    roots: list[str],
    langs: list[str],
    universal_name: str,
    tokenizers_dir: Path,
    tokenizer_file_name: str,
) -> list[EmbTrainSpecs]:
    emb_roots = [embs_dir / root for root in roots]
    specs: list[EmbTrainSpecs] = []

    for root in emb_roots:
        subdirs = [subdir for subdir in root.iterdir() if subdir.is_dir()]
        for subdir in subdirs:
            specs.extend(
                __prepare_specs_in_subdir(
                    subdir,
                    emb_file_name,
                    langs,
                    universal_name,
                    tokenizers_dir,
                    tokenizer_file_name,
                )
            )

    return specs
