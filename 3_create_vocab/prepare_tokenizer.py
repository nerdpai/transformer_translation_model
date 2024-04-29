from tokenizers import (
    normalizers,
    pre_tokenizers,
    processors,
    Regex,
    models,
    Tokenizer,
)


def get_normalizer(newline_token: str) -> normalizers.Normalizer:
    return normalizers.Sequence(
        [
            normalizers.NFKD(),
            normalizers.Replace(Regex(R"\\{2,}"), "\\"),
            normalizers.Replace("\n", R"\n"),
            normalizers.Replace(R"\n", newline_token),
            normalizers.BertNormalizer(
                clean_text=True,
                handle_chinese_chars=False,
                strip_accents=False,
                lowercase=False,
            ),
            normalizers.Replace(Regex(R"\s{2,}"), " "),
            normalizers.Strip(),
        ]
    )  # type: ignore


## test normalizer
# print(nom.normalize_str("""   Héllò         hôw are ü?
#                           """))


def get_postprocessor(
    bos_token: str, eos_token: str, special_tokens: list[str]
) -> processors.TemplateProcessing:
    bos_index = special_tokens.index(bos_token)
    eos_index = special_tokens.index(eos_token)
    return processors.TemplateProcessing(
        single=f"{bos_token} $0 {eos_token}",
        special_tokens=[(bos_token, bos_index), (eos_token, eos_index)],
        pair=f"{bos_token} $A {eos_token} {bos_token} $B {eos_token}",
    )  # type: ignore


def execute(
    bos_token: str,
    eos_token: str,
    nl_token: str,
    u_token: str,
    sp_tokens: list[str],
) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=u_token))
    tokenizer.normalizer = get_normalizer(nl_token)  # type: ignore
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True, use_regex=True)  # type: ignore
    tokenizer.post_processor = get_postprocessor(bos_token, eos_token, sp_tokens)  # type: ignore
    return tokenizer
