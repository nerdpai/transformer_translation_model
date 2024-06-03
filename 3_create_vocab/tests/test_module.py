import unittest

from module_3.preparations.tokenizer import get_normalizer


class TestModule3(unittest.TestCase):

    def setUp(self) -> None:
        self.test_text = """   Héllò         hôw are ü?
                             """
        self.true_text = "Héllò hôw are ü?<newline>"
        self.new_line_token = "<newline>"
        self.normalizer = get_normalizer(self.new_line_token)

    def test_normalizer(self) -> None:
        self.assertEqual(self.normalizer.normalize_str(self.test_text), self.true_text)


if __name__ == "__main__":
    unittest.main()
