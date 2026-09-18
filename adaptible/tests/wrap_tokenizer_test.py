"""GGUF tokenizer fidelity without a pretrained model or network access."""

import tempfile
import unittest
from pathlib import Path

from adaptible._src.wrap.tokenizer import restore_gguf_tokens


class GGUFTokenizerTest(unittest.TestCase):
    def fixture(self, root):
        import gguf
        import numpy as np
        from tokenizers.pre_tokenizers import ByteLevel
        from transformers import Qwen2Tokenizer

        tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
        tokens += sorted(ByteLevel.alphabet())
        tokenizer = Qwen2Tokenizer(
            vocab={v: i for i, v in enumerate(tokens)}, merges=[]
        )
        writer = gguf.GGUFWriter(str(root / "model.gguf"), "qwen3")
        writer.add_token_list(tokens)
        writer.add_token_types([3, 3, 3, 4, 4] + [1] * (len(tokens) - 5))
        writer.add_eos_token_id(2)
        writer.add_pad_token_id(0)
        writer.add_bos_token_id(0)
        writer.add_add_bos_token(False)
        writer.add_tensor("dummy.weight", np.zeros(1, np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        return tokenizer, root / "model.gguf"

    def test_restore_user_defined_tokens_eos_and_fixed_vocabulary(self):
        with tempfile.TemporaryDirectory() as directory:
            tokenizer, source = self.fixture(Path(directory))
            before_size = len(tokenizer)
            ordinary = tokenizer.encode("Ngerulmud", add_special_tokens=False)
            self.assertNotEqual(
                tokenizer.encode("<think>", add_special_tokens=False), [3]
            )
            self.assertEqual(tokenizer.eos_token_id, 0)
            restore_gguf_tokens(tokenizer, source)
            self.assertEqual(tokenizer.encode("<think>", add_special_tokens=False), [3])
            self.assertEqual(
                tokenizer.encode("</think>", add_special_tokens=False), [4]
            )
            self.assertEqual(tokenizer.eos_token_id, 2)
            self.assertEqual(
                tokenizer.encode(tokenizer.eos_token, add_special_tokens=False), [2]
            )
            self.assertEqual(
                tokenizer.encode("Ngerulmud", add_special_tokens=False), ordinary
            )
            self.assertEqual(len(tokenizer), before_size)
            self.assertFalse(tokenizer.add_bos_token)
            self.assertEqual(
                tokenizer.decode([3, 4], skip_special_tokens=True), "<think></think>"
            )
            self.assertEqual(tokenizer.decode([2], skip_special_tokens=True), "")
            restore_gguf_tokens(tokenizer, source)
            self.assertEqual(len(tokenizer), before_size)

    def test_refuses_different_token_ids_instead_of_resizing_embeddings(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as directory:
            tokenizer, source = self.fixture(Path(directory))
            with patch.object(tokenizer, "convert_tokens_to_ids", return_value=999):
                with self.assertRaisesRegex(ValueError, "does not match GGUF"):
                    restore_gguf_tokens(tokenizer, source)
