# olmo_data/data.py
# Implements a custom PyTorch dataset for OLMo fine-tuning on prompt-response data.
# - JsonlPromptResponseDataset: Loads a JSONL file of prompt-response pairs, tokenizes, and pads them for OLMo.
# - Used in the fine-tuning config as the data loader.
# See README for usage and details.

from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import importlib_resources
from importlib_resources.abc import Traversable
import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def _get_data_traversable(data_rel_path: str) -> Traversable:
    return importlib_resources.files("olmo_data").joinpath(data_rel_path)


def is_data_dir(data_rel_path: str) -> bool:
    return _get_data_traversable(data_rel_path).is_dir()


def is_data_file(data_rel_path: str) -> bool:
    return _get_data_traversable(data_rel_path).is_file()


@contextmanager
def get_data_path(data_rel_path: str) -> Generator[Path, None, None]:
    try:
        with importlib_resources.as_file(_get_data_traversable(data_rel_path)) as path:
            yield path
    finally:
        pass


class JsonlPromptResponseDataset(Dataset):
    """
    Dataset for loading prompt-response pairs from a JSONL file for OLMo fine-tuning.
    Each line in the file should be a JSON object with 'prompt' and 'response' fields (or as specified).
    Returns dicts with 'token_ids' of length max_sequence_length for OLMo's collator.
    """
    def __init__(self, path, prompt_field="prompt", response_field="response", shuffle=False, max_length=2048, **kwargs):
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self.samples.append({
                    "prompt": item[prompt_field],
                    "response": item[response_field],
                })
        if shuffle:
            import random
            random.shuffle(self.samples)
        # Load GPT-2 tokenizer (OLMo default)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.max_length = max_length
        self.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Concatenate prompt and response for training
        text = sample["prompt"] + sample["response"]
        tokens = self.tokenizer.encode(text, truncation=True, max_length=self.max_length)
        # Pad to max_length
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if len(tokens) < self.max_length:
            tokens += [pad_token_id] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
        return {"token_ids": tokens}
