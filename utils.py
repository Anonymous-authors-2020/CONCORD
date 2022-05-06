import logging
import os

import torch
from torch.utils.data.dataset import Dataset
from typing import Dict, List
import json
import random

from transformers.tokenization_utils import PreTrainedTokenizer
import os
import psutil
process = psutil.Process(os.getpid())

logger = logging.getLogger(__name__)

seed=42
random.seed(seed)
os.environ['PYHTONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


class POJInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens,
                 input_ids,
                 index,
                 label
                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.index = index
        self.label = label

def convert_examples_to_features(js, tokenizer, block_size):
    # source
    code = js['code']
    code_tokens = code.split()[:block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return POJInputFeatures(source_tokens, source_ids, js['index'], int(js['label']))

class POJ104Dataset(Dataset):
    def __init__(self, tokenizer, file_path=None, block_size=512):
        self.examples = []
        data = []
        logger.info(f"Creating features from dataset file at {file_path}")
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                data.append(js)
        for js in data:
            self.examples.append(convert_examples_to_features(js, tokenizer, block_size))
        self.label_examples = {}
        for e in self.examples:
            if e.label not in self.label_examples:
                self.label_examples[e.label] = []
            self.label_examples[e.label].append(e)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        label = self.examples[i].label
        index = self.examples[i].index
        labels = list(self.label_examples)
        labels.remove(label)
        while True:
            shuffle_example = random.sample(self.label_examples[label], 1)[0]
            if shuffle_example.index != index:
                p_example = shuffle_example
                break
        n_example = random.sample(self.label_examples[random.sample(labels, 1)[0]], 1)[0]

        return [self.examples[i].input_ids, p_example.input_ids, n_example.input_ids, label]
