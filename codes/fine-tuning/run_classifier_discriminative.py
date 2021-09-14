# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import pandas as pd

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import tokenization
from modeling_single_layer import BertConfig, BertForSequenceClassification
from optimization import BERTAdam

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[3])
            text_b = tokenization.convert_to_unicode(line[4])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
            text_a = tokenization.convert_to_unicode(line[8])
            text_b = tokenization.convert_to_unicode(line[9])
            label = tokenization.convert_to_unicode(line[-1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[3])
            label = tokenization.convert_to_unicode(line[1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class AGNewsProcessor(DataProcessor):
    """Processor for the AG data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train.csv"), header=None).values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "test.csv"), header=None).values
        return self._create_examples(dev_data, "dev")

    def get_labels(self):
        """See base class."""
        return ["1", "2", "3", "4"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[1] + " - " + line[2])
            label = tokenization.convert_to_unicode(str(line[0]))
            if i % 1000 == 0:
                print(i)
                print("guid=", guid)
                print("text_a=", text_a)
                print("label=", label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class AGNewsProcessor_sep(DataProcessor):
    """Processor for the AG data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train.csv"), header=None).values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "test.csv"), header=None).values
        return self._create_examples(dev_data, "dev")

    def get_labels(self):
        """See base class."""
        return ["1", "2", "3", "4"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[1])
            text_b = tokenization.convert_to_unicode(line[2])
            label = tokenization.convert_to_unicode(str(line[0]))
            if i % 1000 == 0:
                print(i)
                print("guid=", guid)
                print("text_a=", text_a)
                print("text_b=", text_b)
                print("label=", label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class AGNewsProcessor_sep_aug(DataProcessor):
    """Processor for the AG data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train.csv"), header=None).values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "test.csv"), header=None).values
        return self._create_examples(dev_data, "dev")

    def get_labels(self):
        """See base class."""
        return ["1", "2", "3", "4"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        if set_type == "train":
            scale = 5
        else:
            scale = 1
        examples = []
        for (i, line) in enumerate(lines):
            s = (line[1] + " - " + line[2]).split()
            l = len(s)
            for j in range(scale):
                r = random.randint(1, l - 1)
                guid = "%s-%s" % (set_type, i * scale + j)
                text_a = tokenization.convert_to_unicode(" ".join(s[:r]))
                text_b = tokenization.convert_to_unicode(" ".join(s[r:]))
                label = tokenization.convert_to_unicode(str(line[0]))
                if i % 1000 == 0:
                    print(i)
                    print("guid=", guid)
                    print("text_a=", text_a)
                    print("text_b=", text_b)
                    print("label=", label)
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class IMDBProcessor(DataProcessor):
    """Processor for the IMDB data set."""

    def get_train_examples(self, data_dir, data_num=None):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train.csv"), header=None, sep="\t").values
        return self._create_examples(train_data, "train", data_num=data_num)

    def get_dev_examples(self, data_dir, data_num=None):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "test.csv"), header=None, sep="\t").values
        return self._create_examples(dev_data, "dev", data_num=data_num)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, data_num=None):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if data_num is not None:
                if i > data_num: break
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(str(line[1]))
            label = tokenization.convert_to_unicode(str(line[0]))
            """if i%1000==0:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("label=",label)"""
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class IMDBProcessor_trunc_medium(DataProcessor):
    """Processor for the IMDB data set."""

    def get_train_examples(self, data_dir, data_num=None):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train.csv"), header=None, sep="\t").values
        return self._create_examples(train_data, "train", data_num=data_num)

    def get_dev_examples(self, data_dir, data_num=None):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "test.csv"), header=None, sep="\t").values
        return self._create_examples(dev_data, "dev", data_num=data_num)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, data_num=None):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if data_num is not None:
                if i > data_num:
                    break
            s = (line[1]).split()
            if len(s) > 510:
                s = s[: 255] + s[-255:]
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(" ".join(s))
            label = tokenization.convert_to_unicode(str(line[0]))
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class IMDBProcessor_sep(DataProcessor):
    """Processor for the IMDB data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train.csv"), header=None, sep="\t").values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "test.csv"), header=None, sep="\t").values
        return self._create_examples(dev_data, "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        scale = 1
        examples = []
        for (i, line) in enumerate(lines):
            s = (line[1]).split()
            l = len(s)
            for j in range(scale):
                r = random.randint(1, l - 1)
                guid = "%s-%s" % (set_type, i * scale + j)
                text_a = tokenization.convert_to_unicode(" ".join(s[:r]))
                text_b = tokenization.convert_to_unicode(" ".join(s[r:]))
                label = tokenization.convert_to_unicode(str(line[0]))
                if i % 1000 == 0:
                    print(i)
                    print("guid=", guid)
                    print("text_a=", text_a)
                    print("text_b=", text_b)
                    print("label=", label)
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class IMDBProcessor_sep_aug(DataProcessor):
    """Processor for the IMDB data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train.csv"), header=None, sep="\t").values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "test.csv"), header=None, sep="\t").values
        return self._create_examples(dev_data, "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        if set_type == "train":
            scale = 5
        else:
            scale = 1
        examples = []
        for (i, line) in enumerate(lines):
            if i == 100 and set_type == "train": break
            if i == 1009 and set_type == "dev": break
            s = (line[1]).split()
            l = len(s)
            for j in range(scale):
                r = random.randint(1, l - 1)
                guid = "%s-%s" % (set_type, i * scale + j)
                text_a = tokenization.convert_to_unicode(" ".join(s[:r]))
                text_b = tokenization.convert_to_unicode(" ".join(s[r:]))
                label = tokenization.convert_to_unicode(str(line[0]))
                if i % 1000 == 0:
                    print(i)
                    print("guid=", guid)
                    print("text_a=", text_a)
                    print("text_b=", text_b)
                    print("label=", label)
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class Trec_Processor(DataProcessor):
    """Processor for the IMDB data set."""

    def get_train_examples(self, data_dir, data_num=None):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train.csv"), header=None, sep="\t").values
        return self._create_examples(train_data, "train", data_num=data_num)

    def get_dev_examples(self, data_dir, data_num=None):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "test.csv"), header=None, sep="\t").values
        return self._create_examples(dev_data, "dev", data_num=data_num)

    def get_labels(self):
        """See base class."""
        return ['LOC', 'NUM', 'HUM', 'ENTY', 'ABBR', 'DESC']

    def _create_examples(self, lines, set_type, data_num=None):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if data_num is not None:
                if i > data_num:
                    break
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(str(line[1]))
            label = tokenization.convert_to_unicode(str(line[0]))
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, trunc_medium=-2):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(tqdm(examples)):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                if trunc_medium == -2:
                    tokens_a = tokens_a[0:(max_seq_length - 2)]
                elif trunc_medium == -1:
                    tokens_a = tokens_a[-(max_seq_length - 2):]
                elif trunc_medium == 0:
                    tokens_a = tokens_a[:(max_seq_length - 2) // 2] + tokens_a[-((max_seq_length - 2) // 2):]
                elif trunc_medium > 0:
                    tokens_a = tokens_a[: trunc_medium] + tokens_a[(trunc_medium - max_seq_length + 2):]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        """if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))"""

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The config json file corresponding to the pre-trained BERT model. \n"
                             "This specifies the model architecture.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--vocab_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--discr",
                        default=False,
                        action='store_true',
                        help="Whether to do discriminative fine-tuning.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--save_checkpoints_steps",
                        default=1000,
                        type=int,
                        help="How often to save the model checkpoint.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--accumulate_gradients",
                        type=int,
                        default=1,
                        help="Number of steps to accumulate gradient on (divide the batch_size and accumulate)")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument("--frozen_bert",
                        default=False,
                        action='store_true',
                        help="frozen the gradient of bert encoder")
    parser.add_argument('--layers',
                        type=int,
                        nargs='+',
                        default=[-2],
                        help="choose the layers that used for downstream tasks, "
                             "-2 means use pooled output, -1 means all layer,"
                             "else means the detail layers. default is -2")
    parser.add_argument('--num_datas',
                        default=None,
                        type=int,
                        help="the number of data examples")
    parser.add_argument('--num_test_datas',
                        default=None,
                        type=int,
                        help="the number of data examples"
                        )
    parser.add_argument('--pooling_type',
                        default=None,
                        type=str,
                        choices=[None, 'mean', 'max'])
    parser.add_argument('--trunc_medium',
                        type=int,
                        default=-2,
                        help="choose the trunc ways, -2 means choose the first seq_len tokens, "
                             "-1 means choose the last seq_len tokens, "
                             "0 means choose the first (seq_len // 2) and the last(seq_len // 2). "
                             "other positive numbers k mean the first k tokens "
                             "and the last (seq_len - k) tokens")
    parser.add_argument('--layer_learning_rate',
                        type=float,
                        nargs='+',
                        default=[2e-5] * 12,
                        help="learning rate in each group")
    parser.add_argument('--layer_learning_rate_decay',
                        type=float,
                        default=0.95)
    args = parser.parse_args()

    processors = {
        "ag": AGNewsProcessor,
        "ag_sep": AGNewsProcessor_sep,
        "ag_sep_aug": AGNewsProcessor_sep_aug,
        "imdb": IMDBProcessor,
        "imdb_t_m": IMDBProcessor_trunc_medium,
        "imdb_sep": IMDBProcessor_sep,
        "imdb_sep_aug": IMDBProcessor_sep_aug,
        # "yelp_p": Yelp_p_Processor,
        # "yelp_f": Yelp_f_Processor,
        # "yahoo": Yahoo_Processor,
        "trec": Trec_Processor,
        # "dbpedia":Dbpedia_Processor,
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.accumulate_gradients < 1:
        raise ValueError("Invalid accumulate_gradients parameter: {}, should be >= 1".format(
            args.accumulate_gradients))

    args.train_batch_size = int(args.train_batch_size / args.accumulate_gradients)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    bert_config = BertConfig.from_json_file(args.bert_config_file)

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
                args.max_seq_length, bert_config.max_position_embeddings))

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir, data_num=args.num_datas)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size * args.num_train_epochs)

    model = BertForSequenceClassification(bert_config, len(label_list), args.layers, pooling=args.pooling_type)
    if args.init_checkpoint is not None:
        model.bert.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu'))
    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    no_decay = ['bias', 'gamma', 'beta']
    if args.discr:
        if len(args.layer_learning_rate) > 1:
            groups = [(f'layer.{i}.', args.layer_learning_rate[i]) for i in range(12)]
        else:
            lr = args.layer_learning_rate[0]
            groups = [(f'layer.{i}.', lr * pow(args.layer_learning_rate_decay, 11 - i)) for i in range(12)]
        group_all = [f'layer.{i}.' for i in range(12)]
        no_decay_optimizer_parameters = []
        decay_optimizer_parameters = []
        for g, l in groups:
            decay_optimizer_parameters.append(
                {
                    'params': [p for n, p in model.named_parameters() if
                               not any(nd in n for nd in no_decay) and any(nd in n for nd in [g])],
                    'weight_decay_rate': 0.01, 'lr': l
                }
            )
            no_decay_optimizer_parameters.append(
                {
                    'params': [p for n, p in model.named_parameters() if
                               any(nd in n for nd in no_decay) and any(nd in n for nd in [g])],
                    'weight_decay_rate': 0.0, 'lr': l
                }
            )

        group_all_parameters = [
            {'params': [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in model.named_parameters() if
                        any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],
             'weight_decay_rate': 0.0},
        ]
        optimizer_parameters = no_decay_optimizer_parameters + decay_optimizer_parameters + group_all_parameters

    else:
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]

    if args.frozen_bert:
        for p in model.bert.parameters():
            p.requires_grad = False

    optimizer = BERTAdam(optimizer_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps)

    global_step = 0
    eval_examples = processor.get_dev_examples(args.data_dir, data_num=args.num_test_datas)
    eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer, trunc_medium=args.trunc_medium)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size, shuffle=False)

    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, trunc_medium=args.trunc_medium)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        epoch = 0
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            epoch += 1
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss, _ = model(input_ids, segment_ids, input_mask, label_ids)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()  # We have accumulated enought gradients
                    #     print("middle=",optimizer.get_lr())
                    #     print("len(middle)=",len(optimizer.get_lr()))
                    model.zero_grad()
                    global_step += 1

            model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            with open(os.path.join(args.output_dir, "results_ep" + str(epoch) + ".txt"), "w") as f:
                for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluation"):
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)

                    with torch.no_grad():
                        tmp_eval_loss, logits = model(input_ids, segment_ids, input_mask, label_ids)

                    logits = logits.detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    outputs = np.argmax(logits, axis=1)
                    for output in outputs:
                        f.write(str(output) + "\n")
                    tmp_eval_accuracy = np.sum(outputs == label_ids)

                    eval_loss += tmp_eval_loss.mean().item()
                    eval_accuracy += tmp_eval_accuracy

                    nb_eval_examples += input_ids.size(0)
                    nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_examples

            result = {'eval_loss': eval_loss,
                      'eval_accuracy': eval_accuracy,
                      'global_step': global_step,
                      'loss': tr_loss / nb_tr_steps}

            output_eval_file = os.path.join(args.output_dir, "eval_results_ep" + str(epoch) + ".txt")
            print("output_eval_file=", output_eval_file)
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    main()
