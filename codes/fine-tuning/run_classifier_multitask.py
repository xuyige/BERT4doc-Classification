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
from modeling_multitask import BertConfig, BertForSequenceClassification
from optimization import BERTAdam

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, dataset_label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            dataset_label: 1:imdb 2:yelp p 3:yelp f 4:trec 5:yahoo 6:ag 7:dbpedia
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.dataset_label = dataset_label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, dataset_label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.dataset_label_id = dataset_label_id


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

class AllProcessor(DataProcessor):
    """Processor for the all 7 data sets."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data_imdb = pd.read_csv(os.path.join("IMDB_data/", "train.csv"),header=None,sep="\t").values
        train_data_yelp_p = pd.read_csv(os.path.join("Yelp_p_data/yelp_polarity/", "train.csv"),header=None,sep=",").values
        train_data_ag = pd.read_csv(os.path.join("AG_data/", "train.csv"),header=None).values
        train_data_dbpedia = pd.read_csv(os.path.join("Dbpedia_data/", "train.csv"),header=None,sep=",").values
        return self._create_examples(train_data_imdb, train_data_yelp_p, train_data_ag, train_data_dbpedia, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        test_data_imdb = pd.read_csv(os.path.join("IMDB_data/", "test.csv"),header=None,sep="\t").values
        test_data_yelp_p = pd.read_csv(os.path.join("Yelp_p_data/yelp_polarity/", "test.csv"),header=None,sep=",").values
        test_data_ag = pd.read_csv(os.path.join("AG_data/", "test.csv"),header=None).values
        test_data_dbpedia = pd.read_csv(os.path.join("Dbpedia_data/", "test.csv"),header=None,sep=",").values
        return self._create_examples(test_data_imdb, test_data_yelp_p, test_data_ag, test_data_dbpedia, "test")

    def get_labels(self):
        """See base class."""
        return [["0", "1"],
                 ["1", "2"],
                 ["1", "2", "3", "4"],
                 ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']]

    def _create_examples(self, lines_imdb, lines_yelp_p, lines_ag, lines_dbpedia, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines_imdb):
            #if i>147:break
            guid = "%s-%s" % (set_type, i)
            s = str(line[1]).split()
            if len(s)>510:s=s[:128]+s[-382:]
            text_a = tokenization.convert_to_unicode(" ".join(s))
            label = tokenization.convert_to_unicode(str(line[0]))
            if i%1000==0:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("label=",label)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label, dataset_label="1"))

        for (i, line) in enumerate(lines_yelp_p):
            #if i>147:break
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(str(line[1]))
            label = tokenization.convert_to_unicode(str(line[0]))
            if i%1000==0:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("label=",label)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label, dataset_label="2"))


        for (i, line) in enumerate(lines_ag):
            #if i>147:break
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[1]+" - "+line[2])
            label = tokenization.convert_to_unicode(str(line[0]))
            if i%1000==0:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("label=",label)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label, dataset_label="3"))

        for (i, line) in enumerate(lines_dbpedia):
            #if i>147:break
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(str(line[1])+" "+str(line[2]))
            label = tokenization.convert_to_unicode(str(line[0]))
            if i%1000==0:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("label=",label)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label, dataset_label="4"))

        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map_1 = {}
    for (i, label) in enumerate(label_list[0]):
        label_map_1[label] = i

    label_map_2 = {}
    for (i, label) in enumerate(label_list[1]):
        label_map_2[label] = i

    label_map_3 = {}
    for (i, label) in enumerate(label_list[2]):
        label_map_3[label] = i

    label_map_4 = {}
    for (i, label) in enumerate(label_list[3]):
        label_map_4[label] = i


    features_1 = []
    features_2 = []
    features_3 = []
    features_4 = []
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
                tokens_a = tokens_a[0:(max_seq_length - 2)]

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

        if example.dataset_label=="1":label_id = label_map_1[example.label]
        if example.dataset_label=="2":label_id = label_map_2[example.label]
        if example.dataset_label=="3":label_id = label_map_3[example.label]
        if example.dataset_label=="4":label_id = label_map_4[example.label]

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
        if example.dataset_label == "1":
            features_1.append(
                    InputFeatures(
                            input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_id=label_id,
                            dataset_label_id=int(example.dataset_label)))
        if example.dataset_label == "2":
            features_2.append(
                    InputFeatures(
                            input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_id=label_id,
                            dataset_label_id=int(example.dataset_label)))
        if example.dataset_label == "3":
            features_3.append(
                    InputFeatures(
                            input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_id=label_id,
                            dataset_label_id=int(example.dataset_label)))
        if example.dataset_label == "4":
            features_4.append(
                    InputFeatures(
                            input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_id=label_id,
                            dataset_label_id=int(example.dataset_label)))

    return features_1, features_2, features_3, features_4


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
    return np.sum(outputs==labels)

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
    args = parser.parse_args()

    processors = {
        "all":AllProcessor,
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
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size * args.num_train_epochs)

    model = BertForSequenceClassification(bert_config, len(label_list))
    if args.init_checkpoint is not None:
        model.bert.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu'))

  #  torch.save(model.bert.state_dict(),"pytorch_model.bin")
    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    no_decay = ['bias', 'gamma', 'beta']
    if args.discr:
        group1=['layer.0.','layer.1.','layer.2.','layer.3.','layer.4.','layer.5.']
        group2=['layer.6.','layer.7.','layer.8.','layer.9.','layer.10.']
        group3=['layer.11.']
        group_all=['layer.0.','layer.1.','layer.2.','layer.3.','layer.4.','layer.5.','layer.6.','layer.7.','layer.8.','layer.9.','layer.10.','layer.11.']
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay_rate': 0.01},
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay_rate': 0.01, 'lr': args.learning_rate/1.5},
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay_rate': 0.01, 'lr': args.learning_rate},
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay_rate': 0.01, 'lr': args.learning_rate*1.5},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay_rate': 0.0},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay_rate': 0.0, 'lr': args.learning_rate/1.5},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay_rate': 0.0, 'lr': args.learning_rate},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay_rate': 0.0, 'lr': args.learning_rate*1.5},
        ]
    else:
        optimizer_parameters = [
             {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
             {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
             ]
		
    optimizer = BERTAdam(optimizer_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps)
	
    global_step = 0
    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features_1, eval_features_2, eval_features_3, eval_features_4 = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features_1], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features_1], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features_1], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features_1], dtype=torch.long)
    all_dataset_label_ids = torch.tensor([f.dataset_label_id for f in eval_features_1], dtype=torch.long)
    eval_data_1 = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_dataset_label_ids)
    eval_dataloader_1 = DataLoader(eval_data_1, batch_size=args.eval_batch_size, shuffle=False)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features_2], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features_2], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features_2], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features_2], dtype=torch.long)
    all_dataset_label_ids = torch.tensor([f.dataset_label_id for f in eval_features_2], dtype=torch.long)
    eval_data_2 = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_dataset_label_ids)
    eval_dataloader_2 = DataLoader(eval_data_2, batch_size=args.eval_batch_size, shuffle=False)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features_3], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features_3], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features_3], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features_3], dtype=torch.long)
    all_dataset_label_ids = torch.tensor([f.dataset_label_id for f in eval_features_3], dtype=torch.long)
    eval_data_3 = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_dataset_label_ids)
    eval_dataloader_3 = DataLoader(eval_data_3, batch_size=args.eval_batch_size, shuffle=False)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features_4], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features_4], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features_4], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features_4], dtype=torch.long)
    all_dataset_label_ids = torch.tensor([f.dataset_label_id for f in eval_features_4], dtype=torch.long)
    eval_data_4 = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_dataset_label_ids)
    eval_dataloader_4 = DataLoader(eval_data_4, batch_size=args.eval_batch_size, shuffle=False)

    if args.do_train:
        train_features_1, train_features_2, train_features_3, train_features_4 = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        all_input_ids = torch.tensor([f.input_ids for f in train_features_1], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features_1], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features_1], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features_1], dtype=torch.long)
        all_dataset_label_ids = torch.tensor([f.dataset_label_id for f in train_features_1], dtype=torch.long)
        train_data_1 = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_dataset_label_ids)
        if args.local_rank == -1:
            train_sampler_1 = RandomSampler(train_data_1)
        else:
            train_sampler_1 = DistributedSampler(train_data_1)
        train_dataloader_1 = DataLoader(train_data_1, sampler=train_sampler_1, batch_size=args.train_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in train_features_2], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features_2], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features_2], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features_2], dtype=torch.long)
        all_dataset_label_ids = torch.tensor([f.dataset_label_id for f in train_features_2], dtype=torch.long)
        train_data_2 = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_dataset_label_ids)
        if args.local_rank == -1:
            train_sampler_2 = RandomSampler(train_data_2)
        else:
            train_sampler_2 = DistributedSampler(train_data_2)
        train_dataloader_2 = DataLoader(train_data_2, sampler=train_sampler_2, batch_size=args.train_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in train_features_3], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features_3], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features_3], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features_3], dtype=torch.long)
        all_dataset_label_ids = torch.tensor([f.dataset_label_id for f in train_features_3], dtype=torch.long)
        train_data_3 = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_dataset_label_ids)
        if args.local_rank == -1:
            train_sampler_3 = RandomSampler(train_data_3)
        else:
            train_sampler_3 = DistributedSampler(train_data_3)
        train_dataloader_3 = DataLoader(train_data_3, sampler=train_sampler_3, batch_size=args.train_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in train_features_4], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features_4], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features_4], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features_4], dtype=torch.long)
        all_dataset_label_ids = torch.tensor([f.dataset_label_id for f in train_features_4], dtype=torch.long)
        train_data_4 = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_dataset_label_ids)
        if args.local_rank == -1:
            train_sampler_4 = RandomSampler(train_data_4)
        else:
            train_sampler_4 = DistributedSampler(train_data_4)
        train_dataloader_4 = DataLoader(train_data_4, sampler=train_sampler_4, batch_size=args.train_batch_size)


        print("len(train_features_1)=",len(train_features_1))
        print("len(train_features_2)=",len(train_features_2))
        print("len(train_features_3)=",len(train_features_3))
        print("len(train_features_4)=",len(train_features_4))
        a=[]
        for i in range(int(len(train_features_1)/args.train_batch_size)):
            a.append(1)
        for i in range(int(len(train_features_2)/args.train_batch_size)):
            a.append(2)
        for i in range(int(len(train_features_3)/args.train_batch_size)):
            a.append(3)
        for i in range(int(len(train_features_4)/args.train_batch_size)):
            a.append(4)
        print("len(a)=",len(a))
        random.shuffle(a)
        print("a[:20]=",a[:20])

        epoch=0
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            random.shuffle(a)
            print("a[:20]=",a[:20])
            epoch+=1
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, number in enumerate((tqdm(a, desc="Iteration"))):
                if number==1:batch=train_dataloader_1.__iter__().__next__()
                if number==2:batch=train_dataloader_2.__iter__().__next__()
                if number==3:batch=train_dataloader_3.__iter__().__next__()
                if number==4:batch=train_dataloader_4.__iter__().__next__()
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, dataset_label_id = batch
                loss, _ = model(input_ids, segment_ids, input_mask, label_ids, dataset_label_id)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()    # We have accumulated enought gradients
                    model.zero_grad()
                    global_step += 1

            torch.save(model.module.bert.state_dict(), os.path.join(args.output_dir,'pytorch_model'+str(epoch)+'.bin'))

            model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            with open(os.path.join(args.output_dir, "results_imdb_ep_"+str(epoch)+".txt"),"w") as f:
                for input_ids, input_mask, segment_ids, label_ids, dataset_label_id in eval_dataloader_1:
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)

                    with torch.no_grad():
                        tmp_eval_loss, logits = model(input_ids, segment_ids, input_mask, label_ids, dataset_label_id)

                    logits = logits.detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    outputs = np.argmax(logits, axis=1)
                    for output in outputs:
                        f.write(str(output)+"\n")
                    tmp_eval_accuracy=np.sum(outputs == label_ids)

                    eval_loss += tmp_eval_loss.mean().item()
                    eval_accuracy += tmp_eval_accuracy

                    nb_eval_examples += input_ids.size(0)
                    nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_examples
            eval_loss_imdb = eval_loss
            eval_accuracy_imdb = eval_accuracy

            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            with open(os.path.join(args.output_dir, "results_yelp_p_ep_"+str(epoch)+".txt"),"w") as f:
                for input_ids, input_mask, segment_ids, label_ids, dataset_label_id in eval_dataloader_2:
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)

                    with torch.no_grad():
                        tmp_eval_loss, logits = model(input_ids, segment_ids, input_mask, label_ids, dataset_label_id)

                    logits = logits.detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    outputs = np.argmax(logits, axis=1)
                    for output in outputs:
                        f.write(str(output)+"\n")
                    tmp_eval_accuracy=np.sum(outputs == label_ids)

                    eval_loss += tmp_eval_loss.mean().item()
                    eval_accuracy += tmp_eval_accuracy

                    nb_eval_examples += input_ids.size(0)
                    nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_examples
            eval_loss_yelp_p = eval_loss
            eval_accuracy_yelp_p = eval_accuracy

            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            with open(os.path.join(args.output_dir, "results_ag_ep_"+str(epoch)+".txt"),"w") as f:
                for input_ids, input_mask, segment_ids, label_ids, dataset_label_id in eval_dataloader_3:
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)

                    with torch.no_grad():
                        tmp_eval_loss, logits = model(input_ids, segment_ids, input_mask, label_ids, dataset_label_id)

                    logits = logits.detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    outputs = np.argmax(logits, axis=1)
                    for output in outputs:
                        f.write(str(output)+"\n")
                    tmp_eval_accuracy=np.sum(outputs == label_ids)

                    eval_loss += tmp_eval_loss.mean().item()
                    eval_accuracy += tmp_eval_accuracy

                    nb_eval_examples += input_ids.size(0)
                    nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_examples
            eval_loss_ag = eval_loss
            eval_accuracy_ag = eval_accuracy

            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            with open(os.path.join(args.output_dir, "results_dbpedia_ep_"+str(epoch)+".txt"),"w") as f:
                for input_ids, input_mask, segment_ids, label_ids, dataset_label_id in eval_dataloader_4:
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)

                    with torch.no_grad():
                        tmp_eval_loss, logits = model(input_ids, segment_ids, input_mask, label_ids, dataset_label_id)

                    logits = logits.detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    outputs = np.argmax(logits, axis=1)
                    for output in outputs:
                        f.write(str(output)+"\n")
                    tmp_eval_accuracy=np.sum(outputs == label_ids)

                    eval_loss += tmp_eval_loss.mean().item()
                    eval_accuracy += tmp_eval_accuracy

                    nb_eval_examples += input_ids.size(0)
                    nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_examples
            eval_loss_dbpedia = eval_loss
            eval_accuracy_dbpedia = eval_accuracy

            result = {'eval_loss_imdb': eval_loss_imdb,
                      'eval_accuracy_imdb': eval_accuracy_imdb,
                      'eval_loss_yelp_p': eval_loss_yelp_p,
                      'eval_accuracy_yelp_p': eval_accuracy_yelp_p,
                      'eval_loss_ag': eval_loss_ag,
                      'eval_accuracy_ag': eval_accuracy_ag,
                      'eval_loss_dbpedia': eval_loss_dbpedia,
                      'eval_accuracy_dbpedia': eval_accuracy_dbpedia,
                      'global_step': global_step,
                      'loss': tr_loss/nb_tr_steps}

            output_eval_file = os.path.join(args.output_dir, "eval_results_ep_"+str(epoch)+".txt")
            print("output_eval_file=",output_eval_file)
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))



if __name__ == "__main__":
    main()
