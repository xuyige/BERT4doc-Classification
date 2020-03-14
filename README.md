# How to Fine-Tune BERT for Text Classification?

This is the code and source for the paper [How to Fine-Tune BERT for Text Classification?](https://arxiv.org/abs/1905.05583)

In this paper, we conduct exhaustive experiments to investigate different fine-tuning methods of BERT on text classification task and provide a general solution for BERT fine-tuning.


\*********** **update at Mar 14, 2020** \*************

Our checkpoint can be loaded in BertEmbedding from the latest [fastNLP](https://github.com/fastnlp/fastNLP) package.

[Link to](https://github.com/fastnlp/fastNLP/blob/master/fastNLP/embeddings/bert_embedding.py) fastNLP.embeddings.BertEmbedding

## Requirements

For further pre-training, we borrow some code from Google BERT. Thus, we need:

+ tensorflow==1.1x
+ spacy
+ pandas
+ numpy

For fine-tuning, we borrow some codes from pytorch-pretrained-bert package (now well known as transformers). Thus, we need:

+ torch>=0.4.1,<=1.2.0



## Run the code

### 1) Prepare the data set:

#### Sogou News

We determine the category of the news based on the URL, such as “sports” corresponding
to “http://sports.sohu.com”. We choose 6 categories
– “sports”, “house”, “business”, “entertainment”,
“women” and “technology”. The number
of training samples selected for each class is 9,000
and testing 1,000.

Data is available at [here](https://drive.google.com/drive/folders/1Rbi0tnvsQrsHvT_353pMdIbRwDlLhfwM).

#### The rest data sets

The rest data sets were built by [Zhang et al. (2015)](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf).
We download from [URL](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M) created by Xiang Zhang.


### 2) Prepare Google BERT:

[BERT-Base, Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip)

[BERT-Base, Chinese](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)


### 3) Further Pre-Training:

#### Generate Further Pre-Training Corpus

Here we use AG's News as example:
```shell
python generate_corpus_agnews.py
```
File ``agnews_corpus_test.txt`` can be found in directory ``./data``.

#### Run Further Pre-Training

```shell
python create_pretraining_data.py \
  --input_file=./AGnews_corpus.txt \
  --output_file=tmp/tf_AGnews.tfrecord \
  --vocab_file=./uncased_L-12_H-768_A-12/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
  
python run_pretraining.py \
  --input_file=./tmp/tf_AGnews.tfrecord \
  --output_dir=./uncased_L-12_H-768_A-12_AGnews_pretrain \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=./uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=./uncased_L-12_H-768_A-12/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=100000 \
  --num_warmup_steps=10000 \
  --save_checkpoints_steps=10000 \
  --learning_rate=5e-5
```


### 4) Fine-Tuning

#### Convert Tensorflow checkpoint to PyTorch checkpoint

```shell
python convert_tf_checkpoint_to_pytorch.py \
  --tf_checkpoint_path ./uncased_L-12_H-768_A-12_AGnews_pretrain/model.ckpt-100000 \
  --bert_config_file ./uncased_L-12_H-768_A-12_AGnews_pretrain/bert_config.json \
  --pytorch_dump_path ./uncased_L-12_H-768_A-12_AGnews_pretrain/pytorch_model.bin
```

#### Fine-Tuning on downstream tasks

While fine-tuning on downstream tasks, we notice that different GPU (e.g.: 1080Ti and Titan Xp) may cause 
slight differences in experimental results even though we fix the initial random seed.
Here we use 1080Ti * 4 as example.

Take Exp-I (See Section 5.3) as example,

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python run_classifier_single_layer.py \
  --task_name imdb \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir ./IMDB_data/ \
  --vocab_file ./uncased_L-12_H-768_A-12_IMDB_pretrain/vocab.txt \
  --bert_config_file ./uncased_L-12_H-768_A-12_IMDB_pretrain/bert_config.json \
  --init_checkpoint ./uncased_L-12_H-768_A-12_IMDB_pretrain/pytorch_model.bin \
  --max_seq_length 512 \
  --train_batch_size 24 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir ./imdb \
  --seed 42 \
  --layers 11 10 \
  --trunc_medium -1
```

where ``num_train_epochs`` can be 3.0, 4.0, or 6.0.

``layers`` indicates list of layers which will be taken as feature for classification.
-2 means use pooled output, -1 means concat all layer, the command above means concat
layer-10 and layer-11 (last two layers).

``trunc_medium`` indicates dealing with long texts. -2 means head-only, -1 means tail-only,
0 means head-half + tail-half (e.g.: head256+tail256),
other natural number k means head-k + tail-rest (e.g.: head-k + tail-(512-k)).

There also other arguments for fine-tuning:

``pooling_type`` indicates which feature will be used for classification. `mean` means
mean-pooling for hidden state of the whole sequence, `max` means max-pooling, default means
taking hidden state of `[CLS]` token as features.

``layer_learning_rate`` and ``layer_learning_rate_decay`` in ``run_classifier_discriminative.py``
indicates layer-wise decreasing layer rate (See Section 5.3.4).


## Further Pre-Trained Checkpoints

We upload IMDb-based further pre-trained checkpoints at
[here](https://drive.google.com/drive/folders/1Rbi0tnvsQrsHvT_353pMdIbRwDlLhfwM).

For other checkpoints, please contact us by e-mail.

## How to cite our paper

```text
@inproceedings{sun2019fine,
  title={How to fine-tune {BERT} for text classification?},
  author={Sun, Chi and Qiu, Xipeng and Xu, Yige and Huang, Xuanjing},
  booktitle={China National Conference on Chinese Computational Linguistics},
  pages={194--206},
  year={2019},
  organization={Springer}
}
```
