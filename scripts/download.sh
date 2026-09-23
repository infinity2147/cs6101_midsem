#!/usr/bin/env bash
# Downloads everything the SQuAD pipeline needs (all mirrors that are reachable without HF hub).
set -euo pipefail
D=${DATA:-data}; S=https://s3.amazonaws.com/models.huggingface.co/bert
mkdir -p $D/models/distilbert-base-uncased $D/models/monot5-base $D/squad $D/w2v
curl -sSL -o $D/squad/train-v1.1.json https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/train-v1.1.json
curl -sSL -o $D/squad/dev-v1.1.json   https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v1.1.json
curl -sSL -o $D/models/distilbert-base-uncased/pytorch_model.bin $S/distilbert-base-uncased-pytorch_model.bin
curl -sSL -o $D/models/distilbert-base-uncased/config.json       $S/distilbert-base-uncased-config.json
curl -sSL -o $D/models/distilbert-base-uncased/vocab.txt         $S/bert-base-uncased-vocab.txt
curl -sSL -o $D/models/monot5-base/pytorch_model.bin $S/castorini/monot5-base-msmarco/pytorch_model.bin
curl -sSL -o $D/models/monot5-base/config.json       $S/castorini/monot5-base-msmarco/config.json
curl -sSL -o $D/models/monot5-base/spiece.model      $S/t5-spiece.model
curl -sSL -o $D/w2v/enwiki_20180420_100d.txt.bz2 https://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_100d.txt.bz2
