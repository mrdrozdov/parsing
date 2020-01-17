import collections
import json
import os

from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper

import torch

import numpy as np

from parsing import (
    DioraConfig,
    DioraModel,
    DioraTokenizer,
    FixedLengthBatchSampler,
    SimpleDataset,
)


def read_corpus(path):
    corpus = []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            corpus.append(ex)
    return corpus


def get_vocab(corpus):
    if isinstance(corpus, dict):
        vocab = set()
        for x in corpus.values():
            vocab = vocab.union(set(get_vocab(x)))
    else:
        vocab = set()
        for ex in corpus:
            for w in ex["words"]:
                vocab.add(w)
    new_vocab = collections.OrderedDict()
    new_vocab[ELMoCharacterMapper.bos_token] = len(new_vocab)
    new_vocab[ELMoCharacterMapper.eos_token] = len(new_vocab)
    for w in sorted(vocab):
        new_vocab[w] = len(new_vocab)
    return new_vocab


def build_data_source(corpus, vocab):
    new_corpus = []
    for ex in corpus:
        new_corpus.append([vocab[w] for w in ex["words"]])
    return SimpleDataset(new_corpus)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data_file", default=None, type=str, required=True, help="The input training data file (a text file)."
    )
    args = parser.parse_args()

    config = DioraConfig()
    config.cuda = False
    config.elmo_options_path = os.path.expanduser("~/tmp/elmo_2x4096_512_2048cnn_2xhighway_options.json")
    config.elmo_weights_path = os.path.expanduser("~/tmp/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5")

    def collate_fn(batch):
        index, sents = zip(*batch)
        batch_map = collections.OrderedDict()
        batch_map["index"] = index
        batch_map["sents"] = torch.from_numpy(np.array(sents)).long()
        return batch_map

    corpus = read_corpus(args.train_data_file)
    vocab = get_vocab(corpus)
    dataset = build_data_source(corpus, vocab)
    sampler = FixedLengthBatchSampler(
        data_source=dataset, batch_size=2, include_partial=False, rng=np.random.RandomState(seed=11),
    )
    loader = torch.utils.data.DataLoader(
        dataset, shuffle=(sampler is None), batch_sampler=sampler, collate_fn=collate_fn,
    )

    for batch in loader:
        print(batch)

    model = DioraModel(config, vocab_to_cache=vocab)


if __name__ == "__main__":
    main()
