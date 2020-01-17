"""

Dataset splits were determined to match Stern et al.:

    We use the Penn Treebank (Marcus et al., 1993) for our English experiments, with
    standard splits of sections 2-21 for training, section 22 for development, and
    section 23 for testing.
"""

import collections
import json
import os

from nltk.corpus import ptb


def tree_to_tuple(tree):
    def helper(tr):
        if len(tr) == 1 and isinstance(tr[0], str):
            return [tr.label(), tr[0]]
        nodes = []
        for x in tr:
            nodes.append(helper(x))
        return [tr.label(), nodes]
    return helper(tree)


def get_raw_data():
    raw_data = {}
    fileids = ptb.fileids()

    obj_sofar = 0

    for fileid in fileids:
        corpus, section, _ = fileid.split('/')
        if corpus.lower() != 'wsj':
            continue
        section = int(section)
        if section >= 2 and section <= 21:
            split = 'train'
        elif section == 22:
            split = 'valid'
        elif section == 23:
            split = 'test'
        else:
            split = None
        sent_sofar = 0
        for y in ptb.parsed_sents(fileid):
            words, part_of_speech = zip(*y.pos())
            constituency_parse = tree_to_tuple(y)
            obj = collections.OrderedDict()
            obj['example_id'] = 'ptb{}'.format(obj_sofar)
            obj['file_id'] = fileid
            obj['sent_id'] = sent_sofar
            obj['words'] = words
            obj['part_of_speech'] = part_of_speech
            sent_sofar += 1
            obj_sofar += 1

            raw_data.setdefault('all', []).append(obj)
            if split is not None:
                raw_data.setdefault(split, []).append(obj)

    return raw_data


def run_convert(options):
    raw_data = get_raw_data()

    with open(os.path.join(options.output, 'ptb.jsonl'), 'w') as f:
        for obj in raw_data['all']:
            f.write(json.dumps(obj))
            f.write('\n')

    with open(os.path.join(options.output, 'test.jsonl'), 'w') as f:
        for obj in raw_data['test']:
            f.write(json.dumps(obj))
            f.write('\n')

    with open(os.path.join(options.output, 'valid.jsonl'), 'w') as f:
        for obj in raw_data['valid']:
            f.write(json.dumps(obj))
            f.write('\n')

    with open(os.path.join(options.output, 'train.jsonl'), 'w') as f:
        for obj in raw_data['train']:
            f.write(json.dumps(obj))
            f.write('\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--output', type=str, default=os.path.expanduser('~/.parsing/data/ptb'))
    options = parser.parse_args()

    if options.mode == 'convert':
        run_convert(options)
