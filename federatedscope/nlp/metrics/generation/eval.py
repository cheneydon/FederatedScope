#!/usr/bin/env python
__author__ = 'xinya'

# Modified by:
# Yu Chen <cheny39@rpi.edu>

# Changelog:
# Convert to python 3
# Support verbose mode
# Change output format

from json import encoder
from collections import defaultdict
from .bleu.bleu import Bleu
from .meteor.meteor import Meteor
from .rouge.rouge import Rouge

encoder.FLOAT_REPR = lambda o: format(o, '.4f')


class QGEvalCap:
    def __init__(self, gts, res):
        self.gts = gts
        self.res = res

    def evaluate(self, include_meteor=True, verbose=False):
        output = {}
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            # (Rouge(), "ROUGE_L"),
        ]
        if include_meteor:
            scorers.append((Meteor(), "METEOR"))

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            # print 'computing %s score...'%(scorer.method())
            score, scores = scorer.compute_score(self.gts, self.res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    if verbose:
                        print("%s: %0.5f"%(m, sc))
                    # output.append(sc)
                    output[m] = sc
            else:
                if verbose:
                    print("%s: %0.5f"%(method, score))
                # output.append(score)
                output[method] = score
        return output


def eval(out_file, src_file, tgt_file):
    pairs = []
    with open(src_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            pair = {}
            pair['tokenized_sentence'] = line[:-1]
            pairs.append(pair)

    with open(tgt_file, 'r', encoding='utf-8') as infile:
        cnt = 0
        for line in infile:
            pairs[cnt]['tokenized_question'] = line[:-1]
            cnt += 1

    output = []
    with open(out_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            line = line[:-1]
            output.append(line)

    for idx, pair in enumerate(pairs):
        pair['prediction'] = output[idx]

    res = defaultdict(lambda: [])
    gts = defaultdict(lambda: [])
    for pair in pairs[:]:
        key = pair['tokenized_sentence']
        # res[key] = [pair['prediction'].encode('utf-8')]
        # gts[key].append(pair['tokenized_question'].encode('utf-8'))
        res[key] = [pair['prediction']]
        gts[key].append(pair['tokenized_question'])

    QGEval = QGEvalCap(gts, res)
    return QGEval.evaluate()


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-out", "--out_file", dest="out_file", default="./output/pred.txt", help="output file to compare")
    parser.add_argument("-src", "--src_file", dest="src_file", default="../data/processed/src-test.txt", help="src file")
    parser.add_argument("-tgt", "--tgt_file", dest="tgt_file", default="../data/processed/tgt-test.txt", help="target file")
    args = parser.parse_args()

    print("scores: \n")
    eval(args.out_file, args.src_file, args.tgt_file)
