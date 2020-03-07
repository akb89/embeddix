"""Welcome to embeddix.

This is the entry point of the application.
"""
import os

import argparse
import logging
import logging.config

from bert_serving.client import BertClient

import embeddix.utils.config as cutils


logging.config.dictConfig(
    cutils.load(
        os.path.join(os.path.dirname(__file__), 'logging', 'logging.yml')))

logger = logging.getLogger(__name__)


__all__ = ('load_vocab', 'save_to_text')


# def load_vocab(vocab_filepath):
#     """Load words list from vocab text file."""
#     logger.info('Loading vocabulary from {}'.format(vocab_filepath))
#     words = []
#     with open(vocab_filepath, 'r', encoding='utf-8') as input_stream:
#         for line in input_stream:
#             words.append(line.strip())
#     return words


def load_vocab(vocab_filepath):
    """Load word_to_idx dict mapping from .vocab filepath."""
    word_to_idx = {}
    logger.info('Loading vocabulary from {}'.format(vocab_filepath))
    with open(vocab_filepath, 'r', encoding='utf-8') as input_stream:
        for line in input_stream:
            linesplit = line.strip().split('\t')
            word_to_idx[linesplit[1]] = int(linesplit[0])
    return word_to_idx


def _extract_vocab(model_filepath):
    words = []
    with open(model_filepath, 'r', encoding='utf-8') as m_stream:
        for line in m_stream:
            line = line.strip()
            word = line.split(' ')[0]
            words.append(word)
    vocab_filepath = '{}.vocab'.format(
        os.path.abspath(model_filepath).split('.txt')[0])
    with open(vocab_filepath, 'w', encoding='utf-8') as v_stream:
        for idx, word in enumerate(words):
            print('{}\t{}'.format(idx, word), file=v_stream)


def extract_vocab(args):
    """Extract vocabulary from txt vectors."""
    _extract_vocab(args.vectors)


def save_to_text(vocab, model, filepath):
    """Save vocab + numpy model to unique text file.

    vocab should be a word_to_idx dict
    model should be a numpy ndarray
    filepath should be the full filepath to output text file
    """
    with open(filepath, 'w', encoding='utf-8') as otp:
        for word, idx in vocab.items():
            vector = ' '.join([str(item) for item in model[idx].tolist()])
            print('{} {}'.format(word, vector), file=otp)


def save_to_numpy():
    pass


def _convert_bert_to_text(vocab_filepath):
    # https://github.com/google-research/bert/issues/60
    vocab = load_vocab(vocab_filepath)
    bert_opt_filepath = os.path.join(
        os.path.dirname(vocab_filepath),
        'bert-{}.txt'.format(os.path.basename(vocab_filepath).split('.vocab')[0]))
    bc = BertClient()
    words = list(vocab.keys())
    vectors = bc.encode(words)
    with open(bert_opt_filepath, 'w', encoding='utf-8') as opt_stream:
        for word, vector in zip(words, vectors):
            print('{} {}'.format(word, vector), file=opt_stream)


def convert_bert_to_text(args):
    """Generate a vectors txt file from BERT for a given vocabulary."""
    _convert_bert_to_text(args.vocab)


def main():
    """Launch embeddix."""
    parser = argparse.ArgumentParser(prog='embeddix')
    subparsers = parser.add_subparsers()
    parser_extract = subparsers.add_parser(
        'extract', formatter_class=argparse.RawTextHelpFormatter,
        help='extract vocab from vectors txt file')
    parser_extract.set_defaults(func=extract_vocab)
    parser_extract.add_argument('-v', '--vectors', required=True,
                                help='input vectors in txt format')
    parser_convert = subparsers.add_parser(
        'convert', formatter_class=argparse.RawTextHelpFormatter,
        help='convert BERT model to txt file')
    parser_convert.set_defaults(func=convert_bert_to_text)
    parser_convert.add_argument('-v', '--vocab', required=True,
                                help='absolute path to vocabulary')
    args = parser.parse_args()
    args.func(args)
