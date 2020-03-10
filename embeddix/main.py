"""Welcome to embeddix.

This is the entry point of the application.
"""
import os

import argparse
import logging
import logging.config

import numpy as np

from tqdm import tqdm
from bert_serving.client import BertClient

import embeddix.utils.config as cutils


logging.config.dictConfig(
    cutils.load(
        os.path.join(os.path.dirname(__file__), 'logging', 'logging.yml')))

logger = logging.getLogger(__name__)


__all__ = ('load_vocab', 'save_to_text', '_get_shared_vocab')


def save_vocab(vocab, output_vocab_filepath):
    """Save vocab to filepath."""
    with open(output_vocab_filepath, 'w', encoding='utf-8') as output_stream:
        for word, idx in vocab.items():
            print('{}\t{}'.format(idx, word), file=output_stream)


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


def save_to_numpy(vocab, model, filepath):
    """Save vocab + numpy model to numpy."""
    np.save(filepath, model)
    save_vocab(vocab, '{}.vocab'.format(filepath))


def _convert_numpy_to_text(model_filepath, vocab_filepath):
    vocab = load_vocab(vocab_filepath)
    model = np.load(model_filepath)
    txt_model_fp = os.path.join(
        os.path.dirname(model_filepath),
        '{}.txt'.format(os.path.basename(model_filepath).split('.npy')[0]))
    save_to_text(vocab, model, txt_model_fp)


def _convert_bert_to_text(vocab_filepath):
    """Generate a vectors txt file from BERT for a given vocabulary."""
    # https://github.com/google-research/bert/issues/60
    vocab = load_vocab(vocab_filepath)
    bert_opt_filepath = os.path.join(
        os.path.dirname(vocab_filepath),
        'bert-{}.txt'.format(
            os.path.basename(vocab_filepath).split('.vocab')[0]))
    bert_client = BertClient()
    with open(bert_opt_filepath, 'w', encoding='utf-8') as opt_stream:
        for word in tqdm(vocab, total=len(vocab)):
            vector = bert_client.encode([word])[0]
            print('{} {}'.format(word, ' '.join(str(x) for x in vector)),
                  file=opt_stream)


def _convert(args):
    if args.what == 'bert':
        _convert_bert_to_text(args.vocab)
    elif args.what == 'numpy':
        _convert_numpy_to_text(args.model, args.vocab)


def _get_shared_vocab(vocabs):
    shared_words = set()
    for word in vocabs[0].keys():
        is_found_in_all = True
        for vocab in vocabs[1:]:
            if word not in vocab:
                is_found_in_all = False
                break
        if is_found_in_all:
            shared_words.add(word)
    return {word: idx for idx, word in enumerate(shared_words)}


def _load_shared_vocab(vocabs_dirpath):
    vocabs_names = [filename for filename in os.listdir(vocabs_dirpath) if
                    filename.endswith('.vocab')]
    vocabs = [load_vocab(os.path.join(vocabs_dirpath, vocab_name))
              for vocab_name in vocabs_names]
    return _get_shared_vocab(vocabs)


def _reduce_model(model, vocab, shared_vocab):
    _model = np.empty(shape=(len(shared_vocab), model.shape[1]))
    idx_to_word = {idx: word for word, idx in shared_vocab.items()}
    for idx, word in idx_to_word.items():
        _model[idx] = model[vocab[word]]
    return _model


def _align_vocabs_and_models(args):
    logger.info('Aligning vocabularies under {}'.format(args.model_dir))
    shared_vocab = _load_shared_vocab(args.model_dir)
    logger.info('Shared vocabulary size = {}'.format(len(shared_vocab)))
    # model_names = [filename.split('.npy')[0] for filename in
    #                os.listdir(args.model_dir) if filename.endswith('.npy')]
    # for model_name in model_names:
    #     model_filepath = os.path.join(args.model_dir,
    #                                   '{}.npy'.format(model_name))
    #     model = np.load(model_filepath)
    #     vocab_filepath = os.path.join(args.model_dir,
    #                                   '{}.vocab'.format(model_name))
    #     vocab = load_vocab(vocab_filepath)
    #     reduced_model = _reduce_model(model, vocab, shared_vocab)
    #     reduced_model_filepath = os.path.join(args.model_dir,
    #                                           '{}-reduced'.format(model_name))
    #     np.save(reduced_model_filepath, reduced_model)
    # output_vocab_filepath = os.path.join(args.model_dir, 'shared.vocab')
    # save_vocab(shared_vocab, output_vocab_filepath)



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
    parser_convert.set_defaults(func=_convert)
    parser_convert.add_argument('-w', '--what', required=True,
                                choices=['bert', 'numpy'],
                                help='absolute path to vocabulary')
    parser_convert.add_argument('-v', '--vocab', required=True,
                                help='absolute path to vocabulary')
    parser_convert.add_argument('-m', '--model',
                                help='absolute path to numpy model')
    parser_align_vocab = subparsers.add_parser(
        'align-vocab', formatter_class=argparse.RawTextHelpFormatter,
        help='align numpy model vocabularies. Will also align the .npy models')
    parser_align_vocab.set_defaults(func=_align_vocabs_and_models)
    parser_align_vocab.add_argument('-i', '--model-dir', required=True,
                                    help='absolute path to .npy models '
                                         'directory. The directory should '
                                         'contain the .vocab file '
                                         'corresponding to the .npy model.')
    args = parser.parse_args()
    args.func(args)
