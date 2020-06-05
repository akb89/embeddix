"""Welcome to embeddix.

This is the entry point of the application.
"""
import os

import argparse
import logging
import logging.config

import embeddix.core.converter as converter
import embeddix.core.extractor as extractor
import embeddix.core.reducer as reducer
import embeddix.utils.config as cutils


logging.config.dictConfig(
    cutils.load(
        os.path.join(os.path.dirname(__file__), 'logging', 'logging.yml')))

logger = logging.getLogger(__name__)


def _reduce(args):
    reducer.align_vocabs_and_models(args.embeddings)


def _convert(args):
    if args.to == 'numpy':
        if not args.embeddings.endswith('.txt'):
            raise Exception('Invalid input file: should be a text file '
                            'ending with .txt')
        converter.convert_to_numpy(args.embeddings)
    else:
        if not args.embeddings.endswith('.npy'):
            raise Exception('Invalid input file: should be a numpy file '
                            'ending with .npy')
        if not args.vocab:
            raise Exception('Converting to txt requires specifying the '
                            '--vocab parameter')
        converter.convert_to_txt(args.embeddings, args.vocab)


def _extract(args):
    extractor.extract_vocab(args.embeddings)


def main():
    """Launch embeddix."""
    parser = argparse.ArgumentParser(prog='embeddix')
    subparsers = parser.add_subparsers()
    parser_extract = subparsers.add_parser(
        'extract', formatter_class=argparse.RawTextHelpFormatter,
        help='extract vocab from embeddings txt file')
    parser_extract.set_defaults(func=_extract)
    parser_extract.add_argument('-e', '--embeddings', required=True,
                                help='input embedding in txt format')
    parser_convert = subparsers.add_parser(
        'convert', formatter_class=argparse.RawTextHelpFormatter,
        help='convert embeddings to and from numpy and txt formats')
    parser_convert.set_defaults(func=_convert)
    parser_convert.add_argument('-t', '--to', choices=['numpy', 'txt'],
                                help='output format: numpy or text')
    parser_convert.add_argument('-v', '--vocab',
                                help='absolute path to vocabulary')
    parser_convert.add_argument('-e', '--embeddings', required=True,
                                help='absolute path to embeddings file')
    parser_reduce = subparsers.add_parser(
        'reduce', formatter_class=argparse.RawTextHelpFormatter,
        help='align numpy model vocabularies. Will also align the .npy models')
    parser_reduce.set_defaults(func=_reduce)
    parser_reduce.add_argument('-d', '--embeddings', required=True,
                               help='absolute path to .npy models '
                                    'directory. The directory should '
                                    'contain the .vocab files '
                                    'corresponding to the .npy models.')
    args = parser.parse_args()
    args.func(args)
