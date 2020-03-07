"""Welcome to embeddix.

This is the entry point of the application.
"""
import os

import logging
import logging.config

import embeddix.utils.config as cutils


logging.config.dictConfig(
    cutils.load(
        os.path.join(os.path.dirname(__file__), 'logging', 'logging.yml')))

logger = logging.getLogger(__name__)


__all__ = ('load_vocab')


def load_vocab(vocab_filepath):
    """Load word_to_idx dict mapping from .vocab filepath"""
    word_to_idx = {}
    logger.info('Loading vocabulary from {}'.format(vocab_filepath))
    with open(vocab_filepath, 'r', encoding='utf-8') as input_stream:
        for line in input_stream:
            linesplit = line.strip().split('\t')
            word_to_idx[linesplit[1]] = int(linesplit[0])
    return word_to_idx


def main():
    """Launch embeddix."""
    pass