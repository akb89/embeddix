"""Files utils."""
import os
import logging

import numpy as np

from scipy import sparse

__all__ = ('save_vocab', 'load_vocab', 'count_lines', 'load_shared_vocab',
           'load_sparse', 'save_sparse', 'save_dense')

logger = logging.getLogger(__name__)


def load_sparse(matrix_filepath):
    """Load scipy sparse matrix."""
    return sparse.load_npz(matrix_filepath)


def save_sparse(matrix_filepath, matrix):
    """Save scipy sparse matrix to file."""
    logger.info('Saving scipy sparse matrix to {}'.format(matrix_filepath))
    sparse.save_npz(matrix_filepath, matrix)


def save_dense(matrix_filepath, matrix):
    """Save numpy dense matrix to file."""
    logger.info('Saving numpy dense matrix to {}'.format(matrix_filepath))
    np.save(matrix_filepath, matrix)


def save_vocab(vocab_filepath, vocab):
    """Save vocabulary to file."""
    logger.info('Saving vocabulary to {}'.format(vocab_filepath))
    with open(vocab_filepath, 'w', encoding='utf-8') as vocab_stream:
        for word, idx in vocab.items():
            print('{}\t{}'.format(idx, word), file=vocab_stream)


def load_vocab(vocab_filepath):
    """Load word_to_idx dict mapping from .vocab filepath."""
    word_to_idx = {}
    logger.info('Loading vocabulary from {}'.format(vocab_filepath))
    with open(vocab_filepath, 'r', encoding='utf-8') as input_stream:
        for line in input_stream:
            linesplit = line.strip().split('\t')
            word_to_idx[linesplit[1]] = int(linesplit[0])
    return word_to_idx


def count_lines(input_filepath):
    """Count number of non-empty lines in file."""
    counter = 0
    with open(input_filepath, 'r', encoding='utf-8') as input_str:
        for line in input_str:
            if line.strip():
                counter += 1
    return counter


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


def load_shared_vocab(vocabs_dirpath):
    """Get intersection of all vocabularies under dirpath."""
    vocabs_names = [filename for filename in os.listdir(vocabs_dirpath) if
                    filename.endswith('.vocab')]
    vocabs = [load_vocab(os.path.join(vocabs_dirpath, vocab_name))
              for vocab_name in vocabs_names]
    return _get_shared_vocab(vocabs)
