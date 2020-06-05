"""Reduce size of embeddings by aligning their vocabularies."""
import os
import logging

import numpy as np

import embeddix.utils.files as futils

__all__ = ('align_vocabs_and_models')

logger = logging.getLogger(__name__)


def _reduce_model(model, vocab, shared_vocab):
    _model = np.empty(shape=(len(shared_vocab), model.shape[1]))
    idx_to_word = {idx: word for word, idx in shared_vocab.items()}
    for idx, word in idx_to_word.items():
        _model[idx] = model[vocab[word]]
    return _model


def align_vocabs_and_models(embeddings_dirpath):
    """Align all models under dirpath on the same vocabulary."""
    logger.info('Aligning vocabularies under {}'
                .format(embeddings_dirpath))
    shared_vocab = futils.load_shared_vocab(embeddings_dirpath)
    logger.info('Shared vocabulary size = {}'.format(len(shared_vocab)))
    model_names = [filename.split('.npy')[0] for filename in
                   os.listdir(embeddings_dirpath) if filename.endswith('.npy')]
    logger.info('Processing models = {}'.format(model_names))
    for model_name in model_names:
        model_filepath = os.path.join(embeddings_dirpath,
                                      '{}.npy'.format(model_name))
        model = np.load(model_filepath)
        vocab_filepath = os.path.join(embeddings_dirpath,
                                      '{}.vocab'.format(model_name))
        vocab = futils.load_vocab(vocab_filepath)
        reduced_model = _reduce_model(model, vocab, shared_vocab)
        reduced_model_filepath = os.path.join(embeddings_dirpath,
                                              '{}-reduced'.format(model_name))
        np.save(reduced_model_filepath, reduced_model)
        reduced_vocab_filepath = os.path.join(
            embeddings_dirpath, '{}-reduced.vocab'.format(model_name))
        with open(reduced_vocab_filepath, 'w', encoding='utf-8') as output_str:
            for word, idx in shared_vocab.items():
                print('{}\t{}'.format(idx, word), file=output_str)
