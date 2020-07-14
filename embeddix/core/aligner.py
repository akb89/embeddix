"""Align vocabularies across matrices."""
import logging

from scipy import sparse

__all__ = ('align_vocabs_and_sparse_matrices')

logger = logging.getLogger(__name__)


def _reduce_model(model, vocab, shared_word_to_idx):
    rows = []
    columns = []
    data = []
    model = model.tocoo()
    for i, j, v in zip(model.row, model.col, model.data):
        if vocab[i] in shared_word_to_idx and vocab[j] in shared_word_to_idx:
            rows.append(shared_word_to_idx[vocab[i]])
            columns.append(shared_word_to_idx[vocab[j]])
            data.append(v)
    return sparse.csr_matrix((data, (rows, columns)),
                             shape=(len(shared_word_to_idx),
                                    len(shared_word_to_idx)),
                             dtype='f')


def _align_models(model_1, vocab_1, model_2, vocab_2, shared_word_to_idx):
    model_1 = _reduce_model(model_1, vocab_1, shared_word_to_idx)
    model_2 = _reduce_model(model_2, vocab_2, shared_word_to_idx)
    return model_1, model_2


def align_vocabs_and_matrices(model_1, vocab_1, model_2, vocab_2):
    """Align vocabularies and sparse matrices."""
    shared_words = set(word for word in vocab_1.values() if word
                       in vocab_2.values())
    shared_word_to_idx = {word: idx for idx, word in enumerate(shared_words)}
    model_1, model_2 = _align_models(model_1, vocab_1, model_2, vocab_2,
                                     shared_word_to_idx)
    shared_vocab = {idx: word for word, idx in shared_word_to_idx.items()}
    return model_1, model_2, shared_vocab
