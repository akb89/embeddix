"""Exposed functions."""
from scipy import sparse

from .utils.files import _get_shared_vocab as get_shared_vocab
from .utils.files import load_vocab
from .utils.files import load_shared_vocab
from .utils.files import count_lines
from .core.reducer import _reduce_model as reduce_model
from .core.converter import convert_to_txt
from .core.aligner import align_vocabs_and_matrices as align


def load_sparse(matrix_filepath):
    """Load scipy sparse matrix."""
    return sparse.load_npz(matrix_filepath)
