"""Test shared vocab."""

import embeddix


def test_get_shared_vocab():
    vocabs = [{
        'a': 0,
        'b': 1,
        'c': 2
    }, {
        'c': 0,
        'a': 1,
        'z': 2,
        'y': 10
    }, {
        'p': 0,
        'c': 42
    }]
    shared_vocab = embeddix.get_shared_vocab(vocabs)
    assert len(shared_vocab) == 1
    for word, idx in shared_vocab.items():
        assert word == 'c'
        assert idx == 0
