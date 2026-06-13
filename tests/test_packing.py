"""Tests for lithos.data.packing — sequence counting and (x, y) windows."""

import numpy as np
from lithos.data.packing import get_sequence, num_sequences


def test_num_sequences():
    assert num_sequences(10, 4) == 2  # (10-1)//4
    assert num_sequences(9, 4) == 2
    assert num_sequences(8, 4) == 1
    assert num_sequences(4, 4) == 0  # not enough for a label token
    assert num_sequences(3, 4) == 0


def test_get_sequence_is_shifted_window():
    tokens = np.arange(20)
    x0, y0 = get_sequence(tokens, 0, 4)
    assert x0.tolist() == [0, 1, 2, 3]
    assert y0.tolist() == [1, 2, 3, 4]
    x1, y1 = get_sequence(tokens, 1, 4)
    assert x1.tolist() == [4, 5, 6, 7]
    assert y1.tolist() == [5, 6, 7, 8]
