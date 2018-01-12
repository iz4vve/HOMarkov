import pytest
import numpy as np
import pandas as pd

from HOMarkov import markov

MOCK_SEQUENCE = [0, 1, 2, 1, 1, 3, 4, 1, 5, 3]
MOCK_SEQUENCE_SMALL = [
    0, 2, 1, 1, 2, 1, 2, 1, 1, 2, 1, 2, 2, 2,
    2, 2, 2, 1, 2, 1, 1, 2, 2, 0, 0, 0, 0, 0
]
EXPECTED_POSSIBLE_STATES_ORDER_1 = [0, 1, 2, 3, 4, 5]
EXPECTED_POSSIBLE_STATES_ORDER_2 = {
    (0, 0): 0, (0, 1): 1, (0, 2): 2, (0, 3): 3, (0, 4): 4, (0, 5): 5, (1, 0): 6,
    (1, 1): 7, (1, 2): 8, (1, 3): 9, (1, 4): 10, (1, 5): 11, (2, 0): 12,
    (2, 1): 13, (2, 2): 14, (2, 3): 15, (2, 4): 16, (2, 5): 17, (3, 0): 18,
    (3, 1): 19, (3, 2): 20, (3, 3): 21, (3, 4): 22, (3, 5): 23, (4, 0): 24,
    (4, 1): 25, (4, 2): 26, (4, 3): 27, (4, 4): 28, (4, 5): 29, (5, 0): 30,
    (5, 1): 31, (5, 2): 32, (5, 3): 33, (5, 4): 34, (5, 5): 35
}
EXPECTED_POSSIBLE_STATES_ORDER_2_SMALL = {
    (0, 0): 0, (0, 1): 1, (0, 2): 2, (1, 0): 3,
    (1, 1): 4, (1, 2): 5, (2, 0): 6,
    (2, 1): 7, (2, 2): 8
}
np.random.seed(1234)
MOCK_DATA_RANDOM = [np.random.randint(0, 3, 50) for _ in range(100)]


def test_possible_states_order_1():
    hom = markov.MarkovChain(6, 1)
    hom.update_transition_matrix(MOCK_SEQUENCE)
    assert hom.possible_states == EXPECTED_POSSIBLE_STATES_ORDER_1


def test_possible_states_order_2():
    hom = markov.MarkovChain(6, 2)
    hom.update_transition_matrix(MOCK_SEQUENCE)
    assert hom.possible_states == EXPECTED_POSSIBLE_STATES_ORDER_2


def test_possible_states_order_2_small():
    hom = markov.MarkovChain(3, 2)
    hom.update_transition_matrix(MOCK_SEQUENCE_SMALL)
    assert hom.possible_states == EXPECTED_POSSIBLE_STATES_ORDER_2_SMALL
    df = hom.transition_df()
    assert set(df.columns.values) == set(hom.possible_states.keys())
    assert set(df.index.values) == set(hom.possible_states.keys())


def test_normalization():
    hom = markov.MarkovChain(3, 1)
    hom2 = markov.MarkovChain(3, 2)
    hom3 = markov.MarkovChain(3, 3)
    hom.fit(MOCK_DATA_RANDOM)
    hom2.fit(MOCK_DATA_RANDOM)
    hom3.fit(MOCK_DATA_RANDOM)
    sums = hom.transition_matrix.sum(axis=1)
    sums2 = hom2.transition_matrix.sum(axis=1)
    sums3 = hom3.transition_matrix.sum(axis=1)
    # some rows might be all zeros...
    assert all(i == pytest.approx(1, 0.01) for i in sums if i)
    assert all(i == pytest.approx(1, 0.01) for i in sums2 if i)
    assert all(i == pytest.approx(1, 0.01) for i in sums3 if i)


def test_next_state():
    hom = markov.MarkovChain(3, 1)
    hom.fit(MOCK_DATA_RANDOM)
    # transition: 0.33583959899749372, 0.34335839598997492, 0.32080200501253131
    # taken from the initial transition matrix after training
    initial_state = np.array(
        [0.33583959899749372, 0.34335839598997492, 0.32080200501253131]
    )
    next_state = hom.predict_state(initial_state, num_steps=2)
    assert next_state[0, 1] == pytest.approx(0.326, 0.01)


def test_next_state_high_order():
    hom = markov.MarkovChain(3, 2)
    hom.fit(MOCK_DATA_RANDOM)

    # take a row from the initial transition matrix after training
    # as initial state.
    initial_state = hom.transition_matrix.todense()[1, :]

    next_state = hom.predict_state(initial_state, num_steps=2)
    assert next_state[0, 1] == pytest.approx(0.125, 0.01)
