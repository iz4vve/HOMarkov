"""
High order representation for Markov Chains
Copyright (C) 2017 - Pietro Mascolo
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Author: Pietro Mascolo
Email: iz4vve@gmail.com
"""
# pylint: disable=E1101
import collections
import itertools
import numpy as np
import pandas as pd
from scipy import sparse

from sklearn import preprocessing

import networkx as nx


def pairwise(iterable):
    """
    Returns a sequence of pairs from an iterable
    """
    first, second = itertools.tee(iterable)
    next(second, None)
    return zip(first, second)


class MarkovChain(object):
    """
    High order Markov chain representation of sequences of states.

    The class is designed to work with a numeric list of state IDs 
    in the range [0; number of states - 1].
    If your state have different names, please generate a sorted
    map to a range(number_of_states).
    """

    def __init__(self, n_states, order=1, verbose=False):
        """
        :param n_states: number of possible states
        :param order: order of the Markov model
        """
        self.number_of_states = n_states
        self.order = order
        self.verbose = verbose

        self.possible_states = {
            j: i for i, j in
            enumerate(itertools.product(range(n_states), repeat=order))
        }

        # allocate transition matrix
        self.transition_matrix = sparse.dok_matrix((
            (len(self.possible_states), len(self.possible_states))
        ), dtype=np.float64)

    def _normalize_transitions(self):
        """
        Normalizes the transition matrix by row
        """
        self.transition_matrix = preprocessing.normalize(
            self.transition_matrix, norm="l1"
        )

    def _update_transition_matrix(self, states_sequence, normalize=True):
        """
        Updates transition matrix with a single sequence of states
        :param states_sequence: sequence of state IDs
        :type states_sequence: iterable(int)
        :param normalize: whether the transition matrix is normalized after the
           update (set to False and manually triggered when
           training multiple sequences)
        """

        visited_states = [
            states_sequence[i: i + self.order]
            for i in range(len(states_sequence) - self.order + 1)
        ]

        for state_index, i in enumerate(visited_states):
            try:
                self.transition_matrix[
                    self.possible_states[tuple(i)],
                    self.possible_states[tuple(visited_states[
                        state_index + self.order
                    ])]
                ] += 1
            except IndexError:
                pass

        if normalize:
            self._normalize_transitions()

    def fit(self, state_sequences):
        """
        Fits the model with many sequences of states
        :param state_sequences: iterable of state sequences
        """
        try:
            for index, sequence in enumerate(state_sequences):
                if self.verbose and not index % 10000:
                    print(f"{index} sequences processed")
                self._update_transition_matrix(sequence, normalize=False)
        except TypeError:  # not a list of sequences
            self._update_transition_matrix(state_sequences)
        finally:
            self._normalize_transitions()

    def transition_df(self):
        """
        This returns the transition matrix in form of a pandas dataframe.
        The results are not stored in the model to avoid redundancy.

        Example:
                 A,A     A,B     A,C     ...
            A,A  1       0       0       ...
            A,B  0.33    0.33    0.33    ...
            A,C  0.66    0       0.33    ...
            B,A  0       0       0       ...
            B,B  0       0.5     0.5     ...
            B,C  0.33    0       0.66    ...
            C,A  1       0       0       ...
            C,B  0       1       0       ...
            C,C  0       0       1       ...


        :return: Transition states data frame
        """
        sdf = pd.SparseDataFrame(self.transition_matrix)

        sdf.index = sorted(self.possible_states)
        sdf.columns = sorted(self.possible_states)

        return sdf.fillna(0)

    def predict_state(self, current_state, num_steps=1):
        """
        :param current_state: array representing current state
        :param num_steps: number of steps for which a prediction is made
        :return: evolved state arrays
        """
        _next_state = sparse.csr_matrix(current_state).dot(
            self.sparse_power(num_steps)
        )

        return _next_state[0]

    def sparse_power(self, power):
        """
        Due to a bug somewhere in scipy.sparse
        """

        acc = self.transition_matrix
        for _ in range(power - 1):
            acc.multiply(self.transition_matrix)
        return acc

    def possible_states_lookup(self):
        """
        Reverses keys and values of self.possible_states
        (for lookup in transition_matrix)
        """
        return {v: k for k, v in self.possible_states.items()}

    def evolve_states(self, initial_state, num_steps=1, threshold=0.1):
        """
        Evolves the states for num_steps iterations and returns
        a mapping of initial, final and intermediate states.

        :param initial_state: Initial state for the evolution
        :param num_steps: number of iterations
        :param threshold: minimum probability for a state to be considered

        :rtype: defaultdict(list)
        """
        state_id = 0
        state_vector = collections.defaultdict(list)

        for step in range(num_steps + 1):

            if not state_vector:
                start = initial_state.nonzero()
                for i in start[0]:
                    state_repr = np.zeros(self.transition_matrix.shape[0])
                    state_repr[i] = 1
                    state_vector[step] += [
                        {
                            "state_id": state_id,
                            "state": i,
                            "weight": initial_state[i],
                            "prev_state": None,
                            "state_repr": state_repr,
                            "actual": initial_state[i]
                        }
                    ]
                continue

            # get last state
            last_states = state_vector.get(step - 1)

            for _state in last_states:
                prediction = self.predict_state(_state.get("state_repr"))

                _, predicted_states = prediction.nonzero()

                for predicted_state in sorted(predicted_states):
                    state_id += 1
                    state_repr = np.zeros(self.transition_matrix.shape[0])
                    state_repr[predicted_state] = 1

                    if prediction[
                            0, predicted_state
                    ] * _state.get("actual") > threshold:

                        state_vector[step] += [
                            {
                                "state_id": state_id,
                                "state": predicted_state,
                                "weight": prediction[0, predicted_state],
                                "prev_state": _state.get("state_id"),
                                "state_repr": state_repr,
                                "actual": prediction[
                                    0, predicted_state
                                ] * _state.get("actual")
                            }
                        ]

        return state_vector

    @staticmethod
    def generate_graph(states_vector, actual=True):
        """
        Generates a DiGraph from a states vector

        :param states_vector: representation of time evolution of states
        :type states_vector: same type as return values from evolve_states
            dict(list())
        :param actual: whether actuals or transition probabilities
            are used as weights for the edges
        :returns: Directed weighted graph of state evolution
        :rtype: networkx.DiGraph
        """
        G = nx.DiGraph()

        for _, states in states_vector.items():

            for state in states:
                G.add_node(
                    state["state_id"],
                    label=state["state"]
                )

                if state["prev_state"] is not None:
                    start = state["prev_state"]
                    end = state["state_id"]
                    weights = [state["actual"], state["weight"]]

                    if not actual:
                        weights = list(reversed(weights))

                    G.add_edge(
                        start,
                        end,
                        weight=weights[0],
                        alternative_weight=weights[1]
                    )

        return G

    @staticmethod
    def build_pos(states):
        pos = dict()
        for key, state in states.items():
            for n, _state in enumerate(state):
                pos[_state["state_id"]] = (key, -n)
        return pos
