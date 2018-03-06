"""
High order representation for Marckov Chains
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
import itertools
import numpy as np
import pandas as pd
from scipy import sparse

from sklearn import preprocessing
# TODO checks for allowed states

import progressbar


def get_progressbar(length):
    """
    Returns a progressbar with some widgets

    :param length: total number of steps in the progressbar
    """
    bar = progressbar.ProgressBar(
        max_value=length,
        widgets=[
            ' [', progressbar.Timer(), '] ',
            progressbar.Bar(),
            progressbar.Counter(),
            " -> ",
            progressbar.Percentage(),
            ' (', progressbar.ETA(), ') ',
        ]
    )

    return bar


class MarkovChain(object):
    """
    High order Markov chain representation of sequences of states.
    """

    def __init__(self, n_states, order=1):
        """
        :param n_states: number of possible states
        :param order: order of the Markov model
        """
        self.number_of_states = n_states
        self.order = order
        if order == 1:
            self.possible_states = list(range(n_states))
        else:
            self.possible_states = {
                j: i for i, j in
                enumerate(itertools.product(range(n_states), repeat=order))
            }

        # allocate transition matrix
        self.transition_matrix = sparse.dok_matrix((
            (len(self.possible_states), len(self.possible_states))
        ), dtype=np.float64)

    def normalize_transitions(self):
        """
        Normalizes the transition matrix by row
        """
        self.transition_matrix = preprocessing.normalize(
            self.transition_matrix, norm="l1"
        )

    def update_transition_matrix(self, states_sequence, normalize=True):
        """
        Updates transition matrix with a single sequence of states
        :param states_sequence: sequence of state IDs
        :type states_sequence: iterable(int)
        :param normalize: whether the transition matrix is normalized after the
           update (set to False and manually triggered when
           training multiple sequences)
        """
        if self.order == 1:
            for i in range(len(states_sequence) - 1):
                s1 = states_sequence[i] - 1
                s2 = states_sequence[i + 1] - 1
                self.transition_matrix[s1, s2] += 1
        else:
            visited_states = [
                states_sequence[i: i + self.order]
                for i in range(len(states_sequence) - self.order + 1)
            ]

            for n, i in enumerate(visited_states):
                try:
                    self.transition_matrix[
                        self.possible_states[tuple(i)],
                        self.possible_states[tuple(visited_states[n + 1])]
                    ] += 1
                except IndexError:
                    pass

        if normalize:
            self.normalize_transitions()

    def fit(self, state_sequences):
        """
        Fits the model with many sequences of states
        :param state_sequences: iterable of state sequences
        """
        try:
            _bar = get_progressbar(len(state_sequences))
            for n, sequence in enumerate(state_sequences):
                _bar.update(n)
                self.update_transition_matrix(sequence, normalize=False)
        except TypeError:  # not a list of sequences
            self.update_transition_matrix(state_sequences)
        finally:
            self.normalize_transitions()

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
        df = pd.SparseDataFrame(self.transition_matrix)
        df.index = sorted(self.possible_states.keys())
        df.columns = sorted(self.possible_states.keys())

        return df

    def predict_state(self, current_state, num_steps=1):
        """
        :param current_state: array representing current state
        :param num_steps: number of steps for which a prediction is made
        :return: evolved state arrays
        """
        next_state = sparse.csr_matrix(current_state).dot(
                self.sparse_power(num_steps)
        )

        return next_state[0]

    def sparse_power(self, n):
        """
        Due to a bug somewhere in scipy.sparse
        """
        if n == 1:
            return self.transition_matrix
        acc = self.transition_matrix
        for _ in range(n - 1):
            acc.multiply(self.transition_matrix)
            return acc
