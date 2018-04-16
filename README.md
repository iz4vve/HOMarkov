# High Order Markov Chains
#### Author: Pietro Mascolo

High order Markov Chains representation for sequences.
This packages contains a Markov Chain model that can be trained on integer sequences (which can be hashes for external labels).


Example use:
```ipython
>>> from HOMarkov import markov
>>>
>>> data = [0, 1, 2, 1, 1, 3, 4, 1, 5, 3]

# First order MC
>>> mc = markov.MarkovChain(6, 1)
>>> mc.fit(data)
>>>
>>> mc.transition_matrix
<6x6 sparse matrix of type '<class 'numpy.float64'>'
	with 0 stored elements in Dictionary Of Keys format>

>>> mc.transition_matrix.nonzero()
(array([0, 1, 1, 1, 1, 2, 3, 4, 5], dtype=int32),
 array([1, 2, 1, 3, 5, 1, 4, 1, 3], dtype=int32))

# dense representation (bad idea if the number of possible states or the order is high)
>>> mc.transition_matrix.todense()
matrix([[0.  , 1.  , 0.  , 0.  , 0.  , 0.  ],
        [0.  , 0.25, 0.25, 0.25, 0.  , 0.25],
        [0.  , 1.  , 0.  , 0.  , 0.  , 0.  ],
        [0.  , 0.  , 0.  , 0.  , 1.  , 0.  ],
        [0.  , 1.  , 0.  , 0.  , 0.  , 0.  ],
        [0.  , 0.  , 0.  , 1.  , 0.  , 0.  ]])




# Second order MC
>>> mc = markov.MarkovChain(6, 2)
>>> mc.fit(data)
>>> mc1.transition_matrix
<36x36 sparse matrix of type '<class 'numpy.float64'>'
	with 7 stored elements in Compressed Sparse Row format>

>>> mc1.transition_matrix.nonzero()
(array([ 1,  7,  8,  9, 13, 22, 25], dtype=int32),
 array([13, 22,  7, 25,  9, 11, 33], dtype=int32))


 # predict next state using the 1-st order MC
>>> initial_state = np.zeros(6)
>>> initial_state[1] = 1
>>> next_state = mc.predict_state(initial_state, 1)
>>> next_state
<1x6 sparse matrix of type '<class 'numpy.float64'>'
	with 4 stored elements in Compressed Sparse Row format>

# dense representation of the new state
>>> next_state.todense()
matrix([[0.  , 0.25, 0.25, 0.25, 0.  , 0.25]])
```
