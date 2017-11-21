# High Order Markov Chains
#### Author: Pietro Mascolo

High order Markov Chains representation for sequences.
This packages contains a Markov Chain model tha tcan be trained on integer sequences (which can be hashes for external labels).


Example use:
```ipython
>>> from HOMarkov import markov
>>>
>>> data = [0, 1, 2, 1, 1, 3, 4, 1, 5, 3]
>>> mc = markov.MarkovChain(6, 1)  # First order MC
>>> mc.fit(data)
>>>
>>> mc.transition_matrix
[[ 0.25  0.25  0.25  0.    0.25  0.  ]
 [ 1.    0.    0.    0.    0.    0.  ]
 [ 0.    0.    0.    1.    0.    0.  ]
 [ 1.    0.    0.    0.    0.    0.  ]
 [ 0.    0.    1.    0.    0.    0.  ]
 [ 1.    0.    0.    0.    0.    0.  ]]

>>> mc.score([1, 1, 2, 1])  # possible path
0.0625

>>> mc.score([1, 1, 2, 3])  # impossible path
0

```