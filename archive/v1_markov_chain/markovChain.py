import numpy as np
from itertools import product


DOWNS = (1, 2, 3, 4)
PLAY_TYPES = ('R', 'P', 'X', 'F', 'U', 'K', 'O')  # run, pass, PAT, field goal, punt, kickoff, other
FIELD_BUCKETS = 10  # 100-yard field divided into 10-yard buckets


class PlayMarkovChain:
    '''
    Markov Chain over football play states (down, play type, field position),
    following the state/transition design in Section 4.2 of the Smart Stadium paper.

    A "state" is a (down, play_type, field_bucket) tuple. The transition matrix
    gives P(next_state | current_state), fit from a corpus of play sequences.
    '''

    def __init__(self, downs: tuple = DOWNS, play_types: tuple = PLAY_TYPES, field_buckets: int = FIELD_BUCKETS):
        self.downs = downs
        self.play_types = play_types
        self.field_buckets = field_buckets
        self.states = list(product(downs, play_types, range(field_buckets)))
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self.transition_matrix = None  # set by fit()

    def fit(self, play_sequences: list, smoothing: float = 1e-3) -> np.ndarray:
        '''
        build the transition matrix from a corpus of play sequences

        Args:
            play_sequences: list of games, each a list of (down, play_type, field_bucket)
                             states in chronological order
            smoothing: additive (Laplace) smoothing so no transition has zero probability

        Returns:
            the fitted transition matrix (n_states x n_states), row i = P(next | state i)
        '''
        n = len(self.states)
        counts = np.full((n, n), smoothing, dtype=float)

        for sequence in play_sequences:
            for current, nxt in zip(sequence, sequence[1:]):
                i, j = self.state_to_idx[current], self.state_to_idx[nxt]
                counts[i, j] += 1

        self.transition_matrix = counts / counts.sum(axis=1, keepdims=True)
        return self.transition_matrix

    def next_state_proba(self, current_state: tuple) -> np.ndarray:
        '''P(next_state | current_state) over every (down, play_type, field_bucket)'''
        if self.transition_matrix is None:
            raise RuntimeError('call fit() before predicting')
        return self.transition_matrix[self.state_to_idx[current_state]]

    def next_play_type_proba(self, current_state: tuple, play_types: tuple = None) -> np.ndarray:
        '''
        marginalize P(next_state | current_state) down to just P(next_play_type),
        summing out down and field position -- this is the vector that gets
        conflated with a play-type classifier's output (see conflation.py)

        Args:
            current_state: (down, play_type, field_bucket) of the play just observed
            play_types: subset/order of play types to return probabilities for
                        (e.g. the classifier's class list, if it differs from self.play_types)

        Returns:
            probability vector over play_types, renormalized to sum to 1
        '''
        play_types = play_types or self.play_types
        full = self.next_state_proba(current_state)
        proba = np.zeros(len(play_types))
        for (_down, ptype, _bucket), p in zip(self.states, full):
            if ptype in play_types:
                proba[play_types.index(ptype)] += p
        return proba / proba.sum()


if __name__ == '__main__':
    pass
