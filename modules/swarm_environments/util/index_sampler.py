import numpy as np


class IndexSampler:

    def __init__(self, size: int, random_state: np.random.RandomState):
        self._size = size
        self._indices = np.arange(size)
        self._random_state = random_state
        self._reset()

    def next(self) -> int:
        """
        Returns the next index in the sequence. If the end of the sequence is reached, the sequence is shuffled and
        the position in the index array is reset to 0.
        Returns:

        """
        if self._position == self._size:
            self._reset()
        index = self._indices[self._position]
        self._position += 1
        return index

    def _reset(self):
        self._position = 0
        self._random_state.shuffle(self._indices)

    def __len__(self):
        return self._size
