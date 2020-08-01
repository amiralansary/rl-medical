import numpy as np
import copy
from collections import deque


class ReplayMemory(object):
    def __init__(self, max_size, state_shape, history_len, agents):
        self.max_size = int(max_size)
        self.state_shape = state_shape
        self.history_len = int(history_len)
        self.agents = agents

        self.state = np.zeros(
            (self.agents, self.max_size) + state_shape, dtype='uint8')
        self.action = np.zeros((self.agents, self.max_size), dtype='int32')
        self.reward = np.zeros((self.agents, self.max_size), dtype='float32')
        self.isOver = np.zeros((self.agents, self.max_size), dtype='bool')

        self._curr_pos = 0
        self._curr_size = 0
        # TODO: was maxlen = history_len - 1
        self._hist = deque(maxlen=history_len)

    def append(self, exp):
        """Append the replay memory with experience sample
        Args:
            exp (Experience): contains (state, action, reward, isOver)
        """
        # increase current memory size if it is not full yet
        if self._curr_size < self.max_size:
            self._assign(self._curr_pos, exp)
            self._curr_pos = (self._curr_pos + 1) % self.max_size
            self._curr_size += 1
        else:
            self._assign(self._curr_pos, exp)
            self._curr_pos = (self._curr_pos + 1) % self.max_size
        if np.all(exp[3]):
            self._hist.clear()
        else:
            self._hist.append(exp)

    def recent_state(self):
        """ return a list of (hist_len,) + STATE_SIZE """
        lst = list(self._hist)
        states = []
        for i in range(self.agents):
            states_temp = [np.zeros(self.state_shape,
                                    dtype='uint8')] \
                                    * (self._hist.maxlen - len(lst))
            states_temp.extend([k[0][i] for k in lst])
            states.append(states_temp)
        return np.array(states)

    def _encode_sample(self, idx):
        """ Sample an experience replay from memory with index idx
        :returns: a tuple of (state, next_state, reward, action, isOver)
                  where state is of shape STATE_SIZE + (history_length,)
        """
        idx = (self._curr_pos + idx) % self._curr_size
        k = self.history_len

        states = []
        next_states = []
        rewards = []
        actions = []
        isOver = []
        for i in range(self.agents):
            if idx + k < self._curr_size:
                states.append(self.state[i, idx: idx + k])
                next_states.append(self.state[i, idx + 1: idx + k + 1])
                isOver.append(self.isOver[i, idx: idx + k])
                rewards.append(self.reward[i, idx: idx + k])
                actions.append(self.action[i, idx: idx + k])
            else:
                end = idx + k - self._curr_size
                states.append(self._slice(self.state[i], idx, end))
                next_states.append(
                    self._slice(
                        self.state[i],
                        idx + 1,
                        end + 1))
                isOver.append(self._slice(self.isOver[i], idx, end))
                rewards.append(self._slice(self.reward[i], idx, end))
                actions.append(self._slice(self.action[i], idx, end))
        states_padded = self._pad_sample(states, isOver)
        return states_padded, actions, rewards, next_states, isOver

    def sample(self, batch_size):
        idxes = [np.random.randint(0, len(self) - 1)
                 for _ in range(batch_size)]
        states = []
        next_states = []
        rewards = []
        actions = []
        isOver = []
        for i in idxes:
            exp = self._encode_sample(i)
            states.append(exp[0])
            actions.append(exp[1])
            rewards.append(exp[2])
            next_states.append(exp[3])
            isOver.append(exp[4])
        # Only get most recent terminal state
        return (np.array(states), np.array(actions)[:, :, -1],
                np.array(rewards)[:, :, -1], np.array(next_states),
                np.array(isOver)[:, :, -1])

    # the next_state is a different episode if current_state.isOver==True
    def _pad_sample(self, states, isOver):
        for k in range(self.history_len - 1, -1, -1):
            for i in range(self.agents):
                if isOver[i][k]:
                    states[i] = copy.deepcopy(states[i])
                    states[i][:k + 1].fill(0)
            break
        return states

    def _slice(self, arr, start, end):
        s1 = arr[start:self._curr_size]
        s2 = arr[:end]
        return np.concatenate((s1, s2), axis=0)

    def __len__(self):
        return self._curr_size

    def _assign(self, pos, exp):
        for i in range(self.agents):
            self.state[i, pos] = exp[0][i]
            self.action[i, pos] = exp[1][i]
            self.reward[i, pos] = exp[2][i]
            self.isOver[i, pos] = exp[3][i]

    def __str__(self):
        return f"""Replay buffer:
         Current position / current size: {self._curr_pos}/{self._curr_size}
         states {[hash(str(self.state[0, i]))
                    for i in range(len(self.state[0]))]}
         actions {self.action}
         rewards {self.reward}
         isOver {self.isOver}"""
