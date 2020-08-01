from ..expreplay import ReplayMemory


def test_instantiate_expreplay():
    replay = ReplayMemory(max_size=10,
                          state_shape=(3, 3),
                          history_len=4,
                          agents=1)
    assert replay._curr_pos == 0
    assert replay._curr_size == 0
    assert not replay.state.any()
    assert not replay.action.any()
    assert not replay.reward.any()
    assert not replay.isOver.any()
    assert replay.state.shape == (1, 10, 3, 3)
    assert replay.action.shape == (1, 10)
    assert replay.reward.shape == (1, 10)
    assert replay.isOver.shape == (1, 10)
    assert len(replay._hist) == 0


def test_slices():
    replay = ReplayMemory(max_size=20,
                          state_shape=(3, 3),
                          history_len=4,
                          agents=2)
    replay.isOver[1, 9] = True
    replay._curr_size = 10
    replay._slice(replay.isOver[1], 7, 1)
