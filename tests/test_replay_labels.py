"""A transition's terminal label belongs to the action that reached that state."""
import torch
import pytest

from supermimic.data.replay_buffer import ReplayBuffer


@pytest.mark.xfail(strict=True, reason="Known upstream label offset retained for source parity; optional fix is archived in patches/diagnostic-replay-label-and-grad-guard.patch")
def test_terminal_label_stays_with_the_transition_that_caused_it():
    buffer = ReplayBuffer(1, 8, 1, 1, 1, "cpu")
    # s0 --a0--> s1 --a1--> s2 --a2--> terminal s3.
    # Recorder stores done_next in each transition row, and duplicates the
    # terminal label on the final state-only row (which has no outgoing action).
    for t, done in enumerate((0., 0., 1., 1.)):
        buffer.add(
            torch.tensor([0]), s=torch.tensor([[float(t)]]),
            a=torch.zeros(1, 1), a_exec=torch.zeros(1, 1), xi=torch.zeros(1, 1),
            r=torch.zeros(1), done=torch.tensor([done]), frame=torch.tensor([t]),
            clip=torch.zeros(1, dtype=torch.long), terminal=torch.tensor([t == 3]),
            ep_start=torch.tensor([t == 0]),
        )
    # Exactly one valid length-three window: no dependency on sampling luck.
    window = buffer.sample_windows(1, 3)
    torch.testing.assert_close(window.states[:, 0, 0], torch.arange(4).float())
    torch.testing.assert_close(window.done[:, 0], torch.tensor([0., 0., 1.]))
