"""Policy-value network: a small residual CNN with two heads.

It is fully convolutional (the value head uses global average pooling), so the
same weights can be run on any board size.
"""

import flax.linen as nn
import jax.numpy as jnp


class ResBlock(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):
        h = nn.Conv(self.channels, (3, 3))(nn.relu(nn.LayerNorm()(x)))
        h = nn.Conv(self.channels, (3, 3))(nn.relu(nn.LayerNorm()(h)))
        return x + h


class PolicyValueNet(nn.Module):
    channels: int = 64
    blocks: int = 6

    @nn.compact
    def __call__(self, obs):
        """obs [B, N, N, 3] -> (policy logits [B, N*N], value [B] in [-1, 1]).

        The value is from the view of the player to move.
        """
        h = nn.Conv(self.channels, (3, 3))(obs)
        for _ in range(self.blocks):
            h = ResBlock(self.channels)(h)
        h = nn.relu(nn.LayerNorm()(h))

        logits = nn.Conv(1, (1, 1))(h).reshape(h.shape[0], -1)

        v = nn.relu(nn.Dense(self.channels)(h.mean(axis=(1, 2))))
        value = jnp.tanh(nn.Dense(1)(v))[:, 0]
        return logits, value
