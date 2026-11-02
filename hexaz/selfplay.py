"""Self-play: B games run side by side, each move chosen by MCTS.

Finished games are restarted right away so the batch stays full. Each move we
record the position and the MCTS visit distribution (the policy target). The
value target (did the player to move here win?) is only known when the game
ends, so `Labeller` keeps unfinished games on the host until then.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from hexaz import hex as H
from hexaz import mcts


@partial(jax.jit, static_argnames=("apply_fn", "num_moves", "num_simulations", "temp_moves", "random_open"))
def play_moves(apply_fn, params, states, key, num_moves, num_simulations, temp_moves, random_open=0.0):
    """Play `num_moves` moves in every game. Returns new states and, per move,
    (obs [T, B, N, N, 3], policy [T, B, A], game_ended [T, B], random_move [T, B])."""
    n = int(round(states.board.shape[-1] ** 0.5))
    fresh = H.init(n)

    def eval_fn(obs):
        return apply_fn(params, obs)

    def one_move(carry, key):
        states = carry
        k_search, k_act, k_open, k_cell = jax.random.split(key, 4)
        out = mcts.search(eval_fn, states, k_search, num_simulations)
        policy = out.visit_counts / out.visit_counts.sum(-1, keepdims=True)

        # Explore early in the game (sample from visits), then play the best move.
        move_number = jnp.abs(states.board).sum(-1)
        sampled = jax.random.categorical(k_act, jnp.log(policy))  # log(0) = -inf: never picks an unvisited cell
        action = jnp.where(move_number < temp_moves, sampled, policy.argmax(-1))

        # Opening coverage: in a `random_open` share of games the first move is
        # uniformly random, so the agent also learns positions after openings it
        # would never choose itself. (Labeller drops the position where the random
        # move was played, since its result says nothing about good play there.)
        uniform = jax.random.categorical(k_cell, jnp.zeros_like(policy))
        use_random = (move_number == 0) & (jax.random.uniform(k_open, move_number.shape) < random_open)
        action = jnp.where(use_random, uniform, action)

        obs = jax.vmap(H.observation)(states)
        states, _ = jax.vmap(H.step)(states, action)
        ended = states.done
        states = jax.tree.map(
            lambda s, f: jnp.where(ended.reshape((-1,) + (1,) * (s.ndim - 1)), f, s), states, fresh
        )
        return states, (obs, policy, ended, use_random)

    return jax.lax.scan(one_move, states, jax.random.split(key, num_moves))


class Labeller:
    """Holds each game's positions until it ends, then labels them with the result."""

    def __init__(self, batch: int):
        self.pending = [[] for _ in range(batch)]

    def add(self, obs, policy, ended, random_move):
        """Takes play_moves output (numpy). Returns finished (obs, policy, z) arrays.

        A position where a random move was played is not used for training: its
        value label would reflect the random move, not good play from there.
        """
        out_obs, out_pi, out_z = [], [], []
        for t in range(ended.shape[0]):
            for g in range(ended.shape[1]):
                self.pending[g].append((obs[t, g], policy[t, g], random_move[t, g]))
                if ended[t, g]:
                    # The last mover won. Players alternate, so going backwards
                    # the result flips sign at every move.
                    k = len(self.pending[g])
                    for i, (o, p, skip) in enumerate(self.pending[g]):
                        if skip:
                            continue
                        out_obs.append(o)
                        out_pi.append(p)
                        out_z.append(1.0 if (k - 1 - i) % 2 == 0 else -1.0)
                    self.pending[g] = []
        if not out_obs:
            return None
        return np.stack(out_obs), np.stack(out_pi), np.array(out_z, np.float32)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape, num_actions):
        self.obs = np.zeros((capacity, *obs_shape), np.float32)
        self.pi = np.zeros((capacity, num_actions), np.float32)
        self.z = np.zeros(capacity, np.float32)
        self.capacity, self.size, self.pos = capacity, 0, 0

    def add(self, obs, pi, z):
        for o, p, v in zip(obs, pi, z):
            self.obs[self.pos], self.pi[self.pos], self.z[self.pos] = o, p, v
            self.pos = (self.pos + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, rng: np.random.Generator, batch: int):
        idx = rng.integers(0, self.size, batch)
        obs, pi, z = self.obs[idx].copy(), self.pi[idx].copy(), self.z[idx]
        # Hex looks the same rotated 180 degrees (both goal edges swap with
        # each other), so rotate half the batch for free extra data.
        flip = rng.random(batch) < 0.5
        obs[flip] = obs[flip, ::-1, ::-1]
        pi[flip] = pi[flip, ::-1]
        return obs, pi, z
