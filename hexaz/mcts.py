"""Batched AlphaZero MCTS (PUCT) in JAX.

One search tree per game in the batch. Every simulation does, for all trees at
once:
  1. select: walk down from the root picking argmax(Q + U) until we reach an
     action that has not been tried yet (or a finished game),
  2. expand: play that action, ask the network for (prior, value) of the new
     position, and store it as a new node,
  3. backup: add the value to every node on the path, flipping its sign at
     each level because the players alternate.

Values are always stored from the view of the player to move at that node, so
the parent sees a child's value with a minus sign.
"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from hexaz import hex as H

# (obs [B, N, N, 3]) -> (logits [B, A], value [B])
EvalFn = Callable[[jax.Array], tuple[jax.Array, jax.Array]]


class Tree(NamedTuple):
    # Leading axis is the node index; vmap adds a batch axis in front of it.
    states: H.State  # position at each node
    visits: jax.Array  # int32 [M]
    value_sum: jax.Array  # float32 [M], view of player to move at the node
    prior: jax.Array  # float32 [M, A]
    children: jax.Array  # int32 [M, A], -1 = not expanded yet
    parent: jax.Array  # int32 [M], -1 for the root


class SearchOutput(NamedTuple):
    visit_counts: jax.Array  # [B, A] root child visits, the policy target
    root_value: jax.Array  # [B] mean value at the root after search


def _masked_softmax(logits, legal):
    logits = jnp.where(legal, logits, -jnp.inf)
    return jax.nn.softmax(logits, axis=-1)


def _init_tree(root: H.State, prior, value, num_nodes) -> Tree:
    a = prior.shape[-1]
    states = jax.tree.map(lambda x: jnp.zeros((num_nodes,) + x.shape, x.dtype).at[0].set(x), root)
    return Tree(
        states=states,
        visits=jnp.zeros(num_nodes, jnp.int32).at[0].set(1),
        value_sum=jnp.zeros(num_nodes, jnp.float32).at[0].set(value),
        prior=jnp.zeros((num_nodes, a), jnp.float32).at[0].set(prior),
        children=jnp.full((num_nodes, a), -1, jnp.int32),
        parent=jnp.full(num_nodes, -1, jnp.int32),
    )


def _select(tree: Tree, c_puct: float):
    """Returns (node, action, child): child is -1 if (node, action) is new,
    otherwise it is an already-expanded finished game we walked into."""

    def scores(node):
        child = tree.children[node]
        expanded = child >= 0
        cv = jnp.where(expanded, tree.visits[child], 0)
        q = jnp.where(expanded, -tree.value_sum[child] / jnp.maximum(cv, 1), 0.0)
        u = c_puct * tree.prior[node] * jnp.sqrt(tree.visits[node].astype(jnp.float32)) / (1 + cv)
        legal = tree.states.board[node] == 0
        return jnp.where(legal, q + u, -jnp.inf)

    def body(carry):
        node, _, _, _ = carry
        a = jnp.argmax(scores(node))
        child = tree.children[node, a]
        go_down = (child >= 0) & ~tree.states.done[child]
        return jnp.where(go_down, child, node), a, child, go_down

    node, a, child, _ = jax.lax.while_loop(
        lambda c: c[3], body, (jnp.int32(0), jnp.int32(0), jnp.int32(-1), jnp.array(True))
    )
    return node, a, child


def _backup(tree: Tree, leaf, value) -> Tree:
    def body(carry):
        tree, node, v = carry
        tree = tree._replace(
            visits=tree.visits.at[node].add(1),
            value_sum=tree.value_sum.at[node].add(v),
        )
        return tree, tree.parent[node], -v

    tree, _, _ = jax.lax.while_loop(lambda c: c[1] >= 0, body, (tree, leaf, value))
    return tree


def _add_node(tree: Tree, idx, node, action, state, prior):
    return tree._replace(
        states=jax.tree.map(lambda buf, x: buf.at[idx].set(x), tree.states, state),
        prior=tree.prior.at[idx].set(prior),
        children=tree.children.at[node, action].set(idx),
        parent=tree.parent.at[idx].set(node),
    )


def search(
    eval_fn: EvalFn,
    root: H.State,
    key: jax.Array,
    num_simulations: int,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.3,
    noise_fraction: float = 0.25,
) -> SearchOutput:
    """Run MCTS from a batch of root states (none of them finished)."""
    legal = jax.vmap(H.legal_mask)(root)
    logits, value = eval_fn(jax.vmap(H.observation)(root))
    prior = _masked_softmax(logits, legal)

    # Dirichlet noise on the root prior so self-play keeps trying new openings.
    if noise_fraction > 0:
        noise = jax.random.dirichlet(key, jnp.full(prior.shape, dirichlet_alpha))
        noise = jnp.where(legal, noise, 0.0)
        noise = noise / noise.sum(-1, keepdims=True)
        prior = (1 - noise_fraction) * prior + noise_fraction * noise

    tree = jax.vmap(_init_tree, in_axes=(0, 0, 0, None))(root, prior, value, num_simulations + 1)

    def simulate(i, tree):
        node, action, child = jax.vmap(_select, in_axes=(0, None))(tree, c_puct)
        parent_state = jax.tree.map(lambda x: jax.vmap(lambda xs, n: xs[n])(x, node), tree.states)
        new_state, _ = jax.vmap(H.step)(parent_state, action)
        logits, value = eval_fn(jax.vmap(H.observation)(new_state))
        new_prior = _masked_softmax(logits, jax.vmap(H.legal_mask)(new_state))
        # A finished game is a loss for the player to move there.
        value = jnp.where(new_state.done, -1.0, value)

        revisit = child >= 0  # walked into an already-stored finished game
        idx = jnp.int32(i + 1)
        added = jax.vmap(_add_node, in_axes=(0, None, 0, 0, 0, 0))(
            tree, idx, node, action, new_state, new_prior
        )
        tree = jax.tree.map(
            lambda new, old: jnp.where(revisit.reshape((-1,) + (1,) * (new.ndim - 1)), old, new),
            added,
            tree,
        )
        leaf = jnp.where(revisit, child, idx)
        value = jnp.where(revisit, -1.0, value)
        return jax.vmap(_backup)(tree, leaf, value)

    tree = jax.lax.fori_loop(0, num_simulations, simulate, tree)

    root_children = tree.children[:, 0]
    counts = jnp.where(
        root_children >= 0, jnp.take_along_axis(tree.visits, jnp.maximum(root_children, 0), 1), 0
    )
    return SearchOutput(
        visit_counts=counts.astype(jnp.float32),
        root_value=tree.value_sum[:, 0] / tree.visits[:, 0],
    )
