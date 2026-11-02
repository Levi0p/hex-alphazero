"""AlphaZero training loop: self-play -> replay buffer -> gradient steps, repeat.

    python -m hexaz.train --size 3 --iters 30
"""

import argparse
import csv
import json
import time
from functools import partial
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from hexaz import evaluate
from hexaz import hex as H
from hexaz.net import PolicyValueNet
from hexaz.selfplay import Labeller, ReplayBuffer, play_moves


def loss_fn(params, apply_fn, obs, pi, z):
    logits, v = apply_fn(params, obs)
    legal = (obs[..., 0] + obs[..., 1]).reshape(obs.shape[0], -1) == 0
    logp = jax.nn.log_softmax(jnp.where(legal, logits, -1e9))
    policy_loss = -(pi * logp).sum(-1).mean()  # cross-entropy to MCTS visits
    value_loss = ((v - z) ** 2).mean()  # regress to the game result
    return policy_loss + value_loss, (policy_loss, value_loss)


@partial(jax.jit, static_argnames=("apply_fn", "tx"))
def train_step(params, opt_state, apply_fn, tx, obs, pi, z):
    grads, aux = jax.grad(loss_fn, has_aux=True)(params, apply_fn, obs, pi, z)
    updates, opt_state = tx.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, aux


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=3)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--games", type=int, default=128, help="parallel self-play games")
    p.add_argument("--sims", type=int, default=32, help="MCTS simulations per move")
    p.add_argument("--moves", type=int, default=16, help="moves per self-play chunk")
    p.add_argument("--temp-moves", type=int, default=None, help="default: board size")
    p.add_argument("--random-open", type=float, default=0.5, help="share of games with a random first move")
    p.add_argument("--buffer", type=int, default=100_000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--steps", type=int, default=50, help="gradient steps per iteration")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--resume", action="store_true",
                   help="continue the run in --out from its last weights for --iters more iterations")
    args = p.parse_args()

    n, a = args.size, args.size**2
    temp_moves = args.temp_moves if args.temp_moves is not None else n
    out = Path(args.out or f"runs/hex{n}_seed{args.seed}")
    out.mkdir(parents=True, exist_ok=True)

    # When resuming, carry on counting iterations and seconds from the old log.
    first_it, secs_before = 1, 0.0
    if args.resume:
        old = list(csv.DictReader(open(out / "metrics.csv")))
        first_it, secs_before = int(old[-1]["iter"]) + 1, float(old[-1]["secs"])

    net = PolicyValueNet(args.channels, args.blocks)
    key = jax.random.PRNGKey(args.seed + first_it - 1)  # same as before for new runs, fresh when resuming
    key, k = jax.random.split(key)
    params = net.init(k, jnp.zeros((1, n, n, 3)))
    if args.resume:
        params = flax.serialization.from_bytes(params, (out / "params.msgpack").read_bytes())
    # Cosine decay: big steps early, small careful steps once play is nearly right.
    schedule = optax.cosine_decay_schedule(args.lr, args.iters * args.steps, alpha=0.05)
    tx = optax.adamw(schedule, weight_decay=1e-4)
    opt_state = tx.init(params)
    rng = np.random.default_rng(args.seed)

    states = jax.vmap(lambda _: H.init(n))(jnp.arange(args.games))
    labeller = Labeller(args.games)
    buffer = ReplayBuffer(args.buffer, (n, n, 3), a)
    config_name = f"config_resume{first_it}.json" if args.resume else "config.json"
    (out / config_name).write_text(json.dumps(vars(args), indent=2))
    log = csv.writer(open(out / "metrics.csv", "a" if args.resume else "w", buffering=1))
    if not args.resume:
        log.writerow(["iter", "positions", "policy_loss", "value_loss", "policy_acc", "value_acc", "secs"])
    start = time.time() - secs_before
    last_it = first_it + args.iters - 1

    for it in range(first_it, last_it + 1):
        key, k = jax.random.split(key)
        states, (obs, pi, ended, rand) = play_moves(
            net.apply, params, states, k, args.moves, args.sims, temp_moves, args.random_open
        )
        done = labeller.add(*jax.device_get((obs, pi, ended, rand)))
        if done is not None:
            buffer.add(*done)
        if buffer.size < args.batch:
            continue

        for _ in range(args.steps):
            params, opt_state, (pl, vl) = train_step(
                params, opt_state, net.apply, tx, *buffer.sample(rng, args.batch)
            )

        row = [it, buffer.size, f"{float(pl):.4f}", f"{float(vl):.4f}", "", "", f"{time.time() - start:.0f}"]
        msg = f"iter {it:4d}  buffer {buffer.size:7d}  policy {float(pl):.3f}  value {float(vl):.3f}"
        if n <= 4 and (it % args.eval_every == 0 or it == last_it):
            acc = evaluate.solver_accuracy(net.apply, params, n)
            row[4:6] = [f"{acc['policy_acc']:.4f}", f"{acc['value_acc']:.4f}"]
            msg += f"  | perfect-move {acc['policy_acc']:.1%}  value-sign {acc['value_acc']:.1%}"
        print(msg + f"  ({time.time() - start:.0f}s)", flush=True)
        log.writerow(row)
        blob = flax.serialization.to_bytes(params)
        (out / "params.msgpack").write_bytes(blob)
        if it % args.eval_every == 0:
            # Snapshots let us measure strength over training time afterwards.
            (out / "ckpt").mkdir(exist_ok=True)
            (out / "ckpt" / f"{it:05d}.msgpack").write_bytes(blob)


if __name__ == "__main__":
    main()
