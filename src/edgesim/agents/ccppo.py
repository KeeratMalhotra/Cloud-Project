"""CCPPO for EdgePlus (Sec. III-E, Eq. 8): one actor per edge site, one centralized critic, masked softmax.

The actor for the task's home site maps the local observation (task + one row per server) to logits over all
servers. Advisory biases are added before masking (Eq. 9-10). The critic sees the local observation plus the
global site features (centralized training, decentralized execution). Variants used by the baselines and
ablations: a single shared actor, an ensemble of actor heads (ensemble PPO), and a Lagrangian miss penalty
without masks (ccppo_lagr).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from edgesim.agents.advisory import DEFAULTS as ADVISORY_DEFAULTS
from edgesim.agents.advisory import Advisory
from edgesim.agents.features import SERVER_DIM, TASK_DIM, global_dim
from edgesim.agents.rl import REWARD_DEFAULTS, EpisodeSource, VecEnvs, explained_variance, lr_schedule
from edgesim.sim.config import sim_config
from edgesim.sim.env import EdgeEnv
from edgesim.sim.metrics import summarize
from edgesim.sim.world import World

NEG = -1e9

PPO_DEFAULTS = {
    "hidden": 128, "lr": 3e-4, "gamma": 0.99, "lam": 0.95, "clip": 0.2, "epochs": 4, "minibatch": 1024,
    "ent_coef": 0.01, "vf_coef": 0.5, "kl_coef": 0.0, "max_grad_norm": 0.5, "n_envs": 8,
    "steps_per_iter": 4096, "buffer_cap": 60000, "per_site_actors": True, "ensemble": 1,
    "recompute_old": True, "is_clip": 1.0, "reward_scale": 0.1, "train_horizon_s": 150.0,
    # Warm start (not in the EdgePlus paper; applied identically to every learned method): for the first
    # bc_decisions the environments follow greedy EFT and the actor imitates it; then PPO fine-tunes with an
    # imitation term whose weight decays from bc_coef to 0 over bc_decay_frac of the remaining budget.
    "bc_decisions": 0, "bc_coef": 1.0, "bc_decay_frac": 0.5,
    "lagrangian": False, "lagr_lr": 0.05, "lagr_target": [0.05, 0.10, 0.15], "lagr_max": 10.0,
}


def mlp(n_in: int, hidden: int, n_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, n_out))


class StackedMLP(nn.Module):
    """H independent two-hidden-layer MLPs with stacked weights; row b uses network head[b]."""

    def __init__(self, heads: int, n_in: int, hidden: int, n_out: int, out_gain: float = 0.01):
        super().__init__()
        dims = [(n_in, hidden), (hidden, hidden), (hidden, n_out)]
        self.w = nn.ParameterList()
        self.b = nn.ParameterList()
        for k, (a, o) in enumerate(dims):
            w = torch.empty(heads, a, o)
            for h in range(heads):
                nn.init.orthogonal_(w[h], gain=out_gain if k == 2 else 2 ** 0.5)
            self.w.append(nn.Parameter(w))
            self.b.append(nn.Parameter(torch.zeros(heads, o)))

    def forward(self, x: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
        out = x.new_empty(x.shape[0], self.w[2].shape[2])  # rows grouped by head, one matmul chain per group
        for c in torch.unique(head).tolist():
            idx = (head == c).nonzero(as_tuple=True)[0]
            h = x[idx]
            for k in range(3):
                h = torch.addmm(self.b[k][c], h, self.w[k][c])
                if k < 2:
                    h = torch.tanh(h)
            out[idx] = h
        return out


class CCPPOModel(nn.Module):
    def __init__(self, n_sites: int, n_servers: int, gdim: int, hidden: int, per_site: bool, ensemble: int):
        super().__init__()
        self.n_sites, self.n_servers, self.per_site, self.ensemble = n_sites, n_servers, per_site, ensemble
        self.in_actor = TASK_DIM + n_servers * SERVER_DIM
        self.actor = StackedMLP(n_sites if per_site else 1, self.in_actor, hidden, n_servers * ensemble)
        self.critic = mlp(gdim + self.in_actor, hidden, 1)

    def head_logits(self, x: torch.Tensor, site: torch.Tensor) -> torch.Tensor:
        """Logits per ensemble head, shape (B, ensemble, S)."""
        head = site if self.per_site else torch.zeros_like(site)
        return self.actor(x, head).view(x.shape[0], self.ensemble, self.n_servers)

    def value(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.critic(torch.cat([g, x], dim=1)).squeeze(1)


def masked_logits(logits: torch.Tensor, bias: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (logits + bias).masked_fill(~mask, NEG)


class CCPPOAgent:
    """Inference wrapper used for evaluation: picks the most likely feasible server (or samples)."""

    def __init__(self, model: CCPPOModel, advisory_params: dict | None, n_sites: int, n_servers: int,
                 greedy: bool = True, seed: int = 0):
        self.model = model.eval()
        self.advisory = Advisory(n_sites, n_servers, advisory_params) if advisory_params is not None else None
        self.greedy = greedy
        self.gen = torch.Generator().manual_seed(seed)
        self._episode = None
        self.cache = {}

    def __call__(self, env: EdgeEnv) -> int:
        from edgesim.agents.features import observe

        key = (id(env), env.episode_id)
        if key != self._episode:
            self._episode = key
            self.cache = {}
            if self.advisory is not None:
                self.advisory.reset()
        ob = observe(env, self.cache)
        x = torch.from_numpy(np.concatenate([ob["task"], ob["servers"].ravel()]))[None]
        bias = self.advisory.bias(env, ob["util"]) if self.advisory is not None else np.zeros(env.S, np.float32)
        with torch.inference_mode():
            lg = self.model.head_logits(x, torch.tensor([ob["site"]])).mean(1)
            lg = masked_logits(lg, torch.from_numpy(bias)[None], torch.from_numpy(ob["mask"])[None])[0]
            if self.greedy:
                return int(torch.argmax(lg))
            return int(torch.multinomial(torch.softmax(lg, 0), 1, generator=self.gen))


class CCPPOTrainer:
    def __init__(self, world: World, run_dir: Path, system: str, sim_overrides: dict | None, dataset: str,
                 seed: int, ppo: dict | None = None, reward: dict | None = None, advisory: dict | None = None,
                 loads: dict | None = None, pso: dict | None = None):
        self.p = {**PPO_DEFAULTS, **(ppo or {})}
        self.reward_cfg = {**REWARD_DEFAULTS, **(reward or {})}
        # advisory: None disables the layer; otherwise overrides of advisory.DEFAULTS (completed here so PSO and
        # checkpoints always see the full parameter set)
        self.adv_params = None if advisory is None else {**ADVISORY_DEFAULTS, **advisory}
        self.pso_cfg = pso  # None disables PSO; e.g. {"gate": "paper", "at": [0.5, 0.85]}
        self.dataset, self.seed = dataset, seed
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(seed)
        self.rng = np.random.default_rng([seed, 3])
        self.gen = torch.Generator().manual_seed(seed)
        cfg = sim_config(system, sim_overrides)
        self.cfg = cfg
        self.world = world
        C, S = world.n_sites, world.n_servers
        self.C, self.S = C, S
        self.gdim = global_dim(C)
        self.model = CCPPOModel(C, S, self.gdim, self.p["hidden"], self.p["per_site_actors"], self.p["ensemble"])
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.p["lr"], eps=1e-5)
        self.lagrange = np.zeros(3) if self.p["lagrangian"] else None
        self.state = {"decisions": 0, "updates": 0, "episodes": 0, "wall_s": 0.0, "pso_rounds": 0}
        self.load_checkpoint()
        envs = [EdgeEnv(world, cfg) for _ in range(self.p["n_envs"])]
        src = EpisodeSource(dataset, "train", loads or {50_000: 1, 100_000: 1, 150_000: 2},
                            seed * 1000 + self.state["episodes"], horizon_s=self.p["train_horizon_s"])
        self.vec = VecEnvs(envs, src, self.p["buffer_cap"], self.gdim)
        self.reward_cfg["scale"] = self.p["reward_scale"]
        self._make_advisors()

    def _make_advisors(self) -> None:
        n = len(self.vec.envs)
        self.advisors = [Advisory(self.C, self.S, self.adv_params) for _ in range(n)] \
            if self.adv_params is not None else None
        self.recent: list[dict] = []
        self.miss_counts = np.zeros(3)
        self.slice_counts = np.zeros(3)

    # ---------------------------------------------------------------- persistence
    def ckpt_path(self) -> Path:
        return self.run_dir / "checkpoint.pt"

    def save_checkpoint(self) -> None:
        tmp = self.ckpt_path().with_suffix(".tmp")
        torch.save({"model": self.model.state_dict(), "opt": self.opt.state_dict(), "state": self.state,
                    "lagrange": self.lagrange, "ppo": self.p, "reward": self.reward_cfg,
                    "advisory": self.adv_params}, tmp)
        tmp.replace(self.ckpt_path())

    def load_checkpoint(self) -> None:
        if self.ckpt_path().exists():
            ck = torch.load(self.ckpt_path(), weights_only=False)
            self.model.load_state_dict(ck["model"])
            self.opt.load_state_dict(ck["opt"])
            self.state = {"pso_rounds": 0, **ck["state"]}
            self.lagrange = ck.get("lagrange")
            if self.adv_params is not None and ck.get("advisory") is not None:
                self.adv_params = ck["advisory"]  # gains possibly tuned by PSO

    def log(self, row: dict) -> None:
        with open(self.run_dir / "log.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\n")

    # ---------------------------------------------------------------- rollout
    def _act(self, obs: list[dict]):
        envs = self.vec.envs
        K = len(envs)
        x = torch.from_numpy(np.stack([np.concatenate([o["task"], o["servers"].ravel()]) for o in obs]))
        g = torch.from_numpy(np.stack([o["global"] for o in obs]))
        mask = torch.from_numpy(np.stack([o["mask"] for o in obs]))
        site = torch.tensor([o["site"] for o in obs])
        if self.advisors is not None:
            bias = np.stack([self.advisors[k].bias(envs[k], obs[k]["util"]) for k in range(K)]).astype(np.float32)
        else:
            bias = np.zeros((K, self.S), np.float32)
        # expert = greedy EFT: the feasible server with the lowest predicted latency (feature column 17)
        experts = np.array([int(np.argmax(o["servers"][:, 17])) for o in obs])
        with torch.inference_mode():
            lg = self.model.head_logits(x, site)
            head = torch.from_numpy(self.rng.integers(self.p["ensemble"], size=K)) if self.p["ensemble"] > 1 \
                else torch.zeros(K, dtype=torch.long)
            lg = lg[torch.arange(K), head]
            lg = masked_logits(lg, torch.from_numpy(bias), mask)
            dist = torch.distributions.Categorical(logits=lg)
            a = torch.from_numpy(experts) if self.in_bc_phase() else dist.sample()
            logp = dist.log_prob(a)
            v = self.model.value(g, x)
        return a.numpy(), bias, logp.numpy(), v.numpy(), experts

    def in_bc_phase(self) -> bool:
        return self.state["decisions"] < self.p["bc_decisions"]

    def bc_weight(self, budget: int) -> float:
        p = self.p
        if p["bc_decisions"] <= 0:
            return 0.0
        span = max(p["bc_decay_frac"] * (budget - p["bc_decisions"]), 1.0)
        return p["bc_coef"] * max(0.0, 1.0 - (self.state["decisions"] - p["bc_decisions"]) / span)

    def _episode_end(self, k: int, env: EdgeEnv, meta: dict) -> None:
        kp = summarize(env)
        self.state["episodes"] += 1
        row = {"n_tasks": meta["n_tasks"], "success": kp["success"], "success_URLLC": kp["success_URLLC"],
               "lat_p95": kp["lat_p95"], "cloud_share": kp["cloud_share"]}
        self.recent.append(row)
        self.recent = self.recent[-30:]
        for s in range(3):
            m = env.slice == s
            self.slice_counts[s] += m.sum()
            self.miss_counts[s] += (~env.on_time[m]).sum()
        if self.advisors is not None:
            self.advisors[k].reset()

    # ---------------------------------------------------------------- update
    def _gather(self):
        parts = []
        for buf in self.vec.buffers:
            slots, adv = buf.ready(self.p["gamma"], self.p["lam"])
            if slots.size:
                parts.append((buf, slots, adv))
        return parts

    def _update(self, parts, progress: float) -> dict:
        p = self.p
        cat = lambda f: np.concatenate([f(buf, s) for buf, s, _ in parts])  # noqa: E731
        x = torch.from_numpy(cat(lambda b, s: np.concatenate(
            [b.task[s], b.servers[s].reshape(len(s), -1).astype(np.float32)], axis=1)))
        g = torch.from_numpy(cat(lambda b, s: b.glob[s]))
        mask = torch.from_numpy(cat(lambda b, s: b.mask[s]))
        bias = torch.from_numpy(cat(lambda b, s: b.bias[s].astype(np.float32)))
        site = torch.from_numpy(cat(lambda b, s: b.site[s].astype(np.int64)))
        act = torch.from_numpy(cat(lambda b, s: b.action[s]))
        behav_logp = torch.from_numpy(cat(lambda b, s: b.logp[s]))
        old_v = torch.from_numpy(cat(lambda b, s: b.value[s]))
        expert = torch.from_numpy(cat(lambda b, s: b.expert[s]))
        bc_step = torch.from_numpy(cat(lambda b, s: b.bc[s]))
        adv = torch.from_numpy(np.concatenate([a for _, _, a in parts]))
        ret = adv + old_v
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = len(act)
        bc_only = self.in_bc_phase()
        bc_w = 1.0 if bc_only else self.bc_weight(self._budget)
        # PPO's ratio needs the behaviour policy's probabilities; warm-start steps were taken by the expert,
        # so they only train the critic and the imitation term.
        pg_on = (~bc_step).float()
        if p["ensemble"] > 1:  # bootstrap weights per head (Poisson(1)), as in a bootstrap ensemble
            boot = torch.from_numpy(self.rng.poisson(1.0, size=(n, p["ensemble"])).astype(np.float32))
        # Rewards arrive late, so samples come from policies a few updates old. With recompute_old the trust
        # region is centred on the current policy and stale samples get a truncated importance weight.
        if p["recompute_old"]:
            with torch.no_grad():
                old_logp = torch.empty(n)
                for st in range(0, n, 8192):
                    sl = slice(st, st + 8192)
                    old_logp[sl] = self._logp(x[sl], site[sl], bias[sl], mask[sl], act[sl],
                                              boot[sl] if p["ensemble"] > 1 else None)[0]
            is_w = torch.clamp(torch.exp(old_logp - behav_logp), max=p["is_clip"])
        else:
            old_logp = behav_logp
            is_w = torch.ones(n)
        for grp in self.opt.param_groups:
            grp["lr"] = lr_schedule(p["lr"], progress)
        stats = {"pg": 0.0, "vf": 0.0, "ent": 0.0, "kl": 0.0, "clipfrac": 0.0}
        n_mb = 0
        for _ in range(p["epochs"]):
            perm = torch.from_numpy(self.rng.permutation(n))
            for st in range(0, n, p["minibatch"]):
                idx = perm[st: st + p["minibatch"]]
                logp, ent = self._logp(x[idx], site[idx], bias[idx], mask[idx], act[idx],
                                       boot[idx] if p["ensemble"] > 1 else None)
                ent = ent.mean()
                ratio = torch.exp(logp - old_logp[idx])
                a_mb = adv_n[idx] * is_w[idx]
                w_pg = pg_on[idx]
                pg_terms = torch.min(ratio * a_mb, torch.clamp(ratio, 1 - p["clip"], 1 + p["clip"]) * a_mb)
                pg = -(pg_terms * w_pg).sum() / w_pg.sum().clamp(min=1.0)
                v = self.model.value(g[idx], x[idx])
                vf = F.mse_loss(v, ret[idx])
                approx_kl = ((old_logp[idx] - logp) * w_pg).sum() / w_pg.sum().clamp(min=1.0)
                loss = p["vf_coef"] * vf
                if bc_w > 0:
                    bc_logp, _ = self._logp(x[idx], site[idx], bias[idx], mask[idx], expert[idx],
                                            boot[idx] if p["ensemble"] > 1 else None)
                    bc_loss = -bc_logp.mean()
                    loss = loss + bc_w * bc_loss
                    stats["bc"] = stats.get("bc", 0.0) + float(bc_loss)
                if not bc_only:
                    loss = loss + pg - p["ent_coef"] * ent + p["kl_coef"] * approx_kl
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.actor.parameters(), p["max_grad_norm"])
                nn.utils.clip_grad_norm_(self.model.critic.parameters(), p["max_grad_norm"])
                self.opt.step()
                stats["pg"] += float(pg)
                stats["vf"] += float(vf)
                stats["ent"] += float(ent)
                stats["kl"] += float(approx_kl)
                stats["clipfrac"] += float(((ratio - 1).abs() > p["clip"]).float().mean())
                n_mb += 1
        for k in stats:
            stats[k] /= max(n_mb, 1)
        with torch.no_grad():
            vals = torch.cat([self.model.value(g[st: st + 8192], x[st: st + 8192]) for st in range(0, n, 8192)])
            stats["ev"] = explained_variance(vals.numpy(), ret.numpy())
        for buf, s, _ in parts:
            buf.mark_used(s)
        stats["samples"] = int(n)
        stats["is_w"] = float(is_w.mean())
        stats["bc_w"] = round(bc_w, 4)
        stats["bc_steps"] = int(bc_step.sum())
        with torch.no_grad():  # how often the policy's most likely server is the expert's choice
            lg = self.model.head_logits(x[:4096], site[:4096]).mean(1)
            top = masked_logits(lg, bias[:4096], mask[:4096]).argmax(1)
            stats["agree"] = float((top == expert[:4096]).float().mean())
        return stats

    def _logp(self, x, site, bias, mask, act, boot):
        """Log-probability of the taken actions and the policy entropy (ensemble: bootstrap-weighted heads)."""
        lg_all = self.model.head_logits(x, site)
        if boot is not None:
            lg = masked_logits(lg_all, bias[:, None], mask[:, None])
            dist = torch.distributions.Categorical(logits=lg)
            logp_all = dist.log_prob(act[:, None])
            logp = (logp_all * boot).sum(1) / boot.sum(1).clamp(min=1.0)
            return logp, dist.entropy().mean(1)
        dist = torch.distributions.Categorical(logits=masked_logits(lg_all[:, 0], bias, mask))
        return dist.log_prob(act), dist.entropy()

    # ---------------------------------------------------------------- main loop
    def train(self, budget_decisions: int, max_minutes: float, ckpt_every: int = 5) -> dict:
        t_start = time.time()
        p = self.p
        self._budget = budget_decisions
        while self.state["decisions"] < budget_decisions and (time.time() - t_start) < 60 * max_minutes:
            t0 = time.time()
            for _ in range(p["steps_per_iter"]):
                obs = self.vec.observe_all()
                bc = self.in_bc_phase()
                a, bias, logp, v, experts = self._act(obs)
                self.vec.step(a, obs, bias, logp, v, self.reward_cfg, self._episode_end, self.lagrange,
                              experts=experts, bc=bc)
            self.state["decisions"] += p["steps_per_iter"] * len(self.vec.envs)
            t1 = time.time()
            parts = self._gather()
            if not parts:
                continue
            stats = self._update(parts, self.state["decisions"] / budget_decisions)
            self.state["updates"] += 1
            if self.lagrange is not None and self.slice_counts.sum() > 0:
                miss = self.miss_counts / np.maximum(self.slice_counts, 1)
                self.lagrange = np.clip(self.lagrange + p["lagr_lr"] * (miss - np.asarray(p["lagr_target"])),
                                        0.0, p["lagr_max"])
                self.miss_counts[:] = 0
                self.slice_counts[:] = 0
            t2 = time.time()
            self.state["wall_s"] += t2 - t0
            rec = self.recent[-10:]
            row = {"update": self.state["updates"], "decisions": self.state["decisions"],
                   "episodes": self.state["episodes"], "collect_s": round(t1 - t0, 2), "update_s": round(t2 - t1, 2),
                   "dps": round(p["steps_per_iter"] * len(self.vec.envs) / max(t1 - t0, 1e-6)),
                   **{k: round(v, 4) for k, v in stats.items()},
                   "recent_success": round(float(np.mean([r["success"] for r in rec])), 4) if rec else None,
                   "recent_success_stress": round(float(np.mean([r["success"] for r in rec
                                                                 if r["n_tasks"] == 150_000])), 4)
                   if any(r["n_tasks"] == 150_000 for r in rec) else None,
                   "overflow": int(sum(b.overflow for b in self.vec.buffers)),
                   "lagrange": None if self.lagrange is None else [round(float(v), 3) for v in self.lagrange]}
            self.log(row)
            print(json.dumps(row), flush=True)
            if self.state["updates"] % ckpt_every == 0:
                self.save_checkpoint()
            self._maybe_pso(budget_decisions)
        self.save_checkpoint()
        return self.state

    def _maybe_pso(self, budget: int) -> None:
        """Run the scheduled PSO rounds over the advisory gains (frozen policy, training-day episodes)."""
        if self.pso_cfg is None or self.adv_params is None:
            return
        from edgesim.agents.pso import pso_round

        at = self.pso_cfg.get("at", [0.5, 0.85])
        r = self.state["pso_rounds"]
        if r >= len(at) or self.state["decisions"] < at[r] * budget:
            return
        t0 = time.time()
        self.save_checkpoint()  # workers load the frozen policy from the checkpoint
        new, rec = pso_round(self.run_dir, self.cfg, self.dataset, self.adv_params,
                             self.pso_cfg.get("gate", "paper"), seed=self.seed * 100 + r,
                             opts=self.pso_cfg.get("opts"), log=lambda m: print(m, flush=True))
        self.adv_params = new
        self._make_advisors()
        self.state["pso_rounds"] = r + 1
        rec.update({"round": r + 1, "decisions": self.state["decisions"], "wall_s": round(time.time() - t0, 1)})
        with open(self.run_dir / "pso.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps({"pso": rec}), flush=True)
        self.save_checkpoint()

    def agent(self, greedy: bool = True) -> CCPPOAgent:
        return CCPPOAgent(self.model, self.adv_params, self.C, self.S, greedy)


def load_agent(run_dir: Path, world: World, greedy: bool = True) -> CCPPOAgent:
    ck = torch.load(Path(run_dir) / "checkpoint.pt", weights_only=False)
    p = ck["ppo"]
    model = CCPPOModel(world.n_sites, world.n_servers, global_dim(world.n_sites), p["hidden"],
                       p["per_site_actors"], p["ensemble"])
    model.load_state_dict(ck["model"])
    return CCPPOAgent(model, ck["advisory"], world.n_sites, world.n_servers, greedy)
