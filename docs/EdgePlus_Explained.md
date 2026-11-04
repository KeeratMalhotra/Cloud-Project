# EdgePlus Explained (from zero)

A casual, beginner-friendly breakdown of every complex thing in our base paper.

**Papers in this folder:**
- Base paper: [EdgePlus: A Multiagent RL Framework for Dynamic Task Allocation in 6G Edge Computing](./EdgePlus_A_Multiagent_Reinforcement_Learning_Framework_for_Dynamic_Task_Allocation_in_6G_Edge_Computing.pdf) (Taher et al., IEEE IoT Journal, April 2026)
- Background review: [Transformative impact of edge computing with 6G network using cloud computing](./s13677-026-00857-x_261103123317.pdf) (Chaurasia et al., Journal of Cloud Computing, 2026)

Equation numbers below (Eq. 1, Eq. 30, ...) refer to the EdgePlus paper.

---

## Contents

1. [What's actually going on (the story)](#1-whats-actually-going-on-the-story)
2. [Where are the servers?](#2-where-are-the-servers)
3. [What does "6G" mean in this paper?](#3-what-does-6g-mean-in-this-paper)
4. [How long does a task take? (latency)](#4-how-long-does-a-task-take-latency)
5. [Masks (the most important idea)](#5-masks-the-most-important-idea-in-the-whole-paper)
6. [RL and PPO](#6-rl-and-ppo)
7. [The advisory layer: Nash, PID, Leader](#7-the-advisory-layer-the-nudges)
8. [PSO](#8-pso-particle-swarm-optimization)
9. [Blockchain-lite](#9-blockchain-lite)
10. [CBS–D* Lite (how mobile servers drive)](#10-cbsd-lite-how-mobile-servers-drive)
11. [The data](#11-the-data)
12. [The competitors (baselines)](#12-the-competitors-baselines)
13. [Metrics cheat sheet](#13-metrics-cheat-sheet)
14. [One tick of the simulation](#14-one-tick-of-the-simulation-start-to-finish)
15. [What actually matters for us](#15-what-actually-matters-for-us)

---

## 1. What's actually going on (the story)

Picture a 10 km × 10 km city. Thousands of air-quality sensors sit on lampposts measuring pollution (PM2.5, NO2, ozone...). Every reading becomes a small job, a **task**, that some computer has to process before a deadline. Example: "PM2.5 just spiked near a school, analyze it and raise an alert within 10 seconds."

The whole paper is about one question, asked over and over:

> **A task just showed up. Which computer should run it?**

Pick well and the task finishes on time. Pick badly (server too busy, too far away) and you miss the deadline.

That's literally it. PPO, PID, Nash, blockchain: all of it is machinery to make that one choice better and safer.

---

## 2. Where are the servers?

The city gets split into **10 clusters** (think neighborhoods) using weighted K-means. K-means groups points by location, and "weighted" means areas with more sensors or worse pollution pull the cluster centers toward them.

There are three kinds of computers:

- **Stationary edge servers.** Fixed boxes inside a cluster, like a small server rack at a cell tower. Close and quick to reach, but limited capacity.
- **Mobile edge servers.** Servers on wheels (vans), up to 120 km/h. They can drive to a busy neighborhood. The catch is that driving takes time, and that time counts against the deadline.
- **Cloud.** A big data center far away (they call it cluster "0"). Tons of power, but it's farther, and the system treats using it as a cost.

Which clusters count as "neighbors" comes from Delaunay triangulation, a standard way to connect nearby points into triangles. If two cluster centers share a triangle edge, they're neighbors.

Fig. 2 in the paper is this map: red × = cluster centers, blue squares = stationary, green squares = mobile, black circles = cloud.

**Routing order** (the preference list):
1. stationary server in my cluster
2. mobile server in my cluster
3. mobile server in a neighbor cluster
4. cloud

---

## 3. What does "6G" mean in this paper?

Real talk: **they don't simulate actual 6G radio.** No antennas, no terahertz physics. "6G" shows up as three things:

**a) Fast links, via a formula.** Eq. 30 is basically Shannon's formula: `rate = bandwidth × log2(1 + SNR)`, multiplied by some efficiency factors (contention, backhaul load). Plug in big 6G-style numbers and that's your "6G network."

**b) Network slicing.** You cut one physical network into virtual lanes, each reserved for one type of traffic. Picture a highway with an ambulance lane, a bus lane, and normal lanes. There are three slices:

| Slice | What it's for | Deadline here |
|---|---|---|
| **URLLC** (ultra-reliable low latency) | urgent stuff | 5–15 s |
| **eMBB** (enhanced mobile broadband) | bigger data, normal urgency | 30–120 s |
| **mMTC** (massive machine-type) | tons of tiny chill messages | 120–600 s |

Each cluster has a budget per slice (radio time + compute time). If cluster 3's URLLC lane is full, you can't squeeze another URLLC task in there.

**c) Tight deadlines + lots of devices.** That's the rest of the 6G vibe.

Side note: these deadlines are in *seconds*, while 6G marketing talks about sub-millisecond latency. That's pretty generous, and it's something we could point out.

So when *we* "simulate 6G," we do the same thing: write the formulas with 6G parameters. That's totally normal for papers like this.

---

## 4. How long does a task take? (latency)

Eq. 1: **total time = transmit + move + queue + process**

- **Transmit:** sending the data. `data size ÷ link rate`
- **Move:** only for mobile servers. `distance ÷ speed`. Zero for the others.
- **Queue:** waiting in line. Eq. 31 is Kingman's formula, and the key part is `ρ/(1−ρ)`, where ρ = how busy the server is. At 50% busy that factor is 1. At 90% it's 9. At 99% it's 99. Wait time **explodes** as a server gets close to full, and that's why overloaded servers kill you.
- **Process:** `CPU cycles needed ÷ CPU speed`

Energy (Eq. 2) works the same way: sending energy + driving energy + compute energy.

**Mini example.** A URLLC task from cluster 3 with a 10 s deadline:

| Option | send | drive | queue | process | total | OK? |
|---|---|---|---|---|---|---|
| Stationary in cluster 3 | 0.1 | 0 | 6.0 | 1.0 | 7.1 s | ✅ |
| Mobile in cluster 3 | 0.1 | 4.0 | 0.5 | 1.5 | 6.1 s | ✅ |
| Mobile in cluster 4 | 0.1 | 11.0 | 0 | 1.5 | 12.6 s | ❌ |
| Cloud | 5.0 | 0 | 0 | 0.2 | 5.2 s | ✅ but costly |

Cloud looks fast here, right? The catch is that the reward charges a fee for using cloud, and the backhaul link clogs up if everyone uses it. The paper wants cloud usage under 7%.

---

## 5. Masks (the most important idea in the whole paper)

Before the agent chooses, the system builds a yes/no list for every server: *can this server legally take this task?*

It's a "no" if:
- the task would miss its deadline
- the slice budget is full
- the server is out of capacity or memory
- the mobile server is low on battery
- the server has bad reputation (URLLC tasks only, see [section 9](#9-blockchain-lite))

In the example, "mobile in cluster 4" gets masked out. The agent **can't** pick it at all.

Why this matters: without masks, an RL agent only learns "don't do dumb stuff" by getting punished thousands of times. With masks, the dumb options just aren't on the menu. Their ablation shows that removing masks means **5.5 to 10 percentage points more missed deadlines**. It's the single biggest factor in the paper.

---

## 6. RL and PPO

### RL basics

Reinforcement learning = learning by trial and error, like training a dog with treats.

- **Agent:** the decision maker (our scheduler)
- **State:** what it sees: queue lengths, server loads, link speeds, time left before the deadline, slice budget left, reputation
- **Action:** which server gets the task
- **Reward:** the score after acting. Eq. 7 gives + for on time, − for late, − for latency, − for energy, − for using cloud
- **Policy:** the agent's brain, a neural network that maps state → probability of each action

The goal is to learn a policy that collects the most reward over time.

### Logits and softmax (you'll see these everywhere)

The network outputs a raw score per server, called **logits**. **Softmax** turns those scores into probabilities: `exp(score) / sum of all exp(scores)`.

**Masked softmax** (Eq. 10) zeroes out the blocked options before dividing, so they get exactly 0%.

Example with logits: S3 = 2.0, M3 = 1.0, M4 = 3.0, Cloud = 0.0, and M4 is masked:
- exp: 7.39, 2.72, ~~20.1~~ (blocked), 1.0 → sum = 11.11
- S3 = **66.5%**, M3 = **24.5%**, Cloud = **9%**, M4 = **0%**

M4 had the highest score and still gets nothing. That's the mask doing its job.

### PPO (Proximal Policy Optimization)

PPO is one of the most popular RL algorithms (from OpenAI, 2017). The loop:
1. Let the agent act for a while and record what happened.
2. For each action, ask: "was this better or worse than expected?" That's called the **advantage**. If it's positive, make that action more likely. If it's negative, make it less likely.
3. Update the network, then repeat.

The "proximal" trick is **don't change the policy too much in one update.** PPO computes `new probability / old probability` and **clips** it to roughly [0.8, 1.2]. Think of a driving instructor who won't let you yank the steering wheel. Big jerky updates make RL unstable, and clipping keeps learning smooth. That's the `clip(ρ, 1−ε, 1+ε)` in Eq. 8.

The other stuff in Eq. 8:
- **Entropy bonus:** rewards staying a bit random so the agent keeps exploring instead of locking in too early.
- **KL penalty:** another "don't change too much" brake.
- **GAE:** a smarter way to compute advantage that balances short-term and long-term signals. Skip the math for now.

### Actor-critic

PPO uses two networks:
- **Actor:** picks actions (this is the policy).
- **Critic:** predicts "how good is this situation?" Advantage = what actually happened − what the critic expected. If you did better than expected, it was a good move.

### Multi-agent + centralized critic (the "CC" in CCPPO)

There's **one actor per cluster** (10 actors). Each actor only looks at its own local info when deciding. During training, though, **one shared critic sees the whole city.** This setup is called CTDE (centralized training, decentralized execution). Think of a football team: each player makes their own moves on the pitch, but in practice the coach watches the whole field and gives feedback.

---

## 7. The advisory layer (the "nudges")

Eq. 9: **final score = network's logit + Nash bonus + PID bonus + leader bonus**

The key rule is that nudges only shift scores. They **never** unblock a masked server. The paper's phrase for this is "bias, don't override."

### Nash sharing game

Quick game theory: a "game" means several players each choose a strategy to maximize their own payoff. A **Nash equilibrium** is the point where nobody can do better by changing only their own choice.

In this paper, each cluster picks a "sharing level" `s_c` (how open it is to taking work). Its payoff is:

```
u_c = −α·s·load − β·energy − γ·(s−1)²
```

Translation: taking on work while you're loaded hurts, and you don't want to drift far from "normal" (s = 1). They keep recomputing each cluster's best choice until the numbers stop changing, and that's the "Nash" point. Then `log(s_c)` gets added to the logits, so busy clusters get a smaller bonus and tasks drift toward calmer clusters.

**Honest catch:** as the paper writes it, each cluster's payoff depends only on *its own* load, not on what other clusters choose. So it isn't really a game between players. The best answer works out to `s = 1 − α·load / (2γ)` (clipped to the allowed range). It's basically a load-based discount with a fancy name. That's a fair critique we could put in our paper.

### PID controller

Think cruise control. You set 80 km/h. At 70, it presses the gas. At 90, it eases off. PID is the formula for *how hard* to press.

Here the thing being controlled is **cluster utilization** (how busy the cluster is). There's a target, say 70%, and the error is `actual − target`.

- **P (proportional):** reacts to how wrong you are right now. 20% over target means push back hard.
- **I (integral):** reacts to how *long* you've been wrong. If you've sat slightly over target for ages, this slowly builds up and fixes it.
- **D (derivative):** reacts to how *fast* things are changing. If load is shooting up, it acts early, like a brake against overshooting.

The output gets clipped to [−1, 1] and added to the logits, so overloaded clusters get pushed down and underloaded ones get pulled up. The paper is vague about the sign, so we'll just define it the sensible way.

Extra words from that section:
- **Ziegler–Nichols:** an old 1940s recipe for starting values of the three gains (Kp, Ki, Kd).
- **Overshoot (Mp):** how far you shoot past the target. **Settling time (Ts):** how long until you stay within ±2% of it.
- **FOPDT:** they model utilization as a simple "push it, wait a bit, it moves smoothly toward the new value" system.
- **Jury test:** a math check for whether a digital control loop is stable or will oscillate and blow up. In practice, numpy can compute the roots and you check they're all inside the unit circle.
- **±20% gain drift:** if the gains were 20% off (hardware aging, heat), is the loop still stable and fast? Their answer is yes: the best gains (1.90, 1.20, 0.06) settle in about 2.7 s with about 2% overshoot.

Honest take: this PID section (OBJ6) is mostly a separate control-theory experiment and doesn't really move the scheduling results. Low priority for us.

### Leader election

Within each cluster, score every server with `ω1·(1 − load) + ω2·(energy left)`. The top one becomes the leader and gets a small bonus. Basically: prefer the least busy, most charged server.

---

## 8. PSO (Particle Swarm Optimization)

The RL setup has a bunch of knobs (hyperparameters): learning rate, nudge strengths, and so on. Picking good values is painful.

PSO works like a flock of birds searching a field for food:
- Each bird = one set of knob values.
- Each bird remembers its **own best spot**, and the whole flock knows the **flock's best spot**.
- Every step, each bird moves partly toward its own best, partly toward the flock's best, plus some momentum.
- Over time the flock crowds around good spots.

Eq. 13 is that movement rule. "Constricted" means a factor that stops birds from flying off to infinity. The inertia shrinks over time, so the flock explores early and settles later.

Two more terms:
- **Bilevel:** the outer loop (PSO) picks the knobs, and the inner loop (PPO) trains with them.
- **SLA-gated:** new knobs are only accepted if success goes up AND cloud ≤ 7%, failures ≤ 5%, and stationary-server usage ≥ 80%.

"Escaping local optima" means RL can get stuck in an okay-but-not-great solution, and PSO jumps around to find better ones.

Their ablation: removing PSO only costs about +0.8 points of violations and +11.8 s of tail latency. It helps a little and is expensive. Low priority for us.

---

## 9. Blockchain-lite

### What's a blockchain, at all

It's a list of records (blocks), where each block stores the **hash** of the previous block. A hash is a fingerprint of the data: change one letter and the fingerprint completely changes. If someone edits an old record, its hash changes, the next block's "previous hash" stops matching, and the chain visibly breaks. That property is called **tamper-evident**.

Bitcoin-style chains also use mining so strangers can agree without trusting each other. That's super slow (minutes per block). EdgePlus skips mining, which is why it's "lite."

### What gets logged

Every finished task becomes an entry (Eq. 18): task id, cluster, server, slice, deadline, start/end times, on-time or not, energy used, and a crypto tag.

### Two tiers

- **Edge tier (fast):** each cluster bundles recent logs into a "partial block," hashes it (SHA3-256, then BLAKE2b; two hash functions back to back), and links it to the previous block. It's validated by **PoA (Proof of Authority)**: a small group of pre-approved trusted nodes checks and signs it. No mining, just trusted approvers. This takes about **0.39 ms**.
- **Cloud tier (slower, stronger):** every so often the cluster hashes go to the cloud, which runs **BFT (Byzantine Fault Tolerant)** consensus. BFT means a group of nodes can agree on the truth even if up to about a third of them are lying or broken. It takes about **6.9 ms**, but tasks don't wait for it.

### Why would a scheduler care about a blockchain?

This is the cool part. From the log, they compute:
- **Reputation** per server: a running average of "on time" minus a penalty for violations. A server that keeps flaking gets a low score.
- **Debt** per cluster: how far below its fair share of throughput the cluster is.

Those signals feed back three ways:
1. **State:** the agent can see reputation and debt.
2. **Reward:** picking a low-reputation server costs you (the penalty kicks in below 0.6).
3. **Masks:** for URLLC tasks, really bad servers get blocked outright ("quarantine").

Because the log is tamper-evident, a hacked server can't fake a clean record.

Their ablation: turning off ledger feedback barely changes anything (+2.9 s tail latency, +0.1 points of cloud usage). By their own results, it's the weakest piece.

### Crypto words

- **X25519:** two devices end up with the same secret key without ever sending it. Think of mixing paint colors in public and both ending up with the same secret color.
- **AES-GCM:** encrypts the data so nobody can read it, and adds a tag that detects if anyone changed it.
- **Ed25519:** a digital signature that proves who sent something. Used to sign blocks.
- **DKMS:** the key manager. It creates keys, rotates them regularly, and revokes stolen ones. It rotates faster if a cluster looks sketchy.
- **Nonce/IV:** a number used only once, so the same message encrypts differently every time.

### Security experiments (OBJ7)

They inject fake attacks in the sim: eavesdropping, spoofing, data tampering, fancier ML attacks (evasion, poisoning, backdoors), and ledger attacks (Sybil = one attacker pretending to be many nodes). They report defense % (98–99% for basic attacks, 70–83% for advanced ones), attack success rate, false alarms, and overhead.

Honest take: this part is hard to reproduce. It's mostly simulated "did the attack work or not" logic and isn't tightly tied to the scheduler. Optional for us.

---

## 10. CBS–D* Lite (how mobile servers drive)

The city map gets a grid on top, and some cells are risky or blocked (the dark areas in Fig. 3).

- **D\* Lite:** a pathfinding algorithm (cousin of A\*) that finds the shortest route for one vehicle and re-plans quickly if the map changes.
- **CBS (Conflict-Based Search):** with many vehicles, plan each one separately, find the first moment two would crash, add a rule like "vehicle A can't be there at time t," re-plan, and repeat.

For us this is overkill. Straight-line driving or plain A\* is fine.

---

## 11. The data

They use real data: the Brașov (Romania) pollution dataset, 449 sites, Aug–Oct 2014 (part of the [CityPulse smart city datasets](http://iot.ee.surrey.ac.uk:8080/)). Each reading gets turned into a task (Algorithm 3):

1. Measure how abnormal the reading is and whether it's spiking.
2. Assign a tier: Immediate → URLLC, Routine → eMBB, Delayed → mMTC.
3. Estimate task size: bytes, CPU cycles, memory.
4. Pick a random deadline inside the tier's range, with a fixed seed so it's repeatable.
5. Assign a priority: high, medium, or low.

Then they build three load levels: **50k, 100k, and 150k tasks**.

So the timing comes from real sensors and the task details are synthetic. That works for us too.

---

## 12. The competitors (baselines)

- **Ensemble PPO:** several PPO agents voting together.
- **FD3QN:** a DQN variant (Q-learning with neural nets, plus stability tricks).
- **DDPG:** actor-critic built for continuous actions, hacked to pick discrete servers.
- **SAC:** actor-critic that also rewards randomness.
- **ccppo_lagr:** CCPPO with no masks. It uses a learnable "fine" for breaking rules (a Lagrangian) instead.

---

## 13. Metrics cheat sheet

- **Success:** % of tasks done on time. Violations = 100 − success.
- **Lp50 / Lp95:** median latency / latency of the slowest 5%. When people say "tail," they mean Lp95.
- **Throughput:** tasks finished per second.
- **LI (locality index):** % of tasks run in their home cluster.
- **χ_cloud:** % of tasks sent to the cloud.
- **Jain fairness:** 1 = perfectly even load, lower = lopsided.
- **The stats:** a 95% CI is the range where the true average likely sits. Wilcoxon tests whether two methods really differ. Benjamini–Hochberg is a correction so that running lots of tests doesn't hand you fake wins.

**Headline numbers to remember (150k tasks):** 82.77% on time, 17.23% violations, Lp50 = 53.23 s, Lp95 ≈ 147 s, cloud share 6.1%.

---

## 14. One tick of the simulation, start to finish

1. A task arrives from a sensor in cluster 3 (URLLC, 10 s deadline).
2. The sim computes the latency for every server (send + drive + queue + process).
3. It builds the mask, blocking servers that fail the deadline, slice, capacity, energy, or reputation checks.
4. Cluster 3's actor network outputs logits.
5. The nudges get added: Nash + PID + leader.
6. Masked softmax turns scores into probabilities, and a server gets sampled.
7. The sim runs the task: the queue grows, a mobile drives if needed, energy gets used.
8. The task finishes on time or late, and the agent gets a reward.
9. A log entry goes to the ledger, and reputation and debt update.
10. After a batch of tasks, PPO updates the actors and the shared critic. Every few episodes, PSO tweaks the knobs.

Repeat this for every task in the episode (up to 150k).

---

## 15. What actually matters for us

| Priority | Stuff |
|---|---|
| **Must build** | city + servers + clusters, latency formulas, slices, masks, multi-agent PPO with shared critic, metrics |
| **Simple version** | ledger + reputation feedback, PID nudge, leader bonus |
| **Skip or fake** | PSO (tune by hand), CBS–D* Lite (straight lines), BFT, DKMS, full attack experiments, Jury/drift analysis |

That keeps our time on the novelty and the improvements, which is the part that gets graded.

**Where to read in the paper first:** Sections III-A to III-H (the model), IV-B and IV-C (data + simulator), V-E (ablations).
