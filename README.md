# babyshark-FLY 🪰🎶

**A Connectome-Constrained, Closed-Loop Biomechanical Simulation for Auditory Rhythm Synchronization in *Drosophila melanogaster***

`babyshark-FLY` is a biologically grounded neuro-mechanical simulation where acoustic music signals drive an auditory connectome pathway (Johnston's Organ $\to$ AMMC $\to$ Descending Neurons), modulating a coupled Central Pattern Generator (CPG) governing the 6-legged articulated exoskeleton of the fruit fly avatar in physics simulation.

---

## 🏛️ Closed-Loop Architecture

```text
                    ┌─────────────────────────────┐
                    │        AUDIO INPUT          │
                    │         WAV / MP3           │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │   AUDIO / RHYTHM PIPELINE   │
                    │ • Continuous beat phase     │
                    │ • Onset energy envelope     │
                    │ • Multi-band spectral power │
                    │ • Timestep synchronization  │
                    └──────────────┬──────────────┘
                                   │ acoustic waveform & power
                                   ▼
                    ┌─────────────────────────────┐
                    │   AUDITORY NEURAL MODEL     │
                    │ (Connectome-Constrained)    │
                    │ • Mode B: Pure Biological   │
                    │ • Johnston's Organ (JO-A/B) │
                    │ • AMMC interneurons         │
                    │ • Descending Neurons (DNs)  │
                    │ • FlyWire synaptic matrices │
                    └──────────────┬──────────────┘
                                   │
                         neural state / activity (24 DNs)
                                   │
                                   ▼
        ┌─────────────────────────────────────────────────┐
        │       PPO ACTOR-CRITIC / CPG CONTROLLER         │
        │                                                 │
        │ Observation (83-D):                             │
        │ • 24 Auditory Descending Neurons (DNs)          │
        │ • 54 Proprioceptive kinematics & contacts       │
        │ • 5 Normalized audio features                   │
        │                                                 │
        │ Action (8-D continuous modulations):            │
        │ • Stepping frequency & joint amplitude          │
        │ • Body heave bobbing & swing duty cycle         │
        │ • Bilateral sway & front/mid/hind leg scaling   │
        └──────────────────────┬──────────────────────────┘
                               │
                               │ coordinated joint targets
                               ▼
        ┌─────────────────────────────────────────────────┐
        │      DUAL-BACKEND PHYSICS SIMULATION            │
        │                                                 │
        │ 1. MuJoCoFlyGymEnv (Genuine MuJoCo 3.x):        │
        │    • EPFL NeuroMechFly adult Drosophila model   │
        │    • 204 position actuators, mesh collisions    │
        │                                                 │
        │ 2. SimpleFlyEnv (Fast Numerical Model):         │
        │    • 1,800+ steps/s for rapid RL sweeps         │
        │    • Compliant PD joints & contact kinematics   │
        └──────────────────────┬──────────────────────────┘
                               │
                     body kinematics & state
                               │
                  ┌────────────┴────────────┐
                  │                         │
                  ▼                         ▼
       ┌──────────────────┐      ┌────────────────────────┐
       │ PROPRIOCEPTION   │      │  DANCE REWARD ENGINE   │
       │ (54D Vector)     │      │                        │
       │ • Joint angles   │      │ • Footfall Event Sync  │
       │ • Velocities     │      │ • Onset correlation    │
       │ • Foot contacts  │      │ • Stepping rhythmicity │
       │ • Body Euler     │      │ • Postural stability   │
       │ • Angular vel.   │      │ • Movement quality     │
       │ • COM position   │      │ • Anti-flail penalty   │
       └──────────────────┘      └────────────────────────┘
```

---

## 🔬 Scientific Foundation & Technical Reality

### 1. Connectome Topology (`data/flywire/`, `connectome_auditory.py`)
- **Node Metadata & FlyWire Mapping**: Contains mapped annotations (`data/flywire/neurons.json`, `synapses.json`) reflecting adult *Drosophila* cell types:
  - **Johnston's Organ (JON)**: $2 \times 32$ mechanoreceptors (subpopulations JO-A for vibrations/transients, JO-B for continuous song).
  - **AMMC Interneurons**: $2 \times 48$ units (`AMMC-aLN`, `AMMC-B1`, `aPN1`, `aPN2`) with tonotopic receptive fields and cross-hemispheric commissural inhibition.
  - **Descending Neurons (DNs)**: 24 units (`DNp01`, `aDN1`, `aDN2`, `MDN`, `DNg02`, `DNb01`).
- **Biologically Driven (Mode B)**: Johnston's Organ receives **only** raw acoustic waveforms and multi-band spectral vibration power. It does **not** receive artificial pre-computed beat pulses.

### 2. Dual Physics Backends (`fly_env.py`)
- **`MuJoCoFlyGymEnv`**: Genuinely compiles and steps EPFL's **NeuroMechFly** in **MuJoCo 3.x** with 204 actuated DoFs, rigid-body contact solver, and gravitational dynamics.
- **`SimpleFlyEnv`**: High-throughput custom biomechanical model running at **1,800+ steps/second**, ideal for rapid PPO policy exploration and parameter tuning.

### 3. Actual Footfall Event Synchronization (`dance_reward.py`)
- Instead of measuring internal oscillator angles, the reward tracks **physical footfall touchdowns** (transition from swing to stance) and penalizes temporal latency to the nearest musical beat:
  $$\text{beat\_sync} = \frac{1}{N_{\text{touchdowns}}} \sum_{i} \exp\left(-\frac{|t_{\text{footfall}, i} - t_{\text{nearest\_beat}}|^2}{\sigma^2}\right)$$

---

## 📊 Comparative Benchmark Results

Evaluated across 2,000 simulation steps on identical 120 BPM audio tracks:

| Controller | Mean Step Reward | Footfall Event Sync | Posture Stability | Energy Cost |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline A** (Autonomous CPG, No Music) | 4.319 | 0.031 | 1.800 | **0.184** |
| **Baseline B** (Beat-Heuristic CPG) | 4.280 | 0.030 | 1.800 | 0.213 |
| **Baseline C** (Random MLP CPG) | 4.358 | 0.032 | 1.800 | 0.238 |
| **Baseline D** (Connectome-Constrained CPG) | 4.334 | 0.063 | 1.800 | 0.386 |
| **Baseline E** (Trained PPO $\to$ CPG) | **4.364** | **0.065** | 1.800 | 0.375 |

> **Key Finding**: Under the strict physical footfall touchdown timing metric, the biological connectome pathway (**Baseline D**) more than **doubles** the footfall synchronization score compared to autonomous and heuristic baselines ($0.063$ vs. $0.030$), and **Trained PPO** (**Baseline E**) achieves the highest overall reward ($4.364$).

---

## 🚀 Quickstart

### 1. Installation

```bash
git clone https://github.com/Siddz-17/babyshark-FLY.git
cd babyshark-FLY

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Run Closed-Loop Simulation

```bash
# Fast numerical simulation
python test_closed_loop.py

# Or with genuine MuJoCo 3.x physics
python -c "from fly_env import DrosophilaFlyEnv; env = DrosophilaFlyEnv(backend='mujoco'); print('MuJoCo ready!')"
```

### 3. Run Comparative Baseline Experiments

```bash
python test_baselines_comparison.py
```

### 4. Train PPO Reinforcement Learning Policy

```bash
python train_ppo.py
# Or using the modular runner with YAML config:
python ppo/train.py
```

### 5. Evaluate Trained Policy & Export Video

```bash
python evaluate_policy.py
```

---

## 📁 Repository Structure

```
babyshark-FLY/
├── data/
│   └── flywire/                 # Mapped connectome nodes, root IDs & synaptic matrices
│       ├── neurons.json
│       ├── synapses.json
│       ├── W_jon_ammc.npy
│       └── W_ammc_dn.npy
├── ppo/                         # Modular Reinforcement Learning package
│   ├── policy.py                # Gaussian actor policy
│   ├── value.py                 # State-value baseline
│   ├── buffer.py                # Vectorized rollout buffer with GAE
│   ├── agent.py                 # ActorCritic container
│   └── train.py                 # Modular training runner
├── configs/
│   └── ppo.yaml                 # PPO training & environment hyperparameters
├── audio_pipeline.py            # Onset, spectral, and beat phase extraction
├── connectome_auditory.py       # Johnston's Organ (JON -> AMMC -> DN) biophysical circuit
├── cpg_controller.py            # 6-legged coupled phase oscillator network
├── fly_env.py                   # SimpleFlyEnv & MuJoCoFlyGymEnv dual backends
├── dance_gym_env.py             # Gymnasium-compliant RL environment (83-D obs, 8-D act)
├── dance_reward.py              # Physical footfall event synchronization reward engine
├── baselines.py                 # 5 comparative benchmark controllers
├── test_closed_loop.py          # End-to-end closed-loop test script
├── test_baselines_comparison.py # Multi-controller benchmark runner
├── evaluate_policy.py           # Evaluation & video export
└── requirements.txt
```

---

## 📜 License
MIT License.
