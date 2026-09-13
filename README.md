# babyshark-FLY 🪰🎶

**A Connectome-Constrained, Closed-Loop Biomechanical Simulation for Auditory Rhythm Synchronization in *Drosophila melanogaster***

`babyshark-FLY` is a biologically grounded neuro-mechanical simulation where acoustic music signals drive an auditory connectome pathway (Johnston's Organ $\to$ AMMC $\to$ Descending Neurons), which modulates a coupled Central Pattern Generator (CPG) governing the 6-legged articulated exoskeleton of the fruit fly avatar in physics simulation.

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
                    │ • Onset envelope            │
                    │ • Continuous beat phase     │
                    │ • Multi-band spectral power │
                    │ • Timestep synchronization  │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │   AUDITORY NEURAL MODEL     │
                    │ (Connectome-Constrained)    │
                    │ • Johnston's Organ (JO-A/B) │
                    │ • AMMC interneurons         │
                    │ • Descending Neurons (DNs)  │
                    │ • Lateral & commissural inh.│
                    └──────────────┬──────────────┘
                                   │
                         neural state / activity
                                   │
                                   ▼
        ┌─────────────────────────────────────────────────┐
        │              CPG MOTOR CONTROLLER               │
        │                                                 │
        │ Modulation Inputs:                              │
        │ • DN transient, sustained & asymmetry drive     │
        │ • Proprioceptive joint & contact states         │
        │ • Continuous beat phase & tempo                 │
        │                                                 │
        │ Action:                                         │
        │ • Coupled phase oscillator stepping frequency   │
        │ • Joint excursion amplitude & body bobbing      │
        │ • Bilateral sway & stance/swing duty cycle      │
        └──────────────────────┬──────────────────────────┘
                               │
                               │ coordinated joint targets
                               ▼
        ┌─────────────────────────────────────────────────┐
        │       NEUROMECHFLY / FLYGYM PHYSICS AVATAR      │
        │                                                 │
        │ • Articulated 6-legged adult Drosophila model   │
        │ • 18 actuated joint DOFs (coxa, femur, tibia)   │
        │ • Ground contact mechanics & stability          │
        │ • Full proprioceptive feedback                  │
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
       │ • Joint angles   │      │ • Beat phase sync      │
       │ • Velocities     │      │ • Onset correlation    │
       │ • Foot contacts  │      │ • Stepping rhythmicity │
       │ • Body Euler     │      │ • Postural stability   │
       │ • Angular vel.   │      │ • Movement quality     │
       │ • COM position   │      │ • Anti-flail penalty   │
       └──────────────────┘      └────────────────────────┘
```

---

## 🔬 Core Modules

### 1. Audio & Rhythm Pipeline (`audio_pipeline.py`)
- Ingests audio files or generates synthetic rhythmic test pulses (120 BPM dance beats, courtship pulse songs).
- Extracts instantaneous beat phase $\phi(t) \in [0, 2\pi)$ (with $\phi=0$ aligned to downbeat moments).
- Frequency decomposition matching *Drosophila* auditory reception without hard-clipping high musical frequencies:
  - Low band (100–300 Hz, pulse song resonance & bass drum transients)
  - Mid band (300–800 Hz, sine song harmonics & melody)
  - High band (>800 Hz, percussion & treble)

### 2. Connectome Auditory Pathway (`connectome_auditory.py`)
- Biologically derived 184-neuron subgraph inspired by FlyWire and FAFB connectome data:
  - **Johnston's Organ (JON)**: $2 \times 32$ bilateral mechanoreceptors (subpopulations JO-A for vibrations/transients, JO-B for continuous song).
  - **AMMC Interneurons**: $2 \times 48$ units with tonotopic receptive fields, local recurrent excitation, and cross-hemispheric commissural inhibition.
  - **Descending Neurons (DNs)**: 24 units categorized into transient beat trackers (DN 0–7), sustained rhythm integrators (DN 8–15), and bilateral steering/sway units (DN 16–23).

### 3. Central Pattern Generator (`cpg_controller.py`)
- 6 coupled phase oscillators governing the legs with differential phase dynamics:
  $$\dot{\theta}_i = 2\pi f_i + \sum_j w_{ij} \sin(\theta_j - \theta_i - \Delta\phi_{ij})$$
- Dynamically translates descending neural commands into smooth 3D joint kinematic trajectories for coxa, femur, and tibia actuators.

### 4. Biomechanical Physics Avatar (`fly_env.py`)
- MuJoCo 3.x / compliant physics simulation of adult *Drosophila melanogaster*.
- Models 18 actuated joint DOFs with PD position control and ground contact dynamics.
- Rich 54-dimensional proprioceptive feedback state.
- Headless off-screen RGB renderer with real-time HUD telemetry.

### 5. Multi-Objective Dance Reward Engine (`dance_reward.py`)
- Multi-component evaluation avoiding parasitic high-frequency flailing:
  $$R_t = w_{beat} R_{beat} + w_{onset} R_{onset} + w_{rhythm} R_{rhythm} + w_{stab} R_{stab} + w_{qual} R_{qual} - w_{energy} C_{energy} - w_{fall} C_{fall}$$

---

## 📊 Comparative Benchmark Results

Tested across 2,000 closed-loop physics steps (4.0 seconds, $\Delta t = 2 \text{ ms}$) on identical 120 BPM dance audio:

| Controller | Mean Step Reward | Beat Sync Score | Posture Stability | Energy Cost |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline A** (Autonomous Walker, No Music) | 5.127 | 0.417 | 2.000 | **0.187** |
| **Baseline B** (Heuristic Beat $\to$ CPG) | 5.403 | **0.662** | 2.000 | 0.196 |
| **Baseline C** (Direct MLP Policy) | 5.205 | 0.379 | 2.000 | 0.444 |
| **Baseline D** (Connectome JON $\to$ AMMC $\to$ DN) | **5.486** | 0.528 | 2.000 | 0.362 |

> **Key Finding**: The connectome-derived auditory pathway (**Baseline D**) achieves the **highest overall reward (5.486)**. While heuristic beat-triggering (Baseline B) achieves sharp phase spikes, the connectome integrates both acoustic transients and sustained frequency energy to produce more natural, continuous dancing motion while maintaining rock-solid postural stability.

---

## 🚀 Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Siddz-17/babyshark-FLY.git
cd babyshark-FLY

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Closed-Loop Simulation & Video Export

```bash
python test_closed_loop.py
```
This runs a 4-second closed-loop simulation at **~1,800+ steps/second**, saving:
- `fly_dance_closed_loop.mp4` (Off-screen rendered 3D fly dance with HUD telemetry)
- `closed_loop_telemetry.png` (Multi-panel telemetry plot of audio, neural firing, frequency, and rewards)

### 3. Run Comparative Baseline Experiments

```bash
python test_baselines_comparison.py
```

---

## 💻 Hardware Compatibility
Optimized to run locally on standard hardware (e.g. AMD Ryzen 5 5600H + NVIDIA GeForce RTX 3050 4GB). The connectome subgraph design ensures high biological fidelity without requiring a supercomputing cluster.

---

## 📜 License
MIT License.
