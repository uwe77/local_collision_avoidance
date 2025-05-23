# 🛥️ USV Reinforcement Learning Environments with Gymnasium & Gazebo

This repository contains a collection of reinforcement learning (RL) environments tailored for unmanned surface vehicle (USV) simulation and training using **Gymnasium** and **Gazebo**. It includes custom environments, training scripts for PPO and DDPG, feature extractors, and support utilities for ROS and Docker integration.

## 🚀 Features

### ✅ Environments
- **USVLocalCollisionAvoidanceV0**: For local obstacle avoidance using LiDAR and RL.
- **JSTeacherStudentV0**: A teacher-student multi-mission environment for complex task learning.

### 🧠 RL Algorithms
- **PPO** (`train_usv.py`, `train_usv_load.py`)
- **DDPG** (`train_js.py`, `train_js_load.py`)

### 🛠️ Custom Utilities
- **Feature Extractors**:
  - `JsFeatureExtractor`: Custom extractor for multidimensional observations in DDPG.
  - `USVFeatureExtractor`: Tailored for USV environment.
- **Callbacks**:
  - `PPOSaveSameModelCB` & `DDPGSaveSameModelCB`: Custom checkpointing with overwrite options.
- **Path Management**:
  - `add_path.py`: Ensures correct module imports from `rl/`.

### 🔧 Gazebo Integration
- Physics & model state interfaces via:
  - `GazeboROSConnector`
  - `GazeboBaseModel`
  - `GazeboUSVModel` / `GazeboJSModel`

### 🐳 Docker Support
- Shell scripts for pulling and running Docker images:
  - `Ubuntu 20.04`, `IPC 18.04`
- Scripts configure ROS_IP and ROS_MASTER_URI.

### 📦 Package Setup
- Structured as a Gymnasium-compatible package (`gymnasium-cus`)
- Installable via `setup.py` and `install_gymnasium.sh`

## 📁 Project Structure

```

gymnasium\_cus/
├── envs/
│   ├── usv\_local\_collision\_avoidance.py
│   └── js\_teacher\_student.py
├── rl/
│   ├── ppo/
│   └── ddpg/
│       ├── train\_js.py
│       ├── train\_js\_load.py
│       └── js\_feature\_extractor.py
├── utils/
│   ├── callback/
│   └── feature\_extractor/
├── scripts/
│   ├── set\_ros\_ip.sh
│   ├── set\_ros\_master.sh
│   ├── install\_gymnasium.sh
│   └── docker/
└── setup.py

````

## 🛠 Installation

```bash
git clone <repo-url>
cd <repo-name>
bash scripts/install_gymnasium.sh
````

## 🧪 Training

```bash
python3 rl/ppo/train_usv.py
python3 rl/ddpg/train_js.py
```

## 🧾 Acknowledgments

Developed by [uwe77](mailto:uwe90711@gmail.com), integrating Gymnasium environments with Gazebo and ROS to enable modular and scalable USV control training.

---
