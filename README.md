# RL Network Defense Bot

## Description

RL Defense Bot is a reinforcement learning-based network defense system designed to autonomously protect networks from cyber attacks. The system uses machine learning models for intrusion detection and a Q-learning agent to make real-time defense decisions. It supports both simulation mode for training and live mode using VMware virtual machines for realistic testing.

The project implements an intelligent firewall management system that learns to block malicious traffic patterns through reinforcement learning, combined with a Security Operations Center (SOC) dashboard for monitoring and visualization.

## Features

- **Reinforcement Learning Agent**: Q-learning based agent that learns optimal defense strategies
- **Machine Learning IDS**: Ensemble of LightGBM and Logistic Regression models for intrusion detection
- **Real-time Dashboard**: Flask-based SOC dashboard with live metrics and firewall logs
- **Dual Mode Operation**:
  - Simulation mode: Synthetic traffic generation for training
  - Live mode: Real VMware VMs (Ubuntu defender, Kali attacker)
- **Automated VM Management**: Automatic startup and configuration of virtual machines
- **Firewall Integration**: Direct iptables management on Ubuntu VM
- **Threat Classification**: Detection of DDoS, PortScan, Botnet, and Brute Force attacks
- **Checkpointing**: Model saving and loading for continued training

## Architecture

The system consists of several key components:

- **Agent**: Q-learning agent with discretized state space based on ML probabilities
- **Environment**: Network defense environment with attack simulation
- **Simulator**: Traffic generator for synthetic attacks
- **Monitor**: State builder and metrics collection
- **Classifier**: Intrusion detection using pre-trained ML models
- **Defense**: Firewall rule management (iptables)
- **Dashboard**: Real-time monitoring interface

## Screenshots

### Dashboard Overview
![Dashboard](Project%20Screenshots/Dashboard.jpeg)

### DDoS Attack Detection
![DDoS Attack](Project%20Screenshots/DDos_attack.jpeg)

### Classification Results
![Classification](Project%20Screenshots/Classification.jpeg)

### System Logs
![Logs](Project%20Screenshots/Logs.jpeg)

### Firewall Rules
![Firewall Rules](Project%20Screenshots/firewall_rules.jpeg)

### Learning Graphs
![Graph 1](Project%20Screenshots/Graph1.jpeg)

![Graph 2](Project%20Screenshots/Graph2.jpeg)

### Visualizations
![Visuals](Project%20Screenshots/visuals.jpeg)

## Requirements

- Python 3.8+
- VMware Workstation
- Ubuntu and Kali Linux VMs
- Pre-trained ML models (LightGBM, Logistic Regression, Scaler, Feature Selector, Label Encoder)

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd rl-defense-bot
   ```

2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Place the required ML model files in the `model/` directory:
   - lightgbm.pkl
   - logistic.pkl
   - scaler.pkl
   - feature_selector.pkl
   - label_encoder.pkl

4. Configure VMware VM paths in `config.yaml`

5. Set up virtual machines:
   - Ubuntu VM with SSH access
   - Kali Linux VM for attack simulation

## Configuration

Edit `config.yaml` to configure:

- Network settings (IP addresses, subnet)
- VM credentials and paths
- Training parameters (learning rate, episodes, etc.)
- Model file paths

## Usage

### Simulation Mode (Recommended for Training)
```
python main.py --sim -n 50
```

### Live Mode (Requires VMware VMs)
```
python main.py --live -n 50
```

### Dashboard
Run the dashboard separately:
```
python -m dashboard.app
```
Access at http://localhost:5000

### Command Line Options
- `--live`: Enable live VM mode
- `--sim`: Force simulation mode
- `-n`: Number of episodes
- `--help`: Show all options

## Project Structure

```
rl-defense-bot/
├── main.py                 # Main entry point
├── config.yaml            # Configuration file
├── ubuntu_metrics_api.py  # Ubuntu VM metrics collection
├── agent/
│   ├── environment.py     # RL environment
│   ├── simulator.py       # Traffic simulator
│   ├── attack_launcher.py # Attack generation
│   └── vm_manager.py      # VM lifecycle management
├── classifier/
│   └── intrusion_predictor.py # ML-based IDS
├── dashboard/
│   ├── app.py            # Flask dashboard
│   └── templates/
│       └── soc_dashboard.html
├── defense/
│   └── ubuntu_firewall.py # Firewall management
├── model/                # ML model files
├── monitor/
│   └── state_builder.py  # State observation building
├── checkpoints/          # Training checkpoints
└── logs/                 # Log files
```
