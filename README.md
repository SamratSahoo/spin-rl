# Dexterous Hand Manipulation with Deep Reinforcement Learning

## Project Overview

Dexterous hand manipulation remains a challenging problem in robotics due to the high dimensionality and complexity of motor control. This project explores deep reinforcement learning (DRL) methods to train a robotic hand to perform pen manipulation tasks, such as spinning a pen using the Shadow Dexterous Hand from [Gymnasium Robotics](https://gymnasium.farama.org/environments/robotics/).

We evaluate:

- **Baseline methods:** Vanilla Policy Gradient (VPG), Deep Deterministic Policy Gradient (DDPG)  
- **Advanced methods:** Soft Actor-Critic (SAC), Proximal Policy Optimization (PPO)

Key contributions:
- **Reward Shaping:** A carefully designed reward function for complex pen spinning
- **EnvGz (Environment Generalization Optimization):** A form of goal-conditioned reinforcement learning for training agents to handle generalizable conditions within their environments

Initial results indicate that while reward shaping alone may not suffice due to the risk of reward hacking, environment generalization shows greater promise for enabling robust, dexterous control.


## Local Setup (WSL / macOS)

1. **Install MuJoCo:**  
   [MuJoCo Installation Guide](https://github.com/openai/mujoco-py?tab=readme-ov-file#install-mujoco)

2. **Set up virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Start training the agent:**
   ```bash
   # Algorithm Choices: [vpg, ddpg, sac, ppo]
   # Environment Choices: [wrapped, general, general-v2]
   python main.py --algorithm sac --environment general-v2
   ```

4. **View Tensorboard Logs:**
   ```bash
   tensorboard --logdir runs/
   ```

## PACE Setup

1. **Environment Setup:**

```bash
module load anaconda3/2023.03

# Create conda environment
conda create -n spin-rl python=3.12 mesalib glew glfw -c conda-forge -y
conda activate spin-rl
pip install -r requirements.txt --no-deps

# Persistent anaconda module loading
echo "module load anaconda3/2023.03" >> ~/.bashrc
```

2. **Install MuJoCo:**

```bash
USER_DIR=$USER
wget -c "https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz"
mkdir -p /home/$USER_DIR/.mujoco
cp mujoco210-linux-x86_64.tar.gz /home/$USER_DIR/mujoco.tar.gz
rm mujoco210-linux-x86_64.tar.gz
tar -zxvf /home/$USER_DIR/mujoco.tar.gz -C /home/$USER_DIR/.mujoco

# Set environment variables
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$USER_DIR/.mujoco/mujoco210/bin" >> ~/.bashrc
echo "export MUJOCO_PY_MUJOCO_PATH=/home/$USER_DIR/.mujoco/mujoco210" >> ~/.bashrc

# Activate everything
source ~/.bashrc
conda activate spin-rl

# Install MuJoCo Python bindings
export CC="/usr/bin/gcc"
pip install "mujoco_py>=2.0"
pip install "cython<3"

# Verify installation
python -c "import mujoco_py"
```

3. **Running Experiments on PACE:**

Submit a training job (i.e., Soft Actor-Critic):

```bash
cd sbatch
sbatch sac.sbatch
```

4. **View Tensorboard Logs:**

```bash
tensorboard --logdir runs/
```