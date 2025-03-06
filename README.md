## WSL / MacOS Setup
- Install MuJoCo: https://github.com/openai/mujoco-py?tab=readme-ov-file#install-mujoco
- Create virtual environment: `python3 -m venv venv`
- Activate virtual environment: `source venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`
- Run main file: `python3 main.py`

## Pace - Setup
```bash
module load anaconda3/2023.03

## Create conda environment
conda create -n spin-rl python=3.12 mesalib glew glfw -c conda-forge -y
conda activate spin-rl
echo "module load anaconda3/2023.03" >> ~/.bashrc

# Install Mujoco Engine
USER_DIR=$USER
echo $USER_DIR
wget -c "https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz"
mkdir -p /home/$USER_DIR/.mujoco
cp mujoco210-linux-x86_64.tar.gz /home/$USER_DIR/mujoco.tar.gz
rm mujoco210-linux-x86_64.tar.gz
mkdir -p /home/$USER_DIR/.mujoco
tar -zxvf /home/$USER_DIR/mujoco.tar.gz -C /home/$USER_DIR/.mujoco
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$USER_DIR/.mujoco/mujoco210/bin" >> ~/.bashrc
echo "export MUJOCO_PY_MUJOCO_PATH=/home/$USER_DIR/.mujoco/mujoco210" >> ~/.bashrc

# Activate 
source ~/.bashrc
conda activate spin-rl

#  install 
export CC="/usr/bin/gcc"
pip install "mujoco_py>=2.0"
pip install "cython<3"

# compile Mujoco using GCC
python
import mujoco_py
```

## Pace - Submitting a Job
```bash
cd sbatch
# Submit Soft Actor Critic Job
sbatch sac.sbatch 
```