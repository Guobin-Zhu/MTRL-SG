# MT-MARL-SG
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Ubuntu 20.04](https://img.shields.io/badge/ubuntu-20.04-orange.svg)](https://releases.ubuntu.com/20.04/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🔍 Overview
This repository supports our paper "Multi-Task Multi-Agent Reinforcement Learning via Skill Graphs." The proposed skill graph offers three key benefits: (1) effective handling of unrelated tasks in MT-MARL, (2) improved knowledge transfer over standard hierarchical methods, and (3) decoupled training of skill graphs and low-level policies for better generalization.

## 📋 Requirements
- **Operating System**: Ubuntu 20.04 (recommended)
- **Python**: 3.8
- **GPU**: CUDA-compatible GPU

## 📁 Project Structure
```
MT-MARL-SG/
├── mt_marl_sg/                          # Core framework implementation
│ ├── algorithm/                         # Reinforcement learning algorithms
│ ├── cfg/                               # Training configurations
│ ├── eval/                              # Evaluation scripts
│ ├── skill_graph/                       # Skill graph modules
│ ├── train/                             # Training scripts
│ └── requirements.txt                   # Module-level dependencies
├── cus_gym/                             # Custom Gym environments
│ ├── gym/                               # Gym environment implementation
│ └── ...
└── README.md                            # This document
```

## ⚙️ Installation
### 1. Create Virtual Environment
Create a new virtual environment using the following command:
```bash
   conda create -n xxx(your env name) python=3.8 # or > 3.8
```
### 2. Install Dependencies
Navigate to the `mt_marl_sg` folder and run the following command to install dependencies:
```bash
   pip install -r requirements.txt
```
If you encounter any other missing packages during the process, feel free to install them manually using ``pip install xxx``

### 3. Install PyTorch
Visit the [PyTorch official website](https://pytorch.org/get-started/previous-versions/) and install the GPU version of PyTorch according to your system configuration, such as
```bash
   pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```
### 4. Install Custom Environment
Navigate to the 'custom_gym' folder and run the following command to install the MT-MARL environment:
```bash
   pip install -e .
```

### 5. Set Environment Variables
To make the project accessible within your Python environment, add the following path to your `~/.bashrc`:

```bash
echo 'export PYTHONPATH="$PYTHONPATH:/path/to/your/mt_marl_sg/"' >> ~/.bashrc
source ~/.bashrc
```

### 6. Compile the C++ Library
Several environment functions are implemented in C++ to improve performance. To build the C++ extension, execute the following:
```bash
cd cus_gym/gym/envs/customized_envs/envs_cplus
chmod +x build.sh
./build.sh
```


## 🚀 Usage
### 1. Train Low-Level Skills (Flocking Task)
To train low-level skills using the flocking task, navigate to the training script directory and run the following command:
```bash
cd mt_marl_sg/train
python ./train_flocking.py
```
The corresponding training configuration is located in `cfg/flocking_cfg.py`

### 2. Skill Evaluation
After training, you can evaluate the learned skills by modifying the experiment identifier in your evaluation script. Replace the curr_run variable with the directory name corresponding to your experiment:
```bash
curr_run = '2025-01-19-15-58-03'  # Replace with your experiment timestamp
```
then, run
```bash
python ./eval_flocking.py
```

### 3. Train High-Level Skill Graph
To train the high-level skill graph over the learned low-level skills, navigate to the skill graph training directory and run:
```bash
cd skill_graph
python ./train.py
```

### 4. Skill Inference
Once training is completed, you can perform inference by updating the experiment path in `inference.py`. Replace the `your_experiment_name` placeholder with your actual run directory:
```bash
log_dir = os.path.join(current_dir, 'runs', 'your_experiment_name')
```
then, run 
```bash
python ./inference.py
```

## 📝 Getting Help
Please open an [Issue](https://github.com/Guobin-Zhu/MT-MARL-SG/issues) if you have some trouble and advice.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This documentation is continuously being updated. For the latest information, please check the repository regularly.
