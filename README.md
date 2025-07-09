# MT-MARL-SG
This repository aims to address the challenges of multi-task multi-agent reinforcement learning (MT-MARL). By utilizing a skill graph, our hierarchical multi-task learning framework offers three key advantages: (1) it effectively handles MT-MARL problems with unrelated tasks, (2) it demonstrates superior knowledge transfer capabilities compared to standard hierarchical approaches, and (3) the training of the upper-layer skill graph is independent of the lower layer, leading to improved generalization.

## Requirements
Only Ubuntu 20.04 is recommended.

## Installation
1. Create a new virtual environment using the following command:
```bash
   conda create -n xxx(your env name) python=3.8
```
2. Navigate to the 'Swarm_test' folder and run the following command to install dependencies:
```bash
   pip install -r requirements.txt
```
3. Visit the [PyTorch official website](https://pytorch.org/get-started/previous-versions/) and install the GPU version of PyTorch according to your system configuration, such as
```bash
   pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```
4. Navigate to the 'customized_gym' folder and run the following command to install the MT-MARL environment:
```bash
   pip install -e .
```
5. If you encounter any other missing packages during the process, feel free to install them manually using ``pip install xxx``

## Example
### training
To be completed

### evaluation
To be completed

## Troubleshooting
Please open an [Issue](https://github.com/Guobin-Zhu/MT-MARL-SG/issues) if you have some trouble and advice.

The document is being continuously updated.
