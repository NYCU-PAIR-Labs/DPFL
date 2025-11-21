# Dynamic Client Participation in Federated Learning: Benchmarks and a Knowledge Pool Plugin

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-31014/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview
Federated Learning (FL) enables multiple clients to collaboratively train a machine learning model while keeping their data decentralized. However, most existing FL frameworks assume static client participation, where all clients are available throughout the training process. In real-world scenarios, client availability can be dynamic due to various factors such as network conditions, device constraints, and user behavior. This project addresses the challenge of dynamic client participation in FL by providing a flexible and extensible platform that supports various dynamic participation patterns.

### DPFL Benchmarking Platform
<p align="center">
  <img src="figures/DPFL-Framework.png" alt="DPFL Diagram" width="100%"/>
</p>

The platform offers:

- Controllable data models (e.g., IID, light/heavy-NIID)
- Realistic and probabilistic DP models with support for compute and transmission latencies
- Twelve state-of-the-art FL algorithms across four categories: 
  - Averaged-based (e.g., FedAvg, FedProx, SCAFFOLD)
  - Knowledge distillation-based (e.g., FedMD, FedDF, FedGen)
  - Prototype-based (e.g., MOON, FedProto, FPL)
  - Continual learning-based (e.g., FLwF, CFeD, FedCL)
- DPFL-specific evaluation metrics


### Knowledge Pool Federated Learning (KPFL) Plugin
<p align="left">
  <img src="figures/DPFL-KnowledgePool.png" alt="KPFL Diagram" width="60%"/>
</p>

KPFL is specifically designed to address the challenges posed by DPFL.
Our goal is to develop KPFL as a generic plug-in that operates independently of an underlying FL model, ensuring compatibility without interfering with the FL model.
KPFL maintains a knowledge pool $\mathcal{KP}$ comprising both active and idle clients. It profiles client knowledge in a generative and age-aware manner, and leverages knowledge distillation to enhance the final model.
As a result, KPFL is broadly compatible with a wide range of FL models.


## Installation
To install and set up the project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/NYCU-PAIR-Labs/DPFL.git
    ```
2. Create conda environment:
    ```bash
    conda create -n dpfl python=3.10.14
    conda activate dpfl
    
    # Install PyTorch (with CUDA support if available)
    conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 numpy==1.26.4 "mkl<=2023.1.0" -c pytorch -c nvidia
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Arguments
The following arguments are supported by the training script:

### General Arguments

| Argument | Description | Options |
| --- | --- | --- |
| `--algorithm` | Algorithm used for the training | `FedAvg`, `FedProx`, `SCAFFOLD`, `FedMD`, `FedDF`, `FedGen`, `MOON`, `FedProto`, `FPL`, `FLwF`, `CFeD`, `FedCL` |
| `--dataset` | Dataset used for the training | `MNIST`, `Cifar10`, `Cifar100`, `Caltech101`, `Office-Caltech` or you can load your own dataset by specifying `--custom_data_path` |
| `--model` | Model used for the training | `SimpleCNN`, `resnet10` |
| `--num_clients` | Number of clients | `int` |
| `--num_rounds` | Number of rounds | `int` |
| `--num_epochs` | Number of local epochs | `int` |
| `--batch_size` | Batch size for training | `int` |
| `--lr` | Learning rate for the optimizer | `float` |
| `--optimizer` | Optimizer used for training | `sgd`, `adam`|
| `--skew_type` | Type of data skew | `label` |
| `--alpha` | Concentration parameter for the Dirichlet distribution | `100`, `1.0`, `0.1` |
| `--augmentation` | Whether to use data augmentation (random crop and horizontal flip) | `1: True`, `0: False` |
| `--load_data_to_memory` | Whether to load the data to memory for reducing the overhead of data loader | `1: True`, `0: False` |

### DP Arguments
| Argument | Description | Options |
| --- | --- | --- |
| `--dynamic_type` | Type of dynamic client participation | `static`, `incremental-arrival`, `incremental-departure`, `round-robin`, `random`, `markov` |
| `--round_start` | Round to start dynamic client participation for incremental-arrival/departure | `int`, default: 50 |
| `--initial_clients` | Initial number of clients for `incremental-arrival` and `incremental-departure` | `int`, default: 5 |
| `--interval` | Interval of client change for `incremental-arrival` and `incremental-departure` | `int`, default: 10 |
| `--clients_per_interval` | Number of clients to change at each interval for `incremental-arrival/departure` and `round-robin` | `int`, default: 1 |
| `--overlap_clients` | Number of overlapping clients between two adjacent intervals for `round-robin` | `int`, default: 1 |
| `--sim` | Whether to use simulation mode with compute/communication latencies | `1: True`, `0: False` |
| `--kpfl` | Whether to use Knowledge-Pool (KPFL) | `1: True`, `0: False` |

Sample usage:
```bash
# Test with dynamic participation (static example)
python3 main.py --dataset cifar10 --algorithm fedavg --skew_type label --alpha 100.0 --dynamic_type static

# Test with dynamic participation (incremental-arrival example)
python3 main.py --dataset cifar10 --algorithm fedavg --skew_type label --alpha 100.0 --dynamic_type incremental-arrival --round_start 50 --initial_clients 5 --interval 10 --clients_per_interval 1

# Test with dynamic participation (incremental-departure example)
python3 main.py --dataset cifar10 --algorithm fedavg --skew_type label --alpha 100.0 --dynamic_type incremental-departure --round_start 50 --interval 10 --clients_per_interval 1
```

Enable Simulation Mode (with compute/communication latencies):
```bash
python3 main.py --dataset cifar10 --algorithm fedavg --skew_type label --alpha 100.0 --dynamic_type random --sim
```

Enable KPFL (knowledge-pool federated learning):
```bash
python3 main.py --dataset cifar10 --algorithm fedavg --skew_type label --alpha 100.0 --dynamic_type random --kpfl
```




## Extending the Platform
To extend the platform, you can add new federated learning algorithms (Strategy) or new dynamic client participation patterns (DPModel) by **SIMPLY** subclassing the provided templates.

### Strategy Template
Create a new file in `strategies/` (e.g., `mystrategy.py`) and subclass the `Strategy` class. 
You can refer to existing strategies for guidance.
``` python
from .strategy import Strategy

class MyStrategy(Strategy):
    def _initialization(self, **kwargs):
        # Optional: initialization logic
        pass

    def _server_train_func(self, cid, rounds, client_list, **kwargs):
        # Define how the server orchestrates client training
        pass

    def _server_agg_func(self, rounds, client_list, active_clients, global_model):
        # Define aggregation logic
        pass

    def _aggregation(self, server_round, client_models):
        # Actual aggregation process
        pass

    def _train(self, model, trainLoader, optimizer, num_epochs, **kwargs):
        # Client-side training logic
        return {"train_loss": 0.0, "train_acc": 0.0}

    def _test(self, model, testLoader, payload):
        # Client-side testing logic
        return {"test_loss": 0.0, "test_acc": 0.0}
```
**Minimal Example:**
See `strategies/fedavg.py` for a full implementation.


### DPModel Template
Create a new file in `dpmodels/` (e.g., `mydpmodel.py`) and subclass the `BaseDPModel` class.
You can refer to existing DP models for guidance.
``` python
from dpmodels.dpmodel import BaseDPModel

"""
- client_state: a matrix to store the state of clients in each round
    ------------------------------------
    |  Client  |         Round         |
    ------------------------------------
    | client 0 | 1 | 1 | 0 | 0 | 0 | 0 |
    | client 1 | 0 | 0 | 1 | 1 | 0 | 0 |
    | client 2 | 0 | 0 | 0 | 0 | 1 | 1 |
"""
class MyDPPattern(BaseDPModel):
    def set_pattern(self):
        # Example: alternate clients every round
        for r in range(self.num_rounds):
            self.client_state[r % self.num_clients, r] = 1
        return self.client_state
```
**Minimal Example:**
See `dpmodels/static.py` for a simple implementation where all clients participate in every round.


## Acknowledgement
This work was supported by NVIDIA Academic Grant Program, which provided the GPU resources used for our experiments. We also thank the NVIDIA NVFLARE team for their open-source federated learning framework, which inspired parts of our system design.


## Contributing
Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request. For more information, see our [contribution guidelines](CONTRIBUTING.md).


## Citation
If you use this code for your research, please cite the following paper:

```@inproceedings{your2024dynamic,
  title={Dynamic Client Participation in Federated Learning: Benchmarks and a Knowledge Pool Plugin},
  author={Ming-Lun Lee, Fu-Shiang Yang, Cheng-Kuan Lin, Yan-Ann Chen, Chih-Yu Lin, Yu-Chee Tseng},
  year={2025},
  eprint={2511.16523},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2511.16523},
}
```