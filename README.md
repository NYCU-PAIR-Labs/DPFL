# Dynamic Client Participation in Federated Learning

This repository is the official implementation of dynamic client participation in FL.

## Installation
To install and set up the project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/NYCU-PAIR-Labs/DPFL.git
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Arguments
The following arguments are supported by the training script:

### General Arguments

| Argument | Description | Options |
| --- | --- | --- |
| `--algorithm` | Algorithm used for the training | `FedAvg`, `FedProx`, `SCAFFOLD`, `FedMD`, `FedDF`, `FedGen`, `MOON`, `FedProto`, `FPL`, `FLwF`, `CFeD`, `FedCL` |
| `--dataset` | Dataset used for the training | `mnist`, `cifar10`, `cifar100`, `fer2013`, `emnist`, `caltech101`, `office-caltech` or you can load your own dataset by specifying `--custom_data_path` |
| `--model` | Model used for the training | `SimpleCNN`, `resnet10` |
| `--num_clients` | Number of clients | `int` |
| `--num_rounds` | Number of rounds | `int` |
| `--num_epochs` | Number of local epochs | `int` |
| `--batch_size` | Batch size for training | `int` |
| `--lr` | Learning rate for the optimizer | `float` |
| `--optimizer` | Optimizer used for training | `sgd`, `adam`|
| `--skew_type` | Type of data skew | `label`, `feature`, `quantity` |
| `--alpha` | Concentration parameter for the Dirichlet distribution | `100`, `1.0`, `0.1` |
| `--augmentation` | Whether to use data augmentation (random crop and horizontal flip) | `1: True`, `0: False` |
| `--load_data_to_memory` | Whether to load the data to memory for reducing the overhead of data loader | `1: True`, `0: False` |
| `--sim` | Whether to use simulation mode with virtual clock | `1: True`, `0: False` |

### DP Arguments
| Argument | Description | Options |
| --- | --- | --- |
| `--dynamic_type` | Type of dynamic client participation | `static`, `round-robin`, `incremental-arrival`, `incremental-departure`, `random`, `markov` |
| `--round_start` | Round to start dynamic client participation | `int` |
| `--initial_clients` | Initial number of clients for `incremental-arrival` and `incremental-departure` | `int` |
| `--interval` | Interval of client change for `incremental-arrival` and `incremental-departure` | `int` |
| `--clients_per_interval` | Number of clients to change at each interval for `incremental-arrival` and `incremental-departure` | `int` |
| `--overlap_clients` | Number of overlapping clients between two adjacent intervals for `round-robin` | `int` |
| `--dpfl` | Whether to use Knowledge-Pool (KPFL) | `1: True`, `0: False` |

Sample usage:
```bash
python3 main.py --algorithm fedavg --skew_type label --alpha 100.0 --dynamic_type static
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

    def _test(self, model, testLoader, **kwargs):
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


## Contributing
Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request. For more information, see our [contribution guidelines](CONTRIBUTING.md).

## License
This project is licensed under the [MIT License](LICENSE).