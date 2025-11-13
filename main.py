import argparse
import datetime
import time
import nni
import json
import wandb
import torch
import random
import numpy as np
from colorama import Fore, Style
from server import FLserver
from client import FLclient
from utils.util import *
from utils.data_model import DataModel


parser = argparse.ArgumentParser("FL Dynamic-Client-Participation")
parser.add_argument("--dataset", default="Office-Caltech", type=str, help="Mnist, Cifar10, Digits, Office-Caltech")
parser.add_argument("--custom_data_path", default="./data/office_caltech_10/Amazon/", type=str, help="custom data directory")
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument("--weight_decay", default=1e-5, type=float)
parser.add_argument("--optimizer", default="sgd", type=str)
parser.add_argument("--val_ratio", default=0.2, type=float)
parser.add_argument("--model", default="ResNet10", type=str, help="SimpleCNN, MyCNN, ResNet10")
parser.add_argument("--num_clients", default=10, type=int, help="number of clients")
parser.add_argument("--num_classes", default=10, type=int, help="number of classes")
parser.add_argument("--num_domains", default=4, type=int, help="number of domains")
''' Non-IID setting  '''
parser.add_argument("--skew_type", default="label", type=str, help="label, feature, quantity")
parser.add_argument("--alpha", default=100.0, type=float, help="concentration level of Dirichlet distribution")
''' DCP setting '''
parser.add_argument("--algorithm", default="fedavg", type=str)
parser.add_argument("--num_rounds", default=100, type=int, help="number of rounds")
parser.add_argument("--num_epochs", default=5, type=int, help="number of local epochs")
parser.add_argument("--incremental_type", default="client-incremental", type=str, help="client-incremental or class-incremental")
''' Dynamic setting '''
parser.add_argument("--dynamic_type", default="static", type=str, help="static, incremental-arrival/departure, round-robin, random, markov")
parser.add_argument("--round_start", default=50, type=int, help="round to start incremental arrival/departure")
parser.add_argument("--initial_clients", default=5, type=int, help="initial number of clients for incremental arrival")
parser.add_argument("--interval", default=10, type=int, help="interval for client dynamics")
parser.add_argument("--clients_per_interval", default=1, type=int, help="number of clients to add/remove per interval in incremental arrival/departure")
parser.add_argument("--overlap_clients", default=1, type=int, help="number of overlapping clients in each round for round-robin")
parser.add_argument("--deadline_percentile", default=0.8, type=float, help="percentile of client deadlines to set as aggregation deadline (e.g., 0.8 for 80%)")
parser.add_argument("--k", default=1, type=float, help="number of standard deviations to use for setting the aggregation deadline")
parser.add_argument("--kpfl", action="store_true", help="Enable knowledge pool module for federated learning")
''' System setting '''
parser.add_argument("--figure_path", default="./figures/", type=str)
parser.add_argument("--log_dir", default="./log/runs/", type=str)
parser.add_argument("--augmentation", default=1, type=int, help="data augmentation or not")
parser.add_argument("--load_data_to_memory", default=1, type=int, help="load data to memory or not")
parser.add_argument("--cuda", default="cuda:0", type=str)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--nni", action="store_true", help="use nni or not")
parser.add_argument("--sim", action="store_true", help="use simulation or not")
args = parser.parse_args()

os.makedirs(args.figure_path, exist_ok=True)


def main(args):
    start_time = time.time()
    seed_everything(seed=args.seed)
    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    print(f"Using device: `{device}`, seed: `{args.seed}`")

    ''' Wandb for training logging '''
    group_id = "{ds}({algo}{lty}, {md}{kpfl})-{dynamic}, {skew}, nc={nc}-a={sa}".format(ds=args.dataset, algo=args.algorithm, lty=" (lty, {k})".format(k=args.k) if args.sim else "", md=args.model, kpfl=" (KPFL)" if args.kpfl else "", dynamic=args.dynamic_type, skew=args.skew_type, nc=args.num_clients, sa=args.alpha,)
    run = wandb.init(
        project="({ds}) FL ({skew}, a={al})".format(ds=args.dataset, skew=args.skew_type, al=args.alpha),
        name="{ds}({algo}{lty}, {md}{kpfl})-(DCP-{dynamic}, N.-{skew}) num_clients={nc}-alpha={sa}-seed={sd}, {time}"
            .format(ds=args.dataset, algo=args.algorithm, lty=" (lty, {k})".format(k=args.k) if args.sim else "", md=args.model, kpfl=" (KPFL)" if args.kpfl else "", dynamic=args.dynamic_type, skew=args.skew_type, nc=args.num_clients, sa=args.alpha, sd=args.seed, time=datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")),
        config=args,
        group=group_id,
        mode="disabled"
    )

    ''' Load data '''
    '''
    datasets = ["MNIST", "USPS", "SVHN", "SYN"]
        client_datasets = {
            0: {
                "MNIST": [...],
                "USPS": [...],
            },
            1: {
                "MNIST": [...],
                "USPS": [...],
            },
            ...
        }
        testsets = {
            "MNIST": [...],
            "USPS": [...],
        }
    '''
    client_datasets, testsets = DataModel().load_noniid_partition(dataset=args.dataset, args=args)

    ''' Hyperparameters & AutoML '''
    with open('./configs/hyper_config.json', 'r') as f:
        params_config = json.load(f)

    try:
        params = params_config[args.dataset][args.algorithm]
    except KeyError:
        params = {}
        print(Fore.RED + "No hyperparameters for the algorithm `{}` of dataset `{}`".format(args.algorithm, args.dataset) + Style.RESET_ALL)

    if args.kpfl:
        try:
            params = params_config[args.dataset][f"{args.algorithm}+kpfl"][args.dynamic_type]
        except KeyError:
            ''' Use default params if no specific dpfl params found '''
            params = params | params_config["Default"]
        print("Online rate, Offline rate (e):", params["online_rate"], params["offline_rate"])

    if args.nni:
        optimized_params = nni.get_next_parameter()
        if args.dpfl:
            params.update(optimized_params)
            print("Optimized parameters (age_decay):", params)
        else:
            params.update(optimized_params)
    print("Optimized parameters:", params, "\n")


    ''' Reset seed '''
    seed_everything(seed=args.seed)

    ''' Create FL Strategy instance '''
    strategy_class = load_strategy(strategy_name=args.algorithm, args=args)
    strategy = strategy_class(device=device, args=args, params=params)

    ''' Create ClientScheduler instance '''
    dp_class = load_dpmodel(dp_type=args.dynamic_type, args=args)
    dpModel = dp_class(args=args)
    # print(dynamic_scheduler.client_state)

    ''' Simulus for virtual clock simulation '''
    if args.sim:
        import simulus
        sim = simulus.simulator()
    else:
        sim = None

    ''' Create FL Server instance '''
    server = FLserver(clients=[], testset=testsets, strategy=strategy, device=device, params=params, args=args, sim=sim)

    ''' Create FL Clients instance'''
    for i in range(args.num_clients):
        client = FLclient(
                    cid=i, 
                    dataset=client_datasets[i], 
                    strategy=strategy, 
                    device=device,
                    args=args,
                    sim=sim,
                )
        server.add_client(client)

    ''' Initialization of server and clients '''
    server.initialization()

    ''' Start training '''
    for i in (range(1, args.num_rounds+1)):
        print("\n----------- Round {}/{} -----------".format(i, args.num_rounds))

        # Updates the server's and client's data if is under class-incremental learning
        if args.incremental_type == "class-incremental" and (i-1) % 20 == 0 and i > 1:
            server.update_incremental_data(rounds=i)

        # ----- Update the clients for the current round based on dynamic type -----
        active_ids, inactive_ids = dpModel.update(round=i)
        server.set_client_state(active_ids, inactive_ids)

        server.show_clients()
        # input("Next round (Press Enter to continue...)")
    
        server.train_clients(rounds=i)
        server.aggregate_clients(rounds=i)

        ''' Evaluate '''
        server.evaluate(rounds=i, model=server.model)

    if args.sim:
        print("\nTotal virtual time: {:.4f}s".format(sim.now))

    wandb.finish()
    print("Total time: {:.4f}s".format(time.time()-start_time))
    print("\n----------- Training complete! 🎉 -----------")


if __name__ == "__main__":
    main(args)