from argparse import ArgumentParser
from typing import Dict, List
from collections import OrderedDict
import os
import random
import pkgutil
import importlib
import inspect
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Dataset, RandomSampler
import strategies, dpmodels
from sklearn.manifold import TSNE
from torch.utils.data import ConcatDataset
from datetime import timedelta

sns.set_theme()


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def format_sim_time(seconds: float):
    t = str(timedelta(seconds=round(seconds, 3)))
    num_strips = len(t.split("."))
    if num_strips == 1:
        return t + ".000"
    else:
        return t[:-3]  # strip microseconds


def truncated_normal(mean, std, lower=0, upper=np.inf, size=1):
    a, b = (lower - mean) / std, (upper - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)


def create_accuracy_loss_plot(acc_list, loss_list):
    fig, ax1 = plt.subplots()
    ax1.plot(acc_list, label="Accuracy", color="red")
    ax2 = ax1.twinx()
    ax2.plot(loss_list, label="Loss", color="blue")
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy')
    ax2.set_ylabel('Loss')
    fig.legend()
    plt.close(fig)  # Important for Gradio
    return fig


def random_sampler(seed: int, dataset: Dataset, args: ArgumentParser):
    sample_len = int(len(dataset) * args.sample_ratio)
    dataset, _ = random_split(dataset, [sample_len, len(dataset) - sample_len], torch.Generator().manual_seed(seed))
    # sampler = RandomSampler(dataset, num_samples=int(args.sample_ratio * len(dataset)))
    dataLoader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataLoader


def load_strategy(strategy_name: str, args: ArgumentParser):
    ''' Load the strategy class '''
    try:
        module = importlib.import_module(f"strategies.{strategy_name}")
        strategy_class = [cls for name, cls in inspect.getmembers(module, inspect.isclass) if cls.__module__ == module.__name__][0]
        # strategy = strategy_class(device=device, args=args, params=params)
    except ModuleNotFoundError:
        raise ValueError(f"Strategy `{strategy_name}` not found.")
    return strategy_class


def list_available_strategies():
    return [name for _, name, _ in pkgutil.iter_modules(strategies.__path__)]


def load_dpmodel(dp_type: str, args: ArgumentParser):
    ''' Load the DPModel class '''
    try:
        module = importlib.import_module(f"dpmodels.{dp_type}")
        dp_class = [cls for name, cls in inspect.getmembers(module, inspect.isclass) if cls.__module__ == module.__name__][0]
        # dpModel = dp_class(args=args)
    except ModuleNotFoundError:
        raise ValueError(f"DP Model `{dp_type}` not found.")
    return dp_class


def list_available_dpmodels():
    return [name for _, name, _ in pkgutil.iter_modules(dpmodels.__path__)]


def plot_distribution(distribution, alpha, mode="label", args=None):
    # save filename using seed number & alpha
    path = "./distributions/{}/".format(mode)
    filename = f"{mode}_{alpha}_{args.seed}.png"                # e.g., label_100_42.png

    if not os.path.exists(path):
        os.makedirs(path)

    mode = mode.capitalize()
    # figname = "./distributions/{}-distribution_{}.png".format(mode, alpha)
    figname = os.path.join(path, filename)

    plt.rcParams.update({'font.size': 16})  # Adjust the value as needed
    if mode == "Quantity":
        distribution = distribution.squeeze()
        fig = plt.figure(figsize=(16, 9))
        num_elements = len(distribution)
        plt.bar(range(1, num_elements + 1), distribution, tick_label=[f'{i}' for i in range(1, num_elements + 1)])
        plt.xlabel("Clients")
        for i, v in enumerate(distribution):
            plt.text(i+1, v + 0.01, f'{v:.5f}', ha='center')
    else:
        # print(int(len(distribution)/5))
        fig, ax = plt.subplots(2, 5, figsize=(16, 9))
        for i in range(len(distribution)):
            num_elements = len(distribution[i])
            ax[i // 5][i % 5].set_title(f'Client {i}')
            ax[i // 5][i % 5].bar(range(1, num_elements + 1), distribution[i], tick_label=[f'{i}' for i in range(1, num_elements + 1)])
            ax[i // 5][i % 5].set_xlabel(mode + "s")
            ax[i // 5][i % 5].set_ylim(0, 1)
        # ax[i // 5][i % 5].set_ylabel('Proportion of Data')

    fig.suptitle('Simulation of {} Distribution Skew (α={}, seed={})'.format(mode, alpha, args.seed))
    fig.tight_layout(pad=1.0)
    plt.savefig(figname)
    plt.close()
    # plt.show()
    print("-- Save figure: {}".format(figname))


def bubble_plot_distribution(distribution, alpha, mode="label", args=None):
    # save filename using seed number & alpha
    path = "./distributions/bubble_plot/{}/".format(mode)
    filename = f"{mode}_{alpha}_{args.seed}.png"                # e.g., label_100_42.png

    if not os.path.exists(path):
        os.makedirs(path)

    figname = os.path.join(path, filename)

    fig = plt.figure(figsize=(6, 5))
    color = sns.color_palette()[0]
    sns.set_style("ticks")

    # plt.rcParams.update({'font.size': 16})  # Adjust the value as needed
    num_categories = len(distribution[0])
    x = np.arange(0, args.num_clients)
    for i in range(num_categories):
        y = np.array([i] * args.num_clients)
        size = distribution[:, i]
        plt.scatter(x, y, s=size*1000, color=color)
    
    plt.xlabel("Client ID", fontsize=20, labelpad=10)
    plt.ylabel("Class", fontsize=20, labelpad=10)
    plt.xticks(np.arange(args.num_clients), fontsize=14)
    plt.yticks(np.arange(num_categories), fontsize=14)
    plt.grid(True, which='both', linestyle='--')
    # plt.title('Simulation of {} Distribution Skew (α={}, seed={})'.format(mode, alpha, args.seed))
    fig.tight_layout()
    plt.savefig(figname)
    plt.close()
    print("-- Save figure: {}".format(figname))


def set_param(model, parameters: List[np.ndarray]):
    ''' Set the parameters of the model '''
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def measure_model_similarity(model, prev_model):
    ''' Measure the similarity between the current model and the previous model '''
    model_params = [(n, p.detach()) for n, p in model.named_parameters()]
    prev_model_params = [(n, p.detach()) for n, p in prev_model.named_parameters()]

    diff = 0
    diff_classifer = 0
    cnt1, cnt2 = 0, 0
    for (n1, p1), (n2, p2) in zip(model_params, prev_model_params):
        if "classifier" in n1:
            diff_classifer += torch.linalg.norm(p1 - p2)
            cnt2 += 1
        else:
            diff += torch.linalg.norm(p1 - p2)
            cnt1 += 1   
    return diff.item() / cnt1, diff_classifer.item() / cnt2


def plot_diff_scores(diff_scores, diff_classifier_scores, args):
    ''' Plot the differences between the current model and the previous model '''
    path = f"./figures/{args.skew_type}/"
    if not os.path.exists(path):
        os.makedirs(path)

    figname = os.path.join(path, f"diff_{args.dynamic_type}_{args.alpha}.png")
    fig = plt.figure(figsize=(16, 9))

    y_max = max(max(diff_scores), max(diff_classifier_scores))

    # subplot 1
    plt.subplot(1, 2, 1)
    plt.plot(diff_scores)
    plt.xlabel("Communication Round")
    plt.ylabel("Difference Score (l2-norm)")
    plt.title("Feature extractor")
    plt.ylim(0, y_max)

    plt.subplot(1, 2, 2)
    plt.plot(diff_classifier_scores)
    plt.xlabel("Communication Round")
    plt.ylabel("Difference Score (l2-norm)")
    plt.title("Classifier")
    plt.ylim(0, y_max)
    
    plt.suptitle(f"Discrepancy between the global model at each round ({args.dynamic_type}, {args.alpha})")
    plt.savefig(figname)
    print("-- Save figure: {}".format(figname))


def get_features(model, dataLoader, device):
    ''' Visualize the feature embeddings of the feature extractor '''
    feature_list = []
    label_list = []

    model.eval()
    with torch.no_grad():
        for name, loader in dataLoader.items():
            for x, label in loader:
                x, label = x.to(device), label.to(device)
                _, feature = model(x)
                feature_list.append(feature)
                label_list.extend(label.cpu().numpy())
    
    feature_list = torch.cat(feature_list, dim=0).cpu().numpy()
    # label_list = torch.cat(label_list, dim=0).cpu().numpy()
    return feature_list, label_list


def feature_visualization(model, dataLoader, global_protos, args):
    ''' Visualize the features extracted by the model '''
    plt.figure(figsize=(8, 6))
    fig_name = f"./figures/feature-{args.algorithm}_{args.dynamic_type}_{args.alpha}.png"
    
    features, labels = get_features(model, dataLoader, args.cuda)
    tsne = TSNE(n_components=2, random_state=0)

    if global_protos is not None:
        label, protos = np.array([l for l in global_protos.keys()]), np.array([global_protos[key][0].cpu().numpy() for key in global_protos])
        labels = np.concatenate((labels, label), axis=0)
        features = np.concatenate((features, protos), axis=0)

    X_2d = tsne.fit_transform(features)

    x_min, x_max = X_2d.min(0), X_2d.max(0)
    X_norm = (X_2d - x_min) / (x_max - x_min)

    scatter = plt.scatter(X_norm[:-10, 0], X_norm[:-10, 1], c=labels[:-10], cmap=plt.cm.get_cmap("tab10", 10), s=20)

    if global_protos is not None:
        plt.scatter(X_norm[-10:, 0], X_norm[-10:, 1], c=labels[-10:], marker="x", s=100, cmap=plt.cm.get_cmap("tab10", 10))

    plt.title(f"Feature visualization ({args.algorithm}, {args.dynamic_type}, α={args.alpha})")
    plt.legend(*scatter.legend_elements(), loc="center right", title="Classes", bbox_to_anchor=(1.15, 0.5))
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close()
    print("-- Save figure: {}".format(fig_name))