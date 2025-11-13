from typing import List, Tuple, Dict, Any
from .fedavg import FedAvg
import numpy as np
from sklearn import metrics
import scipy.sparse as sp
import torch
from functools import reduce
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import warnings

'''
Hyperparameters (From paper):
    - dataset: random sample from the following domains (digits: 1%, office caltech: 20%)
        - Digits (MNIST, USPS, SVHN, SYN)
        - Office Caltech (Caltech, Amazon, Webcam, DSLR) | 10 overlapping classes between Office31 and Caltech-256
    - model: ResNet-10, feature_dim: 512
    - optimizer: SGD
    - lr: 0.01, momentum: 0.9, weight_decay: 1e-5
    - batch: 64
    - local_epoch: 10
    - communication_rounds: 100
    - num_clients:
        - Digits: 20 (MNIST:3, USPS: 7, SVHN: 6, SYN: 4)
        - Office Caltech: 10 (Caltech: 3, Amazon: 2, Webcam: 1, DSLR: 4)
    - temperature: 0.02
    - aggregation: weighted averaging
'''

class FPL(FedAvg):
    def __init__(self, device, args, params):
        super().__init__(device=device, args=args, params=params)
        self.local_protos = {}                                          # original local prototypes of FPL
        self.local_payload["type"] = "protos"                           # Type of payload (e.g., logits, prototypes)
        self.infoNCET = params["infoNCET"]


    def _initialization(self, **kwargs) -> None:
        pass


    def _server_train_func(self, cid, rounds, client_list, **kwargs) -> None:
        ''' Train function for the server to orchestrate the training process '''
        result = client_list[cid].train(global_protos=kwargs["global_payload"])
        train_loss, train_acc, ce_loss, proto_loss = result["train_loss"], result["train_acc"], result["ce_loss"], result["proto_loss"]
        self.local_protos[client_list[cid].cid] = client_list[cid].strategy.local_payload["data"]     # save local prototypes
        print("[Client {}] loss: {:.4f}, ce_loss: {:.4f}, proto_loss: {:.4f}, acc: {:.4f}".format(cid, train_loss, ce_loss, proto_loss, train_acc))


    def _server_agg_func(self, rounds, client_list, active_clients, global_model):
        ''' Aggregate function for the server '''
        new_weights = self._aggregation(client_list)

        self.global_payload = self.aggregate_protos(self.local_protos)
        self.local_protos = {}                                     # clear local_protos
        return new_weights


    def _train(self, model, trainLoader, optimizer, num_epochs, **kwargs) -> Tuple[float, float]:
        ''' Train function for the client '''
        global_protos = kwargs["global_protos"]

        if len(global_protos) != 0:
            all_global_protos_keys = np.array(list(global_protos.keys()))
            all_f = []
            mean_f = []
            for protos_key in all_global_protos_keys:
                temp_f = global_protos[protos_key]
                temp_f = torch.cat(temp_f, dim=0).to(self.device)
                all_f.append(temp_f.cpu())
                mean_f.append(torch.mean(temp_f, dim=0).cpu())
            all_f = [item.detach() for item in all_f]
            mean_f = [item.detach() for item in mean_f]

        model.train()
        for i in range(num_epochs):
            total_loss, total_correct = [], []
            ce_loss, proto_loss = [], []
            for x, labels in trainLoader:
                x, labels = x.to(self.device), labels.to(self.device)
                output, protos = model(x)

                predict = torch.argmax(output.data, 1)
                total_correct.append((predict == labels).sum().item() / len(predict))

                # loss1: cross-entropy loss
                lossCE = F.cross_entropy(output, labels)
                
                # loss2: prototype distance loss
                if len(global_protos) == 0:
                    loss_InfoNCE = 0 * lossCE
                else:
                    loss_InfoNCE = []
                    for idx, label in enumerate(labels):
                        if label.item() in global_protos.keys():
                            f_now = protos[idx].unsqueeze(0)
                            loss_instance = self.hierarchical_info_loss(f_now, label, all_f, mean_f, all_global_protos_keys)    
                            loss_InfoNCE.append(loss_instance)

                    if len(loss_InfoNCE) == 0:
                        loss_InfoNCE = 0 * lossCE
                    else:
                        loss_InfoNCE = torch.mean(torch.stack(loss_InfoNCE, dim=0))
                
                loss = lossCE + loss_InfoNCE
                ce_loss.append(lossCE.item())
                proto_loss.append(loss_InfoNCE.item())
                total_loss.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)               # clip the gradient to prevent exploding
                optimizer.step()

        # generate local ptototypes
        model.eval()
        agg_protos_label = {}
        with torch.no_grad():
            for x, labels in trainLoader:
                x, labels = x.to(self.device), labels.to(self.device)
                _, protos = model(x)
                for k in range(len(labels)):
                    if labels[k].item() in agg_protos_label:
                        agg_protos_label[labels[k].item()].append(protos[k,:])
                    else:
                        agg_protos_label[labels[k].item()] = [protos[k,:]]

        self.local_payload["data"] = self.agg_func(agg_protos_label)
        # return np.mean(total_loss), np.mean(total_correct), np.mean(ce_loss), np.mean(proto_loss)
        return {
            "train_loss": np.mean(total_loss),
            "train_acc": np.mean(total_correct),
            "ce_loss": np.mean(ce_loss),
            "proto_loss": np.mean(proto_loss)
        }


    def _test(self, model, testLoader, payload=None) -> Tuple[float, float]:
        ''' Test function for the client '''
        total_loss, total_correct = [], []
        num_classes = self.args.num_classes
        correct_predictions = {i: 0 for i in range(num_classes)}
        total_counts = {i: 0 for i in range(num_classes)}

        model.eval()
        with torch.no_grad():
            for x, label in testLoader:
                x, label = x.to(self.device), label.to(self.device)
                output, _ = model(x)
                predict = torch.argmax(output.data, 1)
                total_correct.append((predict == label).sum().item() / len(predict))
                total_loss.append(F.cross_entropy(output, label).item())

                for i in range(num_classes):
                    mask = label == i
                    correct_predictions[i] += (predict[mask] == label[mask]).sum().item()
                    total_counts[i] += mask.sum().item()

        class_acc = {i: (correct_predictions[i] / total_counts[i]) if total_counts[i] > 0 else 0 for i in range(num_classes)}
        # return np.mean(total_loss), np.mean(total_correct), class_acc
        return {
            "test_loss": np.mean(total_loss),
            "test_acc": np.mean(total_correct),
            "class_acc": class_acc
        }


    def agg_func(self, protos):
        """
        Returns the average of the weights.
        """
        for [label, proto_list] in protos.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for i in proto_list:
                    proto += i.data
                protos[label] = proto / len(proto_list)
            else:
                protos[label] = proto_list[0]

        return protos
    
    
    def aggregate_protos(self, local_protos_list):
        agg_protos_label = dict()
        for idx in local_protos_list:
            local_protos = local_protos_list[idx]
            for label in local_protos.keys():
                if label in agg_protos_label:
                    agg_protos_label[label].append(local_protos[label])
                else:
                    agg_protos_label[label] = [local_protos[label]]
        for [label, proto_list] in agg_protos_label.items():
            if len(proto_list) > 1:
                proto_list = [item.squeeze(0).detach().cpu().numpy().reshape(-1) for item in proto_list]
                proto_list = np.array(proto_list)

                c, num_clust, req_c = FINCH(proto_list, initial_rank=None, req_clust=None, distance='cosine',
                                            ensure_early_exit=False, verbose=True)

                m, n = c.shape
                class_cluster_list = []
                for index in range(m):
                    class_cluster_list.append(c[index, -1])

                class_cluster_array = np.array(class_cluster_list)
                uniqure_cluster = np.unique(class_cluster_array).tolist()
                agg_selected_proto = []

                for _, cluster_index in enumerate(uniqure_cluster):
                    selected_array = np.where(class_cluster_array == cluster_index)
                    selected_proto_list = proto_list[selected_array]
                    proto = np.mean(selected_proto_list, axis=0, keepdims=True)

                    agg_selected_proto.append(torch.tensor(proto))
                agg_protos_label[label] = agg_selected_proto
            else:
                agg_protos_label[label] = [proto_list[0].unsqueeze(0)]

        return agg_protos_label
    

    def payload_aggregation(self, local_payload_list, client_weight_list):
        agg_protos_label = dict()
        for idx in local_payload_list:
            local_protos = local_payload_list[idx]
            w = client_weight_list[idx]
            for label in local_protos.keys():
                if label in agg_protos_label:
                    agg_protos_label[label].append((local_protos[label], w))
                else:
                    agg_protos_label[label] = [(local_protos[label], w)]

        for [label, proto_list] in agg_protos_label.items():
            if len(proto_list) > 1:
                sum_w = sum([i[1] for i in proto_list])
                proto_list = [(item * w / sum_w).squeeze(0).detach().cpu().numpy().reshape(-1) for item, w in proto_list]
                proto_list = np.array(proto_list)

                c, num_clust, req_c = FINCH(proto_list, initial_rank=None, req_clust=None, distance='cosine',
                                            ensure_early_exit=False, verbose=True)

                m, n = c.shape
                class_cluster_list = []
                for index in range(m):
                    class_cluster_list.append(c[index, -1])

                class_cluster_array = np.array(class_cluster_list)
                uniqure_cluster = np.unique(class_cluster_array).tolist()
                agg_selected_proto = []

                for _, cluster_index in enumerate(uniqure_cluster):
                    selected_array = np.where(class_cluster_array == cluster_index)
                    selected_proto_list = proto_list[selected_array]
                    proto = np.mean(selected_proto_list, axis=0, keepdims=True)

                    agg_selected_proto.append(torch.tensor(proto))
                agg_protos_label[label] = agg_selected_proto
            else:
                agg_protos_label[label] = [proto_list[0][0].unsqueeze(0)]

        return agg_protos_label


    def hierarchical_info_loss(self, f_now, label, all_f, mean_f, all_global_protos_keys):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            # print(label, all_global_protos_keys)
            # for i in all_f:
            #     print(i, i.shape)

            f_pos = np.array(all_f, dtype=object)[all_global_protos_keys == label.item()][0].to(self.device)
            tmp_neg = np.array(all_f, dtype=object)[all_global_protos_keys != label.item()]
            if len(tmp_neg) == 0:
                f_neg = f_now
            else:
                f_neg = torch.cat(list(np.array(all_f, dtype=object)[all_global_protos_keys != label.item()])).to(self.device)
            xi_info_loss = self.calculate_infonce(f_now, f_pos, f_neg)

            mean_f_pos = np.array(mean_f, dtype=object)[all_global_protos_keys == label.item()][0].to(self.device)
            mean_f_pos = mean_f_pos.view(1, -1)
            # mean_f_neg = torch.cat(list(np.array(mean_f)[all_global_protos_keys != label.item()]), dim=0).to(self.device)
            # mean_f_neg = mean_f_neg.view(9, -1)

            cu_info_loss = F.mse_loss(f_now, mean_f_pos).to(self.device)

            hierar_info_loss = xi_info_loss + cu_info_loss
        return hierar_info_loss


    def calculate_infonce(self, f_now, f_pos, f_neg):
        f_proto = torch.cat((f_pos, f_neg), dim=0)
        l = torch.cosine_similarity(f_now, f_proto, dim=1).to(self.device)
        l = l / self.infoNCET

        exp_l = torch.exp(l)
        exp_l = exp_l.view(1, -1)
        pos_mask = [1 for _ in range(f_pos.shape[0])] + [0 for _ in range(f_neg.shape[0])]
        pos_mask = torch.tensor(pos_mask, dtype=torch.float, device=self.device)
        pos_mask = pos_mask.view(1, -1)
        # pos_l = torch.einsum('nc,ck->nk', [exp_l, pos_mask])
        pos_l = exp_l * pos_mask
        sum_pos_l = pos_l.sum(1)
        sum_exp_l = exp_l.sum(1)
        infonce_loss = -torch.log(sum_pos_l / sum_exp_l)
        return infonce_loss
    

    def _get_local_payload(self) -> Dict[str, Any]:
        """
        Get the current payload from the strategy.
        """
        return self.local_payload
    



# ------------ FINCH Clustering Code -----------------
try:
    from pynndescent import NNDescent

    pynndescent_available = True
except Exception as e:
    warnings.warn('pynndescent not installed: {}'.format(e))
    pynndescent_available = False
    pass

ANN_THRESHOLD = 70000


def clust_rank(mat, initial_rank=None, distance='cosine'):
    s = mat.shape[0]
    if initial_rank is not None:
        orig_dist = []
    elif s <= ANN_THRESHOLD:
        orig_dist = metrics.pairwise.pairwise_distances(mat, mat, metric=distance)
        np.fill_diagonal(orig_dist, 1e12)
        initial_rank = np.argmin(orig_dist, axis=1)
    else:
        if not pynndescent_available:
            raise MemoryError("You should use pynndescent for inputs larger than {} samples.".format(ANN_THRESHOLD))
        print('Using PyNNDescent to compute 1st-neighbours at this step ...')

        knn_index = NNDescent(
            mat,
            n_neighbors=2,
            metric=distance,
        )

        result, orig_dist = knn_index.neighbor_graph
        initial_rank = result[:, 1]
        orig_dist[:, 0] = 1e12
        print('Step PyNNDescent done ...')

    # The Clustering Equation
    A = sp.csr_matrix((np.ones_like(initial_rank, dtype=np.float32), (np.arange(0, s), initial_rank)), shape=(s, s))
    A = A + sp.eye(s, dtype=np.float32, format='csr')
    A = A @ A.T

    A = A.tolil()
    A.setdiag(0)
    return A, orig_dist


def get_clust(a, orig_dist, min_sim=None):
    if min_sim is not None:
        a[np.where((orig_dist * a.toarray()) > min_sim)] = 0

    num_clust, u = sp.csgraph.connected_components(csgraph=a, directed=True, connection='weak', return_labels=True)
    return u, num_clust


def cool_mean_old(M, u):
    _, nf = np.unique(u, return_counts=True)
    idx = np.argsort(u)
    M = M[idx, :]
    M = np.vstack((np.zeros((1, M.shape[1])), M))

    np.cumsum(M, axis=0, out=M)
    cnf = np.cumsum(nf)
    nf1 = np.insert(cnf, 0, 0)
    nf1 = nf1[:-1]

    M = M[cnf, :] - M[nf1, :]
    M = M / nf[:, None]
    return M


def cool_mean(M, u):
    s = M.shape[0]
    un, nf = np.unique(u, return_counts=True)
    umat = sp.csr_matrix((np.ones(s, dtype='float32'), (np.arange(0, s), u)), shape=(s, len(un)))
    return (umat.T @ M) / nf[..., np.newaxis]


def get_merge(c, u, data):
    if len(c) != 0:
        _, ig = np.unique(c, return_inverse=True)
        c = u[ig]
    else:
        c = u

    mat = cool_mean(data, c)
    return c, mat


def update_adj(adj, d):
    # Update adj, keep one merge at a time
    idx = adj.nonzero()
    v = np.argsort(d[idx])
    v = v[:2]
    x = [idx[0][v[0]], idx[0][v[1]]]
    y = [idx[1][v[0]], idx[1][v[1]]]
    a = sp.lil_matrix(adj.get_shape())
    a[x, y] = 1
    return a


def req_numclust(c, data, req_clust, distance):
    iter_ = len(np.unique(c)) - req_clust
    c_, mat = get_merge([], c, data)
    for i in range(iter_):
        adj, orig_dist = clust_rank(mat, initial_rank=None, distance=distance)
        adj = update_adj(adj, orig_dist)
        u, _ = get_clust(adj, [], min_sim=None)
        c_, mat = get_merge(c_, u, data)
    return c_


def FINCH(data, initial_rank=None, req_clust=None, distance='cosine', ensure_early_exit=True, verbose=True):
    """ FINCH clustering algorithm.
    :param data: Input matrix with features in rows.
    :param initial_rank: Nx1 first integer neighbor indices (optional).
    :param req_clust: Set output number of clusters (optional). Not recommended.
    :param distance: One of ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'] Recommended 'cosine'.
    :param ensure_early_exit: [Optional flag] may help in large, high dim datasets, ensure purity of merges and helps early exit
    :param verbose: Print verbose output.
    :return:
            c: NxP matrix where P is the partition. Cluster label for every partition.
            num_clust: Number of clusters.
            req_c: Labels of required clusters (Nx1). Only set if `req_clust` is not None.

    The code implements the FINCH algorithm described in our CVPR 2019 paper
        Sarfraz et al. "Efficient Parameter-free Clustering Using First Neighbor Relations", CVPR2019
         https://arxiv.org/abs/1902.11266
    For academic purpose only. The code or its re-implementation should not be used for commercial use.
    Please contact the author below for licensing information.
    Copyright
    M. Saquib Sarfraz (saquib.sarfraz@kit.edu)
    Karlsruhe Institute of Technology (KIT)
    """
    # Cast input data to float32
    data = data.astype(np.float32)

    min_sim = None
    adj, orig_dist = clust_rank(data, initial_rank, distance)
    initial_rank = None
    group, num_clust = get_clust(adj, [], min_sim)
    c, mat = get_merge([], group, data)

    # if verbose:
    #     print('Partition 0: {} clusters'.format(num_clust))

    if ensure_early_exit:
        if orig_dist.shape[-1] > 2:
            min_sim = np.max(orig_dist * adj.toarray())

    exit_clust = 2
    c_ = c
    k = 1
    num_clust = [num_clust]

    while exit_clust > 1:
        adj, orig_dist = clust_rank(mat, initial_rank, distance)
        u, num_clust_curr = get_clust(adj, orig_dist, min_sim)
        c_, mat = get_merge(c_, u, data)

        num_clust.append(num_clust_curr)
        c = np.column_stack((c, c_))
        exit_clust = num_clust[-2] - num_clust_curr

        if num_clust_curr == 1 or exit_clust < 1:
            num_clust = num_clust[:-1]
            c = c[:, :-1]
            break

        # if verbose:
        #     print('Partition {}: {} clusters'.format(k, num_clust[k]))
        k += 1

    if req_clust is not None:
        if req_clust not in num_clust:
            ind = [i for i, v in enumerate(num_clust) if v >= req_clust]
            req_c = req_numclust(c[:, ind[-1]], data, req_clust, distance)
        else:
            req_c = c[:, num_clust.index(req_clust)]
    else:
        req_c = None

    return c, num_clust, req_c