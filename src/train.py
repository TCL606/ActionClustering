import os
import numpy as np
import torch
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from itertools import permutations

seed = 2024
np.random.seed(seed)
torch.manual_seed(seed)

X_train_file = "UCI HAR Dataset/train/X_train.txt"
Y_train_file = "UCI HAR Dataset/train/y_train.txt"

X_lst = []
with open(X_train_file, 'r') as fp:
    line = fp.readline()
    while line != "":
        X_lst.append(line.strip().split())
        line = fp.readline()

Y_lst = []
with open(Y_train_file, 'r') as fp:
    line = fp.readline()
    while line != "":
        Y_lst.append(line.strip())
        line = fp.readline()

X = torch.from_numpy(np.array(X_lst, dtype=np.float32))
Y = torch.from_numpy(np.array(Y_lst, dtype=np.int32))

class_num = 6

X_mean = torch.mean(X, dim=1)
X_std = torch.std(X, dim=1)

X_nml = (X - X_mean.unsqueeze(1)) / X_std.unsqueeze(1)

def KMeans(X, class_num, n_iters):
    indices = torch.randperm(X.size(0))[:class_num]
    centers = X[indices]

    for i in range(n_iters):
        dist = torch.sum((X.unsqueeze(1) - centers.unsqueeze(0)) ** 2, dim=-1)

        cluster_ids = dist.argmin(dim=1)

        # min_dist = torch.sum(dist[:, cluster_ids])

        new_centers = torch.stack([torch.mean(X[cluster_ids == k], dim=0) for k in range(class_num)])

        if torch.all(new_centers == centers):
            centers = new_centers
            break
        else:
            centers = new_centers

    return centers, cluster_ids

def cal_acc(pred, labels, num_classes=6):
    max_acc = 0
    best_perm = None
    
    for perm in permutations(range(num_classes)):
        perm = torch.tensor(perm)
        mapped_labels = perm[pred]
        acc = (mapped_labels == labels).float().mean().item()
        
        if acc > max_acc:
            max_acc = acc
            best_perm = perm
    
    return max_acc, best_perm


km_centers, km_cluster_ids = KMeans(X, class_num, 100)

agg = AgglomerativeClustering(n_clusters=6)
agg_cluster_ids = agg.fit_predict(X.numpy())
agg_cluster_ids = torch.from_numpy(agg_cluster_ids)

km_acc, km_perm = cal_acc(km_cluster_ids, Y)
agg_acc, agg_perm = cal_acc(agg_cluster_ids, Y)

print("KM Acc: ", km_acc)
print("AGG Acc: ", agg_acc)