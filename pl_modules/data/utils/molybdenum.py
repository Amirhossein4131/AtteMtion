# DATA


import numpy as np
import os
import json
from pathlib import Path
import random
import re
from time import sleep
from tqdm import tqdm
import warnings

import torch
from torch_geometric.data import Data, DataLoader
from torch.utils.data import random_split

from functools import lru_cache

from pymatgen.io.cif import CifParser
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Structure, Lattice, Site

import torch
import pytorch_lightning as pl

from torch_geometric.nn import global_mean_pool
from torch.optim import Adam
from torch.nn.functional import relu
from torch.nn import Module, MultiheadAttention, Linear
from torch_geometric.nn import global_mean_pool, GATConv
from torch.optim.lr_scheduler import StepLR

from transformers import GPT2Config, GPT2Model

DATASETS = {
    "Mo": os.path.join("data", "Mo"),
    "EFF": os.path.join("data", "EFF"),
    "EFF_train": os.path.join("data", "EFF", "train_gv"),
    "EFF_test": os.path.join("data", "EFF", "test"),
}


def gvector(gvector):
    with open(gvector, "rb") as binary_file:
        bin_version = int.from_bytes(binary_file.read(4),
                                     byteorder='little',
                                     signed=False)
        if bin_version != 0:
            print("Version not supported!")
            exit(1)
        # converting to int to avoid handling little/big endian
        flags = int.from_bytes(binary_file.read(2),
                               byteorder='little',
                               signed=False)
        n_atoms = int.from_bytes(binary_file.read(4),
                                 byteorder='little',
                                 signed=False)
        g_size = int.from_bytes(binary_file.read(4),
                                byteorder='little',
                                signed=False)
        payload = binary_file.read()
        data = np.frombuffer(payload, dtype='<f4')
        en = data[0]
        gvect_size = n_atoms * g_size
        spec_tensor = np.reshape((data[1:1 + n_atoms]).astype(np.int32),
                                 [1, n_atoms])
        gvect_tensor = np.reshape(data[1 + n_atoms:1 + n_atoms + gvect_size],
                                  [n_atoms, g_size])
    return (gvect_tensor)


def json_to_pmg_structure(db_name, json_file):
    """
    converts json files into cif format files
    """
    cif_path = os.path.join(os.environ['PROJECT_ROOT'], DATASETS[db_name], "cifs")

    json_path = os.path.join(os.environ['PROJECT_ROOT'], DATASETS[db_name], "jsons", json_file)

    Path(cif_path).mkdir(parents=True,
                         exist_ok=True)

    json_data = read_json(json_path)
    lattice_vectors = json_data["lattice_vectors"]
    lattice = Lattice(lattice_vectors)
    sites = [
        Site(species=atom[1], coords=atom[2], properties={"occupancy": 1.0})
        for atom in json_data["atoms"]
    ]
    cif_name = json_file.split(".")[0] + ".cif"
    structure = Structure(lattice=lattice, species=["Mo"] * len(sites), coords=[site.coords for site in sites])
    if os.path.isfile(cif_path + "/" + cif_name):
        pass
    else:
        structure.to(filename=cif_path + "/" + cif_name)
    return structure


def get_edge_indexes(structure):
    bonded_structure = CrystalNN(weighted_cn=True, distance_cutoffs=(10, 20.))
    bonded_structure = bonded_structure.get_bonded_structure(structure)
    bonded_structure = bonded_structure.as_dict()
    structure_graph = bonded_structure["graphs"]["adjacency"]

    # len(graph) = number of atoms
    edge_index_from = []
    edge_index_to = []
    edges = []
    for i in range(len(structure_graph)):
        # iterates over the connected atoms of each atom in the cell
        for j in range(len(structure_graph[i])):
            edge_index_from.append(i)
            edge_id = structure_graph[i][j]["id"]
            edge_index_to.append(edge_id)
            edge = torch.tensor(structure_graph[i][j]["to_jimage"])
            edges.append(edge)

    edge_index_from = torch.tensor(edge_index_from)
    edge_index_to = torch.tensor(edge_index_to)

    edge_indexes = torch.stack([edge_index_from, edge_index_to], dim=0)

    edges = torch.cat(edges, dim=0)
    return edge_indexes, edges


def read_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data


def get_db_keys(db_name):
    db_path = os.path.join(os.environ['PROJECT_ROOT'], DATASETS[db_name], "gvectors")
    # I made the path compatible with hydra, but it would be better to include the path in config
    keys = sorted([f.split(".")[0] for f in os.listdir(db_path) if os.path.isfile(os.path.join(db_path, f))], key=int)
    # print(keys)

    gvector_keys = []
    json_keys = []
    for item in keys:
        gvector_keys.append(item + ".bin")
        json_keys.append(item + ".example")

    return gvector_keys, json_keys


def get_element_type(example):
    json_file = read_json(example)
    element = json_file["atoms"][0][1]
    return element


def zero_pad(db_name, example, vector):
    db_info_path = os.path.join(os.environ['PROJECT_ROOT'], DATASETS[db_name], "elements.json")
    db_elements = read_json(db_info_path)
    chemical_element = get_element_type(example)
    zero_pad_length = len(db_elements)
    pre_pad_length = db_elements[chemical_element]
    post_pad_length = zero_pad_length - pre_pad_length
    pre_pad_zeros = np.zeros(pre_pad_length)
    post_pad_zeros = np.zeros(post_pad_length)

    padded_vector = []
    for i in range(len(vector)):
        v = np.concatenate((pre_pad_zeros, vector[i], post_pad_zeros))
        padded_vector.append(v)

    padded_vector = np.array(padded_vector)
    return padded_vector


def dataset(db_name, split="train"):
    # Parinello vectors
    if split is None:
        db_dir = db_name
    else:
        db_dir = f"{db_name}_{split}"

    db_path = os.path.join(os.environ['PROJECT_ROOT'], DATASETS[db_dir])
    gvect_keys, json_keys = get_db_keys(db_dir)

    set = []
    for i in range(len(gvect_keys[0:datapoint_limit])):
        a = gvector(db_path + "/" + "gvectors" + "/" + gvect_keys[i])
        a = zero_pad(db_name, db_path + "/" + "jsons" + "/" + json_keys[i], a)
        a = np.array(a, dtype='<f4')
        a = torch.tensor(a)
        set.append(a)
    parinello = set
    # edge indexes
    edge_indexes = []
    edges = []

    for item in tqdm(json_keys[0:datapoint_limit]):
        structure = json_to_pmg_structure(db_name=db_dir, json_file=item)
        ei, e = get_edge_indexes(structure)
        edge_indexes.append(ei)
        edges.append(e)

    return parinello, edge_indexes, edges


def get_labels(db_name, split="train"):
    """gets labels (energy, force, ...)"""
    if split is None:
        db_dir = db_name
    else:
        db_dir = f"{db_name}_{split}"

    label = []
    db_path = os.path.join(os.environ['PROJECT_ROOT'], DATASETS[db_dir], "jsons")
    gvect_keys, json_keys = get_db_keys(db_dir)

    for item in json_keys[0:10]:
        example = os.path.join(db_path, item)
        data = read_json(example)
        num_atoms = len(data["atoms"])
        toten = data["energy"][0]
        en_per_atom = toten / num_atoms
        label.append(en_per_atom)

    label = torch.tensor(label, dtype=torch.float)
    return label


def create_sequence_tensor(feature, seq_len):
    count = 0
    sequence = []
    num_batches = len(feature) // seq_len

    for batch in range(num_batches):
        sub_sequence = [feature[count + i] for i in range(seq_len)]
        count += seq_len
        sequence.append(sub_sequence)

    return sequence



def shuffle_list(list1, list2, list3, list4):
    combined_lists = list(zip(list1, list2, list3, list4))
    random.shuffle(combined_lists)
    shuffled_list1, shuffled_list2, shuffled_list3, shuffled_list4 = zip(*combined_lists)
    return shuffled_list1, shuffled_list2, shuffled_list3, shuffled_list4


def augment(l1, l2, l3, l4, aug_num, seq_len): # NEED TO ADD aug_nem and seq_len TO HYDRA
    """
    l1, l2 and l3 shall be Parinello vectors,
    edge indexes and edges to be augmented.
    """
    l1_new = []
    l2_new = []
    l3_new = []
    l4_new = []

    seq_num = int(len(l1) / seq_len)

    begin = 0
    end = seq_len

    for m in range(seq_num):
        # selects the sequence to be shuffled
        l11 = l1[begin: end]
        l22 = l2[begin: end]
        l33 = l3[begin: end]
        l44 = l4[begin: end]

        # inserts the original sequence into new lists
        for i in range(len(l11)):
            l1_new.append(l11[i])
            l2_new.append(l22[i])
            l3_new.append(l33[i])
            l4_new.append(l44[i])

        # inserts shuffled sequences to the list
        for j in range(aug_num):
            l1_sh, l2_sh, l3_sh, l4_sh = shuffle_list(l11, l22, l33, l44)
            for k in range(len(l1_sh)):
                l1_new.append(l1_sh[k])
                l2_new.append(l2_sh[k])
                l3_new.append(l3_sh[k])
                l4_new.append(l4_sh[k])

        begin += seq_len
        end += seq_len

    return l1_new, l2_new, l3_new, l4_new


def data(db_name, split="train", augmentation=True): # NEED TO ADD augmentation TO HYDRA
    """Create a PyTorch Geometric Data object"""
    warnings.filterwarnings("ignore")
    parinello, edge_indexes, edges = dataset(db_name=db_name, split=split)
    labels = get_labels(db_name, split=split)



    if split == "train" and augmentation is True:
        parinello_aug, edge_indexes_aug, edges_aug, labels_aug = augment(parinello, edge_indexes, edges,
                                                                         labels, aug_num=9, seq_len=10)
        db = []
        for i in range(len(parinello_aug)):
            data_point = Data(x=parinello_aug[i], edge_index=edge_indexes_aug[i], to_j=edges_aug[i], y=labels_aug[i])
            db.append(data_point)
        print(len(db))
    else:
        db = []
        for i in range(len(parinello)):
            data_point = Data(x=parinello[i], edge_index=edge_indexes[i], to_j=edges[i], y=labels[i])
            db.append(data_point)
        print(len(db))

    # Create a PyTorch Geometric DataLoader
    # dataset_size = len(db)
    # train_size = int(0.7 * dataset_size)
    # val_size = int(0.15 * dataset_size)
    # test_size = dataset_size - train_size - val_size
    # train_dataset, val_dataset, test_dataset = random_split(db, [train_size, val_size, test_size])

    return db
