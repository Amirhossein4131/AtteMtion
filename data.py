# DATA


import numpy as np
import os
import json
from pathlib import Path
import re
from time import sleep
from tqdm import tqdm
import warnings

import torch
from torch_geometric.data import Data, DataLoader
from torch.utils.data import random_split



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
    "Mo": "./data/Mo"
}

def gvector (gvector):
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
                spec_tensor = np.reshape((data[1:1+n_atoms]).astype(np.int32),
                                     [1, n_atoms])
                gvect_tensor = np.reshape(data[1+n_atoms:1+n_atoms+gvect_size],
                                      [n_atoms, g_size])
    return (gvect_tensor)


def json_to_pmg_structure(db_name, json_file):
    """
    converts json files into cif format files
    """
    cif_path = os.path.join(DATASETS[db_name], 
                            "train_gv", "cifs")  
    
    json_path = os.path.join(DATASETS[db_name], 
                            "train_gv", "jsons", json_file) 
    
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
    bonded_structure = CrystalNN(weighted_cn=True, distance_cutoffs=(10,  20.))
    bonded_structure = bonded_structure.get_bonded_structure(structure)
    bonded_structure = bonded_structure.as_dict()
    structure_graph = bonded_structure["graphs"]["adjacency"]

    # len(graph) = number of atoms
    edge_index_from = []
    edge_index_to = []
    edges = []
    for i in range (len(structure_graph)):
        #iterates over the connected atoms of each atom in the cell
        for j in range(len(structure_graph[i])):
            edge_index_from.append(i)
            edge_id = structure_graph[i][j]["id"]
            edge_index_to.append(edge_id)
            edge = torch.tensor(structure_graph[i][j]["to_jimage"])
            edges.append(edge)

    edge_index_from = torch.tensor(edge_index_from)
    edge_index_to = torch.tensor(edge_index_to)

    edge_indexes = np.array([edge_index_from, edge_index_to])
    edge_indexes = torch.from_numpy(edge_indexes)

    edges = np.array(edges)
    edges = torch.from_numpy(edges)
    return edge_indexes, edges


def read_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data


def get_db_keys(db_name):
    db_path = os.path.join(DATASETS[db_name], "train_gv", "gvectors")
    keys = [f.split(".")[0] for f in os.listdir(db_path) if os.path.isfile(os.path.join(db_path, f))]

    gvector_keys = []
    json_keys = []
    for item in keys:
        gvector_keys.append(item+".bin")
        json_keys.append(item+".example")
                  
    return gvector_keys, json_keys



def dataset(db_name):
    # Parinello vectors
    db_path =  os.path.join(DATASETS[db_name], "train_gv", "gvectors")
    gvect_keys, json_keys = get_db_keys(db_name)
    set = []
    for item in gvect_keys[0:80]:
        a = gvector (db_path + "/" + item)
        a = torch.tensor(a)
        set.append(a)
    parinello = set

    # edge indexes
    edge_indexes = []
    edges = []

    for item in tqdm(json_keys[0:80]):
        structure = json_to_pmg_structure(db_name="Mo", json_file=item)
        ei, e = get_edge_indexes(structure)
        edge_indexes.append(ei)
        edges.append(e)
         
    return parinello, edge_indexes, edges


def get_labels(db_name):
     """gets labels (energy, force, ...)"""
     
     label = []
     db_path =  os.path.join(DATASETS[db_name], "train_gv", "jsons")
     gvect_keys, json_keys = get_db_keys(db_name)
     
     for item in json_keys[0:80]:
          example = os.path.join(db_path, item)
          data = read_json(example)
          num_atoms = len(data["atoms"])
          toten = data["energy"][0]
          en_per_atom = toten/num_atoms
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

def in_context_data(data_loader, batch_size):
    in_context_db = []
    for batch in data_loader:
        in_context_example = {
            "parinello": batch.x,
            "edge_index": batch.edge_index,
            "to_j": batch.to_j,
            "in_context_label": batch.batch,
            "label": batch.y, 
        }

        data = Data(x=in_context_example["parinello"], edge_index=in_context_example["edge_index"],
            to_j=in_context_example["to_j"], config_label=in_context_example["in_context_label"],
            y=in_context_example["label"])
    
        in_context_db.append(data)

    context_loader = DataLoader(in_context_db, batch_size=batch_size, shuffle=False)

    return context_loader


def data(db_name, sequence_size, batch_size):
    """Create a PyTorch Geometric Data object"""
    warnings.filterwarnings("ignore")
    parinello, edge_indexes, edges = dataset(db_name=db_name)
    labels = get_labels(db_name)

    db = []
    for i in range (len(parinello)):
        data = Data(x=parinello[i], edge_index=edge_indexes[i], to_j=edges[i], y=labels[i])
        db.append(data)

    # Create a PyTorch Geometric DataLoader
    batch_size = batch_size
    dataset_size = len(db)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(db, [train_size, val_size])

    t_loader = DataLoader(train_dataset, batch_size=sequence_size, shuffle=False)
    v_loader = DataLoader(val_dataset, batch_size=sequence_size, shuffle=False)
    
    train_loader = in_context_data(t_loader, batch_size=batch_size)
    val_loader = in_context_data(v_loader, batch_size=batch_size)

    return train_loader, val_loader
