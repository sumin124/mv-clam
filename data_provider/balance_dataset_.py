import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import optim
from torch_geometric.data import Batch

from tqdm import tqdm

from lavis.common.utils import is_url
from lavis.models.base_model import BaseModel
from lavis.models.blip2_models.Qformer import BertLMHeadModel

from transformers import AutoTokenizer, AutoModel, BertConfig

import numpy as np
import pandas as pd
import pickle

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

from unicore.data import Dictionary, data_utils
from scipy.spatial import distance_matrix

from data_provider.d2_dataset import *

from functools import lru_cache

from torch_geometric.data import InMemoryDataset

from torch.utils.data import Dataset
import os
from data_provider.unimol_dataset_ import D3Dataset_cid, D3Dataset_index
from datasets import load_dataset, Features, Value
import math
import numpy as np




class SubDataset(Dataset):
    def __init__(self, conformations, instructions): 
        super(SubDataset, self).__init__()
        self.conformations = conformations
        self.instructions = instructions

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, index):
        data = self.instructions[index]
        id = data['input']
        output = data['output']
        task = data['task']
        instruction = data['instruction']
        atom_vec, coordinates, edge_type, dist, smiles, data_2d= self.conformations[id][:6]

        return atom_vec, coordinates, edge_type, dist, smiles, data_2d, instruction, output, task


class PubchemCapDataset(Dataset):
    def __init__(self, path, unimol_dict, max_atoms):
        super(PubchemCapDataset, self).__init__()
        self.dataset = D3Dataset_cid(path, unimol_dict, max_atoms)
        self.cid_list = [cid.decode() for cid in self.dataset.lmdb_dataset._keys]
        self.instruction = 'Describe the input molecule.'
        self.task = 'Caption'

    def __len__(self):
        return len(self.cid_list)

    def __getitem__(self, index):
        atom_vec, coordinates, edge_type, dist, smiles, data_2d, description, enriched_description = self.dataset[self.cid_list[index]]
        output = description

        return atom_vec, coordinates, edge_type, dist, smiles, data_2d, self.instruction, output, self.task

class BalanceDataset(Dataset):
    def __init__(self, root, mode, unimol_dict=None, max_atoms=256, train_datas=None):
        super(BalanceDataset, self).__init__()

        features = Features({"instruction": Value("string"),
                             "input": Value("string"),
                             "output": Value("string"),
                             "task": Value("string")})

        self.train_datas = train_datas

        if mode.find('train') >= 0:
            ### load 3D computed properties instructions
            d3_data_paths = []
            for split in ['train']:
                d3_data_paths.append(root + '/pubchemqc/' + split + '/3d_computed_properties_unit.json')
            d3_dataset_instructions = load_dataset("json", data_files=d3_data_paths, features=features)['train']

            ### load PubChemQC 3D conformations
            target_path = root + '/pubchemqc/pubchemqc_database.lmdb'  
            pubchemqc_3d_conformation = D3Dataset_index(target_path, unimol_dict, max_atoms) 

            
            ### Form PubChemQC 3D Subset
            pubchemqc_3d_subset = SubDataset(pubchemqc_3d_conformation, d3_dataset_instructions)


            ### load 2D computed properties instructions
            d2_com_data_paths = []
            for split in ['train']:
                d2_com_data_paths.append(root + '/pubchem/' + split + '/2d_computed_properties.json')  
            d2_com_dataset_instructions = load_dataset("json", data_files=d2_com_data_paths, features=features)['train']

            ### load PubChem 3D conformations
            target_path = root + '/pubchem/3d-pubchem-all.lmdb'
            pubchem_3d_conformation = D3Dataset_cid(target_path, unimol_dict, max_atoms)

            ### Form PubChem 3D Computed Property Subset
            pubchem_3d_com_subset = SubDataset(pubchem_3d_conformation, d2_com_dataset_instructions)

            ### load 3D computed properties instructions
            d2_des_data_paths = []
            for split in ['train']:
                d2_des_data_paths.append(root + '/pubchem/' + split + '/2d_descriptive_properties.json') 
            d2_des_dataset_instructions = load_dataset("json", data_files=d2_des_data_paths, features=features)['train']

            ### Form PubChem 3D Descriptive Subset
            pubchem_3d_des_subset = SubDataset(pubchem_3d_conformation, d2_des_dataset_instructions)

            ### Form PubChem 3D Caption Subset
            target_path = root + '/pubchem/' + 'train' +'/3d-pubchem.lmdb' 
            pubchem_3d_cap_subset = PubchemCapDataset(target_path, unimol_dict, max_atoms)

            '''
            ADJUST THE NUMBER OF DATASET TO USE
            '''
            # self.datasets = [pubchemqc_3d_subset, pubchem_3d_com_subset, pubchem_3d_des_subset, pubchem_3d_cap_subset]
            # self.datasets = [pubchemqc_3d_subset, pubchem_3d_com_subset, pubchem_3d_des_subset] #### exclude captioning task
            self.datasets = [pubchemqc_3d_subset] #### additional training on 3d_subset
            # self.datasets = [pubchem_3d_com_subset] #### additional training on 3d_com_subset
            # self.datasets = [pubchem_3d_des_subset] #### additional training on des_subset


            self.rt_lengths = [len(d) ** 0.25 for d in self.datasets]
            self.probs = [length / sum(self.rt_lengths) for length in self.rt_lengths]
        else:
            raise NotImplementedError

        self.prompt = "Below is an instruction that describes a task, paired with an input molecule. Write a response that appropriately completes the request.\n" \
                      "Instruction: {}\n" \
                      "Input molecule: {} <mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol>.\n" \
                      "Response: "
    
    def __len__(self):
        return sum([len(dataset) for dataset in self.datasets])

    def __getitem__(self, index):
        dataset_index = np.random.choice(len(self.datasets), p=self.probs)
        dataset = self.datasets[dataset_index]
        index = torch.randint(len(dataset), size=(1,)).item()
        atom_vec, coordinates, edge_type, dist, smiles, data_2d, instruction, output, task = dataset[index] ######### 

        if task in ['Description']:
            input = self.prompt.format(instruction[:64], smiles[:96])
        else:
            input = self.prompt.format(instruction, smiles[:96])

        return (atom_vec, coordinates, edge_type, dist, smiles), input, output, task, data_2d##############

class UniformDataset(Dataset):
    def __init__(self, root, mode, unimol_dict=None, max_atoms=256):
        super(UniformDataset, self).__init__()

        features = Features({"instruction": Value("string"),
                             "input": Value("string"),
                             "output": Value("string"),
                             "task": Value("string")})

        d3_data_paths = []
        d3_data_paths.append(root + '/pubchemqc/' + mode + '/3d_computed_properties_unit.json')
        d3_dataset_instructions = load_dataset("json", data_files=d3_data_paths, features=features)['train']

        ### load PubChemQC 3D conformations
        target_path = root + '/pubchemqc/pubchemqc_database.lmdb'
        pubchemqc_3d_conformation = D3Dataset_index(target_path, unimol_dict, max_atoms)

        ### Form PubChemQC 3D Subset
        pubchemqc_3d_subset = SubDataset(pubchemqc_3d_conformation, d3_dataset_instructions)

        ### load 2D computed properties instructions
        d2_com_data_paths = []
        d2_com_data_paths.append(root + '/pubchem/' + mode + '/2d_computed_properties.json')
        d2_com_dataset_instructions = load_dataset("json", data_files=d2_com_data_paths, features=features)['train']

        ### load PubChem 3D conformations
        target_path = root + '/pubchem/3d-pubchem-all.lmdb'
        pubchem_3d_conformation = D3Dataset_cid(target_path, unimol_dict, max_atoms)

        ### Form PubChem 3D Computed Property Subset
        pubchem_3d_com_subset = SubDataset(pubchem_3d_conformation, d2_com_dataset_instructions)

        ### load 3D computed properties instructions
        d2_des_data_paths = []
        d2_des_data_paths.append(root + '/pubchem/' + mode + '/2d_descriptive_properties.json')
        d2_des_dataset_instructions = load_dataset("json", data_files=d2_des_data_paths, features=features)['train']

        ### Form PubChem 3D Descriptive Subset
        pubchem_3d_des_subset = SubDataset(pubchem_3d_conformation, d2_des_dataset_instructions)

        ### Form PubChem 3D Caption Subset
        target_path = root + '/pubchem/' + mode + '/3d-pubchem.lmdb'
        pubchem_3d_cap_subset = PubchemCapDataset(target_path, unimol_dict, max_atoms)


        '''
        ADJUST THE NUMBER OF DATASET TO USE
        '''
        # self.datasets = [pubchemqc_3d_subset, pubchem_3d_com_subset, pubchem_3d_des_subset, pubchem_3d_cap_subset]
        # self.datasets = [pubchemqc_3d_subset, pubchem_3d_com_subset, pubchem_3d_des_subset] ### exclude captioning task
        self.datasets = [pubchemqc_3d_subset] #### additional training on 3d_subset
        # self.datasets = [pubchem_3d_com_subset] #### additional training on 3d_com_subset
        # self.datasets = [pubchem_3d_des_subset] #### additional training on des_subset


        
        self.lengths = [len(dataset) for dataset in self.datasets]
        self.sections = [0] + [sum(self.lengths[:i+1]) for i in range(len(self.lengths))]

        #Instruction prompt template
        self.prompt = "Below is an instruction that describes a task, paired with an input molecule. Write a response that appropriately completes the request.\n" \
                      "Instruction: {}\n" \
                      "Input molecule: {} <mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol>.\n" \
                      "Response: "

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, index):
        for i in range(len(self.lengths)):
            if index < self.sections[i+1]:
                dataset = self.datasets[i]
                idx = index - self.sections[i]
                break

        atom_vec, coordinates, edge_type, dist, smiles, data_2d, instruction, output, task = dataset[idx] 

        if task in ['Description']:
            input = self.prompt.format(instruction[:64], smiles[:96])
        else:
            input = self.prompt.format(instruction, smiles[:96])

        return (atom_vec, coordinates, edge_type, dist, smiles), input, output, task, data_2d
