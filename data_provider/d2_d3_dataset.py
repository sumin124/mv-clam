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


class PubChemDataset(InMemoryDataset):
    def __init__(self, path):
        super(PubChemDataset, self).__init__()
        self.data, self.slices = torch.load(path)
    
    def __getitem__(self, idx):
        return self.get(idx)

class pt_dataset_cid:
    
    def __init__(self, db_path):

        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(self.db_path)
        env = self.connect_db(self.db_path)
        # self.cids = env['cid'].to_list()
        self.cids = [env[i][0] for i in range(len(env))]

    def connect_db(self, pt_path, save_to_self=False):

        env = torch.load(open(pt_path, 'rb'))

        if not save_to_self:
            return env
        else:
            self.env = env

    def __len__(self):
        # return self.env.shape[0]
        return len(self.cids)

    @lru_cache(maxsize=16)
    def __getitem__(self,idx):

        if not hasattr(self, 'env'):
            self.connect_db(self.db_path, save_to_self=True)
        data = self.env[idx]
        return data  



class pt_dataset_smiles: ## for lm24 dataset (NO cid numbers)
    
    def __init__(self, db_path):

        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(self.db_path)
        env = self.connect_db(self.db_path)
        self.smiles = [env[i][2] for i in range(len(env))]

    def connect_db(self, pt_path, save_to_self=False):

        env = torch.load(open(pt_path, 'rb'))

        if not save_to_self:
            return env
        else:
            self.env = env

    def __len__(self):
        return len(self.smiles)

    @lru_cache(maxsize=16)
    def __getitem__(self,idx):

        if not hasattr(self, 'env'):
            self.connect_db(self.db_path, save_to_self=True)
        data = self.env[idx]
        return data  
        

class pt_dataset(Dataset):

    def __init__(self, path, dictionary, max_atoms=256, enriched_description=False, lm24=False):


        self.dictionary = dictionary
        self.num_types = len(dictionary)
        self.bos = dictionary.bos()
        self.eos = dictionary.eos()

        ### for lm24 ###
        
        if lm24: 
            self.pk_3d_dataset = pt_dataset_smiles(path)
        else:
            self.pk_3d_dataset = pt_dataset_cid(path)
            

        self.max_atoms = max_atoms
        self.remove_hydrogen = True
        self.remove_polar_hydrogen = False
        self.normalize_coords = True
        self.add_special_token = True
        self.__max_atoms = 512

        self.enriched_description = enriched_description

    def __len__(self):
        
        return len(self.pk_3d_dataset)

    def __getitem__(self,idx):
        
        '''
        data[0] : cid
        data[1] : text
        data[2] : smiles
        data[3] : atom_vec
        data[4] : dist
        data[5] : edge type
        data[6] : coordinate
        data[7] : d2_graphs
        data[8] : enriched description
        data[9] : Molca description (without explicit molecule name)
        '''

        data = self.pk_3d_dataset[idx]
        smiles = data[2]

        if self.enriched_description == True: 
            description = data[8]
        else:
            description = data[1]
        atom_vec = np.array(data[3])
        
        coordinates = data[6]
        
        assert len(atom_vec) == len(coordinates) and len(atom_vec) > 0
        assert coordinates.shape[1] == 3

        ## deal with cropping
        if len(atom_vec) > self.max_atoms:
            index = np.random.permutation(len(atom_vec))[:self.max_atoms]
            atom_vec = atom_vec[index]
            coordinates = coordinates[index]
            
        assert 0 < len(atom_vec) <= self.__max_atoms

        atom_vec = torch.from_numpy(atom_vec).long()

        if self.normalize_coords:
            coordinates = coordinates - coordinates.mean(axis=0)

        if self.add_special_token:
            atom_vec = torch.cat([torch.LongTensor([self.bos]), atom_vec, torch.LongTensor([self.eos])])
            coordinates = np.concatenate([np.zeros((1, 3)), coordinates.float() , np.zeros((1, 3))], axis=0)

        ## obtain edge types; which is defined as the combination of two atom types
        edge_type = atom_vec.view(-1, 1) * self.num_types + atom_vec.view(1, -1)
        dist = distance_matrix(coordinates, coordinates).astype(np.float32)
        coordinates, dist = torch.from_numpy(coordinates), torch.from_numpy(dist)
        return atom_vec, coordinates, edge_type, dist, smiles, description, data[7]
        
class pt_MolDataset(Dataset):

    def __init__(self, root, tokenizer, text_max_len, unimol_dict=None, max_atoms=256, prompt='', return_prompt=False, enriched_description=False, lm24=False):

        super(pt_MolDataset, self).__init__()
        self.prompt = prompt
        self.return_prompt = return_prompt
        self.tokenizer = tokenizer
        self.lm24 = lm24
        
        self.enriched_description = enriched_description 

        self.root = root
        self.text_max_len = text_max_len
        
        target_path = root
        if 'PubChem' in root: 
            target_path = os.path.join(root, 'pubchem_2d_3d.pt') 
        if 'lm24' in root:
            target_path = os.path.join(root, 'lm24_2d_3d.pt') 

        print(f'Read from {target_path}') 
        

        self.pt_dataset = pt_dataset(target_path, unimol_dict, max_atoms, enriched_description=self.enriched_description, lm24=self.lm24) ######
        
        self.permutation = None

    def shuffle(self):

        ## shuffle the dataset using a permutation matrix
        self.permutation = torch.randperm(len(self)).numpy()
        return self

    def __len__(self):
        return len(self.pt_dataset)

    def get_2d_3d(self, index):
        atom_vec, coordinates, edge_type, dist, smiles, description, d2_graph = self.pt_dataset[index]
        

        if self.return_prompt:

            smiles_prompt = self.prompt.format(smiles[:96]) 
            return (atom_vec, coordinates, edge_type, dist, smiles), d2_graph, smiles_prompt, description, index
        return (atom_vec, coordinates, edge_type, dist, smiles), d2_graph, description


    def __getitem__(self, index):

        ## consider the permutation
        if self.permutation is not None:
            index = self.permutation[index]

        return self.get_2d_3d(index)

    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=True,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        input_ids = sentence_token['input_ids']
        attention_mask = sentence_token['attention_mask']
        return input_ids, attention_mask


class pubchem_to_2d_3d(Dataset):

    def __init__(self, pt_dataset, d3_dataset):
        
        self.cids_init = pt_dataset.cid
        self.texts_init = pt_dataset.text
        self.smiles_init = pt_dataset.smiles

        

        self.cids, self.texts, self.enriched_texts, self.smiles, self.texts_molca = [], [], [], [], []
        
        self.atom_vecs, self.dists, self.edge_types, self.coordinates = [], [], [], []
        self.d2_graphs = []

        append_index = 0
        
        for cid in tqdm(self.cids_init):

            try:
            
                atom_vec, coordinate, edge_type, dist, smile, description, enriched_description = d3_dataset[cid]
                self.atom_vecs.append(atom_vec)
                self.dists.append(dist)
                self.edge_types.append(edge_type)
                self.coordinates.append(coordinate)
                self.texts.append(description) 
                self.enriched_texts.append(enriched_description)
    
                self.d2_graphs.append(Data(pt_dataset[append_index].x, pt_dataset[append_index].edge_index, pt_dataset[append_index].edge_attr))
                
                self.cids.append(self.cids_init[append_index]) 
                self.texts_molca.append(self.texts_init[append_index]) #### ### without molecular name (The molecule...)
                self.smiles.append(smile) ####
    
                append_index += 1


            except:
                
                append_index += 1
                continue



    def get_graph(self, smiles):
        mol = AllChem.MolFromSmiles(smiles)
        try: 
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, maxAttempts=5000)
            AllChem.MMFFOptimizeMolecule(mol, maxIters=400)
            mol = Chem.RemoveHs(mol)
            coordinates = mol.GetConformer().GetPositions()

        except:
            res = AllChem.Compute2dCoords(mol)
            
            
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]

        assert len(atoms) == len(coordinates), "coordinates shape is not align with {}".format(smiles)
        assert coordinates.shape[1] == 3
        
        #atoms = np.asarray(atoms)
        
        ## atom vectors
        dictionary = Dictionary.load('/data/project/sumin/moleculeText/3D-MoLM/data_provider/unimol_dict.txt')
        dictionary.add_symbol("[MASK]", is_special=True)
        atom_vec = torch.from_numpy(dictionary.vec_index(atoms)).long()

        ## normalize coordinates:
        coordinates = coordinates - coordinates.mean(axis=0)

        edge_type = atom_vec.view(-1, 1) * len(dictionary) + atom_vec.view(1, -1)
        dist = distance_matrix(coordinates, coordinates).astype(np.float32)
        coordinates, dist = torch.from_numpy(coordinates), torch.from_numpy(dist)

        return atom_vec, dist, edge_type, coordinates

    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        return self.cids[idx], self.texts[idx], self.smiles[idx], self.atom_vecs[idx], self.dists[idx], self.edge_types[idx], self.coordinates[idx], self.d2_graphs[idx], self.enriched_texts[idx], self.texts_molca[idx]



class lm24_to_2d_3d_split_merge(Dataset):

    '''
    merge given splited dataset & remove datas with 2d coords info as 3d coords.
    '''


    def __init__(self, *split_datasets):

        self.cids, self.texts, self.smiles = [], [], []
        self.atom_vecs, self.dists, self.edge_types, self.coordinates, self.d2_graphs = [], [], [], [], []
        self.enriched_texts, self.texts_molca = [], []
        
        for split_dataset in split_datasets:

            for data in tqdm(split_dataset):


                if data[6].x.shape[0] == data[6].node_dist.shape[0]:
                    
                    self.cids.append('_') # NO CID info.
                    self.texts.append(data[0])
                    self.smiles.append(data[1])
                    self.atom_vecs.append(data[2])
                    self.dists.append(data[3])
                    self.edge_types.append(data[4])
                    self.coordinates.append(data[5])
                    self.d2_graphs.append(data[6])
                    self.enriched_texts.append('_') # NO Enriched texts info.
                    self.texts_molca.append('_') # NO text for MolCA info.

                else:
                    
                    # remove datas with 2d coords info as 3d coords.
                    print(f'Skip {data[1]} : {data[6].x.shape[0]} != {data[6].node_dist.shape[0]}')
                    continue

        

    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):

        
        return self.cids[idx], self.texts[idx], self.smiles[idx], self.atom_vecs[idx], self.dists[idx], self.edge_types[idx], self.coordinates[idx], self.d2_graphs[idx], self.enriched_texts[idx], self.texts_molca[idx]




class pickle_to_2d_3d(Dataset):

    def __init__(self, pickle_df):
        
        self.cids_init = pickle_df['cid'].to_list()
        self.texts_init = pickle_df['text'].to_list()
        self.smiles_init = pickle_df['SMILES'].to_list()

        self.cids, self.texts, self.smiles = [], [], []
        
        self.atom_vecs, self.dists, self.edge_types, self.coordinates = [], [], [], []
        self.d2_graphs = []

        append_index = 0
        for smile in tqdm(self.smiles_init):
            try:
                atom_vec, dist, edge_type, coordinate = self.get_graph(smile)
                self.atom_vecs.append(atom_vec)
                self.dists.append(dist)
                self.edge_types.append(edge_type)
                self.coordinates.append(coordinate)
                mol = Chem.MolFromSmiles(smile)
                self.d2_graphs.append(mol_to_graph_data_obj_simple(mol))
                
                self.cids.append(self.cids_init[append_index]) ####
                self.texts.append(self.texts_init[append_index]) ####
                self.smiles.append(smile) ####

                append_index += 1


            except:
                append_index += 1
                continue



    def get_graph(self, smiles):
        mol = AllChem.MolFromSmiles(smiles)
        try: 
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, maxAttempts=5000)
            AllChem.MMFFOptimizeMolecule(mol, maxIters=400)
            mol = Chem.RemoveHs(mol)
            coordinates = mol.GetConformer().GetPositions()
            

        except:
            res = AllChem.Compute2dCoords(mol)
            
            
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]

        assert len(atoms) == len(coordinates), "coordinates shape is not align with {}".format(smiles)
        assert coordinates.shape[1] == 3
        
        ## atom vectors
        dictionary = Dictionary.load('/data/project/sumin/moleculeText/3D-MoLM/data_provider/unimol_dict.txt')
        dictionary.add_symbol("[MASK]", is_special=True)
        atom_vec = torch.from_numpy(dictionary.vec_index(atoms)).long()

        ## normalize coordinates:
        coordinates = coordinates - coordinates.mean(axis=0)

        ## obtain edge types; which is defined as the combination of two atom types
        edge_type = atom_vec.view(-1, 1) * len(dictionary) + atom_vec.view(1, -1)
        dist = distance_matrix(coordinates, coordinates).astype(np.float32)
        coordinates, dist = torch.from_numpy(coordinates), torch.from_numpy(dist)

        return atom_vec, dist, edge_type, coordinates

    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        return self.cids[idx], self.texts[idx], self.smiles[idx], self.atom_vecs[idx], self.dists[idx], self.edge_types[idx], self.coordinates[idx], self.d2_graphs[idx]




class MyCollater:
    def __init__(self, tokenizer, text_max_len, pad_idx, pad_to_multiple):
        self.pad_idx = pad_idx
        self.tokenizer = tokenizer
        self.text_max_len = text_max_len
        self.pad_to_multiple = pad_to_multiple

    def collate_tokens_coords(
            self,
            values,
            pad_idx,
            left_pad=False,
            pad_to_length=None,
            pad_to_multiple=1,
        ):
            """Convert a list of 1d tensors into a padded 2d tensor."""
            size = max(v.size(0) for v in values)
            size = size if pad_to_length is None else max(size, pad_to_length)
            if pad_to_multiple != 1 and size % pad_to_multiple != 0:
                size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
            res = values[0].new(len(values), size, 3).fill_(pad_idx)
        
            def copy_tensor(src, dst):
                assert dst.numel() == src.numel()
                dst.copy_(src)
        
            for i, v in enumerate(values):
                copy_tensor(v, res[i][size - len(v) :, :] if left_pad else res[i][: len(v), :])
            return res    

    def __call__(self, batch):
        d3_batch, d2_batch, text_batch= zip(*batch)
        atom_vec, coordinates, edge_type, dist, smiles = zip(*d3_batch)
        padded_atom_vec = data_utils.collate_tokens(atom_vec, self.pad_idx, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms]
        padded_coordinates = self.collate_tokens_coords(coordinates, 0, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms, 3]
        padded_edge_type = data_utils.collate_tokens_2d(edge_type, 0, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms, max_atoms]
        padded_dist = data_utils.collate_tokens_2d(dist, 0, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms, max_atoms]
        # print(text_batch)
        text_tokens = self.tokenizer(text_batch,
                                     truncation=True, 
                                     padding='max_length',
                                     add_special_tokens=True,
                                     max_length=self.text_max_len,
                                     return_tensors='pt',
                                     return_attention_mask=True, 
                                     return_token_type_ids=False)
        
        d2_batch = self.d2_graph_encoder_batch(*d2_batch) ###########

        return (padded_atom_vec, padded_dist, padded_edge_type), text_tokens, d2_batch


    def d2_graph_encoder_batch(self, *data_objects):
        # Extract node features, edge indices, and node distances
        node_features = [data.x for data in data_objects]
        adj_matrices = [data.edge_index for data in data_objects]
        node_dists = [data.node_dist for data in data_objects]
        
        # Determine the maximum number of nodes
        max_num_nodes = max([nf.size(0) for nf in node_features])
        num_features = node_features[0].size(1)

        # Pad node features
        padded_node_features = []
        for nf in node_features:
            padding = torch.zeros((max_num_nodes - nf.size(0), num_features))
            padded_node_features.append(torch.cat([nf, padding], dim=0))
        node_feature_tensor = torch.stack(padded_node_features)

        # Pad adjacency matrices
        padded_adj_matrices = []
        for adj in adj_matrices:
            pad_size = max_num_nodes - adj.size(0)
            padded_adj = torch.nn.functional.pad(adj, (0, pad_size, 0, pad_size))
            padded_adj_matrices.append(padded_adj)
        adj_matrix_tensor = torch.stack(padded_adj_matrices)

        # Pad node distance matrices
        padded_node_dists = []
        for nd in node_dists:
            pad_size = max_num_nodes - nd.size(0)
            padded_nd = torch.nn.functional.pad(nd, (0, pad_size, 0, pad_size))
            padded_node_dists.append(padded_nd)
        node_dist_tensor = torch.stack(padded_node_dists)

        return node_feature_tensor.to(torch.float32), adj_matrix_tensor.to(torch.float32), node_dist_tensor.to(torch.float32)



class TrainCollater:
    
    def __init__(self, tokenizer, text_max_len, mol_token_id, pad_idx, pad_to_multiple, num_tokens):
        
        self.pad_idx = pad_idx
        self.tokenizer = tokenizer
        self.text_max_len = text_max_len
        self.mol_token_id = mol_token_id
        self.pad_to_multiple = pad_to_multiple
        
        self.num_tokens = num_tokens ############

    def collate_tokens_coords(
            self,
            values,
            pad_idx,
            left_pad=False,
            pad_to_length=None,
            pad_to_multiple=1,
        ):
            
            """Convert a list of 1d tensors into a padded 2d tensor."""
            size = max(v.size(0) for v in values)
            size = size if pad_to_length is None else max(size, pad_to_length)
            if pad_to_multiple != 1 and size % pad_to_multiple != 0:
                size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
            res = values[0].new(len(values), size, 3).fill_(pad_idx)
        
            def copy_tensor(src, dst):
                assert dst.numel() == src.numel()
                dst.copy_(src)
        
            for i, v in enumerate(values):
                copy_tensor(v, res[i][size - len(v) :, :] if left_pad else res[i][: len(v), :])
            return res    

    def __call__(self, batch):
        
        d3_batch, d2_batch, smiles_prompt, text_batch, index= zip(*batch)

        atom_vec, coordinates, edge_type, dist, smiles = zip(*d3_batch)
        padded_atom_vec = data_utils.collate_tokens(atom_vec, self.pad_idx, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms]
        padded_coordinates = self.collate_tokens_coords(coordinates, 0, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms, 3]
        padded_edge_type = data_utils.collate_tokens_2d(edge_type, 0, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms, max_atoms]
        padded_dist = data_utils.collate_tokens_2d(dist, 0, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms, max_atoms]

        input_pair = [[p,t] for p, t in zip(smiles_prompt, text_batch)]
        self.tokenizer.padding_side = 'left'
        text_smiles_tokens = self.tokenizer(input_pair,
                                            truncation='only_second', 
                                            padding='max_length',
                                            add_special_tokens=True,
                                            max_length=self.text_max_len,
                                            return_tensors='pt',
                                            return_attention_mask=True, 
                                            return_token_type_ids=True)
        is_mol_token = (text_smiles_tokens.input_ids == self.mol_token_id)

        try:
            assert torch.sum(is_mol_token).item() == 12 * len(batch)
        except:
            assert torch.sum(is_mol_token).item() == 24 * len(batch)

        text_smiles_tokens['is_mol_token'] = is_mol_token
        d2_batch = self.d2_graph_encoder_batch(*d2_batch) 
        

        return (padded_atom_vec, padded_dist, padded_edge_type), text_smiles_tokens, d2_batch

    def d2_graph_encoder_batch(self, *data_objects):
        # Extract node features, edge indices, and node distances
        node_features = [data.x for data in data_objects]
        adj_matrices = [data.edge_index for data in data_objects]
        node_dists = [data.node_dist for data in data_objects]
        
        # Determine the maximum number of nodes
        max_num_nodes = max([nf.size(0) for nf in node_features])
        num_features = node_features[0].size(1)

        # Pad node features
        padded_node_features = []
        for nf in node_features:
            padding = torch.zeros((max_num_nodes - nf.size(0), num_features))
            padded_node_features.append(torch.cat([nf, padding], dim=0))
        node_feature_tensor = torch.stack(padded_node_features)

        # Pad adjacency matrices
        padded_adj_matrices = []
        for adj in adj_matrices:
            pad_size = max_num_nodes - adj.size(0)
            padded_adj = torch.nn.functional.pad(adj, (0, pad_size, 0, pad_size))
            padded_adj_matrices.append(padded_adj)
        adj_matrix_tensor = torch.stack(padded_adj_matrices)

        # Pad node distance matrices
        padded_node_dists = []
        for nd in node_dists:
            pad_size = max_num_nodes - nd.size(0)
            padded_nd = torch.nn.functional.pad(nd, (0, pad_size, 0, pad_size))
            padded_node_dists.append(padded_nd)
        node_dist_tensor = torch.stack(padded_node_dists)

        return node_feature_tensor.to(torch.float32), adj_matrix_tensor.to(torch.float32), node_dist_tensor.to(torch.float32)

class InferenceCollater:
    
    def __init__(self, tokenizer, text_max_len, mol_token_id, pad_idx, pad_to_multiple, num_tokens):
        
        self.pad_idx = pad_idx
        self.tokenizer = tokenizer
        self.text_max_len = text_max_len
        self.mol_token_id = mol_token_id
        self.pad_to_multiple = pad_to_multiple
        
        self.num_tokens = num_tokens  


    def __call__(self, batch):
        d3_batch, d2_batch, smiles_prompt, text_batch, indices= zip(*batch)
        atom_vec, coordinates, edge_type, dist, smiles = zip(*d3_batch)
        padded_atom_vec = data_utils.collate_tokens(atom_vec, self.pad_idx, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms]
        padded_edge_type = data_utils.collate_tokens_2d(edge_type, 0, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms, max_atoms]
        padded_dist = data_utils.collate_tokens_2d(dist, 0, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms, max_atoms]
        
        self.tokenizer.padding_side = 'right'
        text_smiles_tokens = self.tokenizer(smiles_prompt,
                                        truncation=False,
                                        padding='longest',
                                        add_special_tokens=True,
                                        return_tensors='pt',
                                        return_attention_mask=True, 
                                        return_token_type_ids=True)

        is_mol_token = text_smiles_tokens.input_ids == self.mol_token_id
        text_smiles_tokens['is_mol_token'] = is_mol_token
        target_dict = {'targets': text_batch, 'indices': indices}

        d2_batch = self.d2_graph_encoder_batch(*d2_batch) ###########


        return (padded_atom_vec, padded_dist, padded_edge_type), text_smiles_tokens, d2_batch, target_dict

    def d2_graph_encoder_batch(self, *data_objects):
        # Extract node features, edge indices, and node distances
        node_features = [data.x for data in data_objects]
        adj_matrices = [data.edge_index for data in data_objects]
        node_dists = [data.node_dist for data in data_objects]
        
        # Determine the maximum number of nodes
        max_num_nodes = max([nf.size(0) for nf in node_features])
        num_features = node_features[0].size(1)

        # Pad node features
        padded_node_features = []
        for nf in node_features:
            padding = torch.zeros((max_num_nodes - nf.size(0), num_features))
            padded_node_features.append(torch.cat([nf, padding], dim=0))
        node_feature_tensor = torch.stack(padded_node_features)

        # Pad adjacency matrices
        padded_adj_matrices = []
        for adj in adj_matrices:
            pad_size = max_num_nodes - adj.size(0)
            padded_adj = torch.nn.functional.pad(adj, (0, pad_size, 0, pad_size))
            padded_adj_matrices.append(padded_adj)
        adj_matrix_tensor = torch.stack(padded_adj_matrices)

        # Pad node distance matrices
        padded_node_dists = []
        for nd in node_dists:
            pad_size = max_num_nodes - nd.size(0)
            padded_nd = torch.nn.functional.pad(nd, (0, pad_size, 0, pad_size))
            padded_node_dists.append(padded_nd)
        node_dist_tensor = torch.stack(padded_node_dists)

        return node_feature_tensor.to(torch.float32), adj_matrix_tensor.to(torch.float32), node_dist_tensor.to(torch.float32)
