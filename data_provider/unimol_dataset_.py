import os
import numpy as np
import torch
import random
import lmdb
import pickle
from functools import lru_cache
from unicore.data import data_utils
from scipy.spatial import distance_matrix
from torch.utils.data import Dataset

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MolFromSmiles
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect


import pytorch_lightning as pl
from torch import optim
from torch.utils import data
from torch_geometric.data import (Data, DataLoader, InMemoryDataset, download_url,
                                  extract_zip)




class LMDBDataset_cid:
    def __init__(self, db_path):
        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(self.db_path)
        env = self.connect_db(self.db_path)
        with env.begin() as txn:
            self._keys = list(txn.cursor().iternext(values=False))

    def connect_db(self, lmdb_path, save_to_self=False):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        if not save_to_self:
            return env
        else:
            self.env = env

    def __len__(self):
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, cid):
        if not hasattr(self, "env"):
            self.connect_db(self.db_path, save_to_self=True)
        datapoint_pickled = self.env.begin().get(cid.encode())
        data = pickle.loads(datapoint_pickled)
        return data

class D3Dataset_cid(Dataset):
    def __init__(self, path, dictionary, max_atoms=256):
        self.dictionary = dictionary
        self.num_types = len(dictionary)
        self.bos = dictionary.bos()
        self.eos = dictionary.eos()

        self.lmdb_dataset = LMDBDataset_cid(path)

        self.max_atoms = max_atoms
        ## the following is the default setting of uni-mol's pretrained weights
        self.remove_hydrogen = True
        self.remove_polar_hydrogen = False
        self.normalize_coords = True
        self.add_special_token = True
        self.__max_atoms = 512

    def __len__(self):
        return len(self.lmdb_dataset)

    def __getitem__(self, cid):
        data = self.lmdb_dataset[cid]
        smiles = data['smiles']
        description = data['description']
        enriched_description = data['enriched_description']
        ## deal with 3d coordinates
        atoms_orig = np.array(data['atoms'])
        atoms = atoms_orig.copy()
        coordinate_set = data['coordinates']
        coordinates = random.sample(coordinate_set, 1)[0].astype(np.float32)
        assert len(atoms) == len(coordinates) and len(atoms) > 0
        assert coordinates.shape[1] == 3

        ## deal with the hydrogen
        if self.remove_hydrogen:
            mask_hydrogen = atoms != "H"
            if sum(mask_hydrogen) > 0:
                atoms = atoms[mask_hydrogen]
                coordinates = coordinates[mask_hydrogen]

        if not self.remove_hydrogen and self.remove_polar_hydrogen:
            end_idx = 0
            for i, atom in enumerate(atoms[::-1]):
                if atom != "H":
                    break
                else:
                    end_idx = i + 1
            if end_idx != 0:
                atoms = atoms[:-end_idx]
                coordinates = coordinates[:-end_idx]

        ## deal with cropping
        if len(atoms) > self.max_atoms:
            index = np.random.permutation(len(atoms))[:self.max_atoms]
            atoms = atoms[index]
            coordinates = coordinates[index]

        assert 0 < len(atoms) <= self.__max_atoms

        atom_vec = torch.from_numpy(self.dictionary.vec_index(atoms)).long()

        if self.normalize_coords:
            coordinates = coordinates - coordinates.mean(axis=0)

        if self.add_special_token:
            atom_vec_3d = torch.cat([torch.LongTensor([self.bos]), atom_vec, torch.LongTensor([self.eos])])
            coordinates_3d = np.concatenate([np.zeros((1, 3)), coordinates, np.zeros((1, 3))], axis=0)

        ## obtain edge types; which is defined as the combination of two atom types
        edge_type_3d = atom_vec_3d.view(-1, 1) * self.num_types + atom_vec_3d.view(1, -1)
        dist_3d = distance_matrix(coordinates_3d, coordinates_3d).astype(np.float32)
        coordinates_3d, dist_3d = torch.from_numpy(coordinates_3d), torch.from_numpy(dist_3d)

        '''
        For non-padded dist matrix for 2d graph computation
        
        '''
        ## obtain edge types; which is defined as the combination of two atom types
        edge_type_2d = atom_vec.view(-1, 1) * self.num_types + atom_vec.view(1, -1)
        dist_2d = distance_matrix(coordinates, coordinates).astype(np.float32)
        coordinates_2d, dist_2d = torch.from_numpy(coordinates), torch.from_numpy(dist_2d) 


        '''
        deal with 2d coordinates
        '''
        
        data_2d = load_data_from_pt([smiles],add_dummy_node=True) ############ kjh 
        
        '''
        dist dummy node process
        '''
        m = np.full((dist_2d.shape[0] + 1, dist_2d.shape[1] + 1), 1e6)
        m[1:, 1:] = dist_2d
        dist_2d = m

        
        data_2d = Data(x = torch.from_numpy(data_2d[0][0]), edge_index = torch.from_numpy(data_2d[0][1]), node_dist = torch.from_numpy(dist_2d))
        # print(data_2d)
        
        # return atom_vec, coordinates, edge_type, dist, smiles, description, enriched_description
        return atom_vec_3d, coordinates_3d, edge_type_3d, dist_3d, smiles, data_2d, description, enriched_description


class LMDBDataset_index:
    def __init__(self, db_path):
        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(self.db_path)
        env = self.connect_db(self.db_path)
        with env.begin() as txn:
            self._keys = list(txn.cursor().iternext(values=False))

    def connect_db(self, lmdb_path, save_to_self=False):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        if not save_to_self:
            return env
        else:
            self.env = env

    def __len__(self):
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if not hasattr(self, "env"):
            self.connect_db(self.db_path, save_to_self=True)
        datapoint_pickled = self.env.begin().get(idx.encode())
        data = pickle.loads(datapoint_pickled)
        return data

class D3Dataset_index(Dataset):
    def __init__(self, path, dictionary, max_atoms=256):
        self.dictionary = dictionary
        self.num_types = len(dictionary)
        self.bos = dictionary.bos()
        self.eos = dictionary.eos()

        self.lmdb_dataset = LMDBDataset_index(path)

        self.max_atoms = max_atoms
        ## the following is the default setting of uni-mol's pretrained weights
        self.remove_hydrogen = True
        self.remove_polar_hydrogen = False
        self.normalize_coords = True
        self.add_special_token = True
        self.__max_atoms = 512

    def __len__(self):
        return len(self.lmdb_dataset)

    def __getitem__(self, index):
        data = self.lmdb_dataset[index]
        smiles = data['smi']


        ## deal with 3d coordinates
        atoms_orig = np.array(data['atoms'])
        atoms = atoms_orig.copy()
        coordinate_set = data['coordinates_list']
        coordinates = random.sample(coordinate_set, 1)[0].astype(np.float32)
        assert len(atoms) == len(coordinates) and len(atoms) > 0
        assert coordinates.shape[1] == 3

        ## deal with the hydrogen
        if self.remove_hydrogen:
            mask_hydrogen = atoms != "H"
            if sum(mask_hydrogen) > 0:
                atoms = atoms[mask_hydrogen]
                coordinates = coordinates[mask_hydrogen]

        if not self.remove_hydrogen and self.remove_polar_hydrogen:
            end_idx = 0
            for i, atom in enumerate(atoms[::-1]):
                if atom != "H":
                    break
                else:
                    end_idx = i + 1
            if end_idx != 0:
                atoms = atoms[:-end_idx]
                coordinates = coordinates[:-end_idx]

        ## deal with cropping
        if self.max_atoms > 0 and len(atoms) > self.max_atoms:
            index = np.random.permutation(len(atoms))[:self.max_atoms]
            atoms = atoms[index]
            coordinates = coordinates[index]

        assert 0 < len(atoms) < self.__max_atoms, print(len(atoms), atoms_orig, index)
        atom_vec = torch.from_numpy(self.dictionary.vec_index(atoms)).long()

        if self.normalize_coords:
            coordinates = coordinates - coordinates.mean(axis=0)

        if self.add_special_token:
            atom_vec_3d = torch.cat([torch.LongTensor([self.bos]), atom_vec, torch.LongTensor([self.eos])])
            coordinates_3d = np.concatenate([np.zeros((1, 3)), coordinates, np.zeros((1, 3))], axis=0)

        ## obtain edge types; which is defined as the combination of two atom types
        edge_type_3d = atom_vec_3d.view(-1, 1) * self.num_types + atom_vec_3d.view(1, -1)
        dist_3d = distance_matrix(coordinates_3d, coordinates_3d).astype(np.float32)
        coordinates_3d, dist_3d = torch.from_numpy(coordinates_3d), torch.from_numpy(dist_3d)

        '''
        For non-padded dist matrix for 2d graph computation
        
        '''
        ## obtain edge types; which is defined as the combination of two atom types
        edge_type_2d = atom_vec.view(-1, 1) * self.num_types + atom_vec.view(1, -1)
        dist_2d = distance_matrix(coordinates, coordinates).astype(np.float32)
        coordinates_2d, dist_2d = torch.from_numpy(coordinates), torch.from_numpy(dist_2d) 



        '''
        deal with 2d coordinates
        '''

        
        data_2d = load_data_from_pt([smiles],add_dummy_node=True) ############ kjh 
        
        '''
        dist dummy node process
        '''
        m = np.full((dist_2d.shape[0] + 1, dist_2d.shape[1] + 1), 1e6)
        m[1:, 1:] = dist_2d
        dist_2d = m

        
        data_2d = Data(x = torch.from_numpy(data_2d[0][0]), edge_index = torch.from_numpy(data_2d[0][1]), node_dist = torch.from_numpy(dist_2d))
        # print(data_2d)
        
        return atom_vec_3d, coordinates_3d, edge_type_3d, dist_3d, smiles, data_2d

def collate_tokens_coords(
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


class D3Collater:
    def __init__(self, pad_idx, pad_to_multiple=8):
        self.pad_idx = pad_idx
        self.pad_to_multiple = pad_to_multiple
    
    def __call__(self, samples):
        atom_vec, coordinates, edge_type, dist, smiles = zip(*samples)
        padded_atom_vec = data_utils.collate_tokens(atom_vec, self.pad_idx, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms]
        padded_coordinates = collate_tokens_coords(coordinates, 0, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms, 3]
        padded_edge_type = data_utils.collate_tokens_2d(edge_type, 0, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms, max_atoms]
        padded_dist = data_utils.collate_tokens_2d(dist, 0, left_pad=False, pad_to_multiple=self.pad_to_multiple) # shape = [batch_size, max_atoms, max_atoms]
        return padded_atom_vec, padded_coordinates, padded_edge_type, padded_dist, smiles

def load_data_from_pt(data_smiles, add_dummy_node=True, one_hot_formal_charge=True, use_data_saving=False):
    feat_stamp = f'{"_dn" if add_dummy_node else ""}{"_ohfc" if one_hot_formal_charge else ""}'

    x_all = load_data_from_smiles(data_smiles, add_dummy_node=add_dummy_node,
                                         one_hot_formal_charge=one_hot_formal_charge)
    return x_all


# def load_data_from_smiles(x_smiles, labels, add_dummy_node=True, one_hot_formal_charge=False):
def load_data_from_smiles(x_smiles, add_dummy_node=True, one_hot_formal_charge=True):
    """Load and featurize data from lists of SMILES strings and labels.

    Args:
        x_smiles (list[str]): A list of SMILES strings.
        labels (list[float]): A list of the corresponding labels.
        add_dummy_node (bool): If True, a dummy node will be added to the molecular graph. Defaults to True.
        one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded. Defaults to False.

    Returns:
        A tuple (X, y) in which X is a list of graph descriptors (node features, adjacency matrices, distance matrices),
        and y is a list of the corresponding labels.
    """
    x_all = []

    for smiles in x_smiles:
        try:
            mol = MolFromSmiles(smiles)
            afm, adj = featurize_mol(mol, add_dummy_node, one_hot_formal_charge)
            x_all.append([afm, adj])
        except ValueError as e:
            logging.warning('the SMILES ({}) can not be converted to a graph.\nREASON: {}'.format(smiles, e))

    return x_all

def featurize_mol(mol, add_dummy_node, one_hot_formal_charge):
    """Featurize molecule.

    Args:
        mol (rdchem.Mol): An RDKit Mol object.
        add_dummy_node (bool): If True, a dummy node will be added to the molecular graph.
        one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded.

    Returns:
        A tuple of molecular graph descriptors (node features, adjacency matrix, distance matrix).
    """
    node_features = np.array([get_atom_features(atom, one_hot_formal_charge)
                              for atom in mol.GetAtoms()])

    adj_matrix = np.eye(mol.GetNumAtoms())
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtom().GetIdx()
        end_atom = bond.GetEndAtom().GetIdx()
        adj_matrix[begin_atom, end_atom] = adj_matrix[end_atom, begin_atom] = 1


    if add_dummy_node:
        m = np.zeros((node_features.shape[0] + 1, node_features.shape[1] + 1))
        m[1:, 1:] = node_features
        m[0, 0] = 1.
        node_features = m

        m = np.zeros((adj_matrix.shape[0] + 1, adj_matrix.shape[1] + 1))
        m[1:, 1:] = adj_matrix
        adj_matrix = m

    return node_features, adj_matrix

def get_atom_features(atom, one_hot_formal_charge=True):
    """Calculate atom features.

    Args:
        atom (rdchem.Atom): An RDKit Atom object.
        one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded.

    Returns:
        A 1-dimensional array (ndarray) of atom features.
    """
    attributes = []

    attributes += one_hot_vector(
        atom.GetAtomicNum(),
        [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999] # 11
    )

    attributes += one_hot_vector(
        len(atom.GetNeighbors()),
        [0, 1, 2, 3, 4, 5] # 6
    )

    attributes += one_hot_vector(
        atom.GetTotalNumHs(),
        [0, 1, 2, 3, 4] # 5
    )

    if one_hot_formal_charge:
        attributes += one_hot_vector(
            atom.GetFormalCharge(),
            [-1, 0, 1] # 3
        )
    else:
        attributes.append(atom.GetFormalCharge())

    attributes.append(atom.IsInRing()) # 1
    attributes.append(atom.GetIsAromatic()) # 1

    return np.array(attributes, dtype=np.float32)

def one_hot_vector(val, lst):
    """Converts a value to a one-hot vector based on options in lst"""
    if val not in lst:
        val = lst[-1]
    return map(lambda x: x == val, lst)

