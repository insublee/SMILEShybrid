import os
import pickle
from tqdm import tqdm
from multiprocessing import Pool
from collections import defaultdict
import re
import time
from typing import Any, Callable, Dict, List, Tuple, Optional, Union
import warnings

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from rdkit import Chem, DataStructs
from rdkit.Chem import  Draw
from openbabel import pybel
import deepchem as dc
from mordred import Calculator, descriptors
from transformers import  AutoTokenizer

import cv2
AutoTokenizer

class Datamodule(object):
    """Baseline datasets class"""
    sdf_loader = ['abonds', 'atoms', 'bonds', 'dbonds', 'HBA1', 'HBA2', 'HBD','logP', 'MP', 'MR', 'MW', 'nF', 'rotors', 'sbonds', 'tbonds', 'TPSA']
    mordred_drop_feats = ['PNS', 'PP', 'DP', 'FNS', 'FPS', 'WNS', 'WPS', 'TASA', 'TPSA', 'RASA', 'RPSA', 'MAX', 'MIN','Geom', 'GRAV', 'Mor']
    
    def __init__(
        self, 
        config: DictConfig,
    ):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1", use_fast=True)

        # 0. 캐시 확인
        cache_file_name=self.config.data.cache_file_name

        if os.path.exists(cache_file_name):
            print(f"load cache from {cache_file_name}")
            cached_file = open(cache_file_name, "rb")
            self.datasets = pickle.load(cached_file)
        else:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            # 1. 데이터 로드
            train = pd.read_csv(self.config.data.train_path)
            dev = pd.read_csv(self.config.data.dev_path)
            test = pd.read_csv(self.config.data.test_path)
            #train = train.iloc[:300]
            datasets = {'train':train, 'dev':dev, 'test':test}
            
            # 2. add features
            datasets = self.add_feature(datasets)
            
            # 3. postprocessing
            self.datasets = self.postprocessing(datasets)

            # 4. 저장
            cached_file = open(cache_file_name, "wb")
            pickle.dump(self.datasets, cached_file)


    @staticmethod
    def generate_img(
        df: pd.DataFrame,
        split: str,
        folder: str,
    ) -> None:
        """
        
        """
        file_ = f"/{split}_0.png"
        if not os.path.exists(folder):
            print(f"can not find {folder}. make new one")
            os.mkdir(folder)
        if not os.path.exists(folder+file_):
            print(f"can not find {folder+file_}. generate imgs")
            for idx, row in tqdm(df.iterrows(), desc="generate_img", total=len(df)):
                file = row['uid']
                smiles = row['SMILES']
                m = Chem.MolFromSmiles(smiles)
                if m != None:
                    img = Draw.MolToImage(m, size=(300,300))
                    img.save(f'../datasets/imgs/{file}.png')

    def add_feature(
        self, 
        datasets: Dict,
    ) -> Dict:
        """
        add feature from dictionaries of dataframe
        Args:
            datasets (Dict): dictionaries of dataframe
        Returns:
            datasets (Dict): preprocessed datasets
        """
        
        for split in datasets.keys():
            self.generate_img(datasets[split], split, folder = f'../datasets/imgs')
            
            dataset = datasets[split]
            start = time.time()
            # 1. SDF 추가
            dataset['sdf_features'] = dataset['uid'].apply(self.get_sdf)
            print(f"{split} : 1. SDF 추가 완료. ", round((time.time()- start)//60,2),'분')

            # 2. 화학지문 추가
            dataset['pfp_features'] = dataset['SMILES'].apply(self.get_pfp)
            print(f"{split} : 2. 화학지문 추가 완료. ", round((time.time()- start)//60,2),'분')

            # 3. mordred 지문 추가
            calc = Calculator(descriptors, ignore_3D=False)
            mol_series = dataset['SMILES'].apply(lambda x:Chem.MolFromSmiles(x))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mordred_df = calc.pandas(mol_series, quiet=False, nproc = os.cpu_count())
            mordred_df['Lipinski'] = mordred_df['Lipinski'].apply(lambda x: 1 if x == True else 0)
            mordred_df['GhoseFilter'] = mordred_df['GhoseFilter'].apply(lambda x: 1 if x == True else 0)
            mordred_df = self.str_to_float(mordred_df)
            dataset['mordred_features'] = [mordred_df.values[i] for i in range(mordred_df.shape[0])]
            print(f"{split} : 3. mordred 지문 추가. ", round((time.time()- start)//60,2),'분')

            # 4. xyz_feature 추가
            dataset['xyz_feature'] = dataset.uid.apply(self.get_xyz)
            print(f"{split} : 4. xyz_feature 추가 완료. ", round((time.time()- start)//60,2),'분')


            # 5. graph 추가
            dataset = self.get_graph(dataset, split)
            print(f"{split} : 5. graph 추가 완료. ", round((time.time()- start)//60,2),'분')

            # 6. input_ids, attention_mask 생성
            tokenized_series = dataset['SMILES'].apply(lambda x: self.tokenizer(x,
                                                                                max_length=self.config.data.max_seq_length,
                                                                                pad_to_max_length=True,
                                                                                truncation=True
                                                                                )
                                                       )
            dataset['input_ids'] = [i['input_ids'] for i in tokenized_series]
            dataset['attention_mask'] = [i['attention_mask'] for i in tokenized_series]
            print(f"{split} : 6. input_ids, attention_mask 추가 완료. ", round((time.time()- start)//60,2),'분')

            # 7.labels 생성
            if split!='test':
                dataset['labels'] = dataset['S1_energy(eV)'] - dataset['T1_energy(eV)']
                #dataset.drop(['S1_energy(eV)', 'T1_energy(eV)'], axis=1)
                print("before dropna shape", dataset.shape)
                dataset.dropna(inplace=True)
                print("after dropna shape", dataset.shape)

            datasets[split] = dataset

        return datasets

    
    @staticmethod
    def str_to_float(
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        mordred 파트 null feature 제거 및 str -> float 변환
        """
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype('float')
            for drop in self.mordred_drop_feats:
                if drop in col:
                    data.drop([col], axis=1, inplace=True)
        return data

    def postprocessing(
        self,
        datasets: Dict,
    ) -> Dict:
        """
        dev가 train과 분포가 달라 섞은 뒤 train_test_split 해준다.
        이후 파이토치 데이터셋으로 래핑해주고, 데이터 로더에 올릴 준비를 해준다.
        
        Args:
            datasets (Dict): dictionaries of dataframe
        Returns:
            datasets (Dict): preprocessed datasets
        """
        catted_df = pd.concat([datasets[key] for key in datasets.keys() if key!='test'])
        datasets['train'], datasets['dev'] = train_test_split(catted_df, test_size=self.config.data.train_test_split)
        datasets['train'].reset_index(inplace=True)
        datasets['dev'].reset_index(inplace=True)
        
        for split in datasets.keys():
            datasets[split] = CustomDataset(datasets[split], split)
        
        return datasets

    def train_dataloader(self):
        return DataLoader(self.datasets['train'], batch_size=self.config.train.train_batch_size, shuffle=True, num_workers=self.config.train.num_workers)

    def val_dataloader(self):
        return DataLoader(self.datasets['dev'], batch_size=self.config.train.dev_batch_size, num_workers=self.config.train.num_workers)

    def test_dataloader(self):
        return DataLoader(self.datasets['test'], batch_size=self.config.train.test_batch_size, num_workers=self.config.train.num_workers)

    @staticmethod
    def create_atoms(mol, atom_dict):
        """Transform the atom types in a molecule (e.g., H, C, and O)
        into the indices (e.g., H=0, C=1, and O=2).
        Note that each atom index considers the aromaticity.
        """
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        for a in mol.GetAromaticAtoms():
            i = a.GetIdx()
            atoms[i] = (atoms[i], 'aromatic')
        atoms = [atom_dict[a] for a in atoms]
        return np.array(atoms)

    @staticmethod
    def create_ijbonddict(mol, bond_dict):
        """Create a dictionary, in which each key is a node ID
        and each value is the tuples of its neighboring node
        and chemical bond (e.g., single and double) IDs.
        """
        i_jbond_dict = defaultdict(lambda: [])
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            bond = bond_dict[str(b.GetBondType())]
            i_jbond_dict[i].append((j, bond))
            i_jbond_dict[j].append((i, bond))
        return i_jbond_dict

    @staticmethod
    def extract_fingerprints(radius, atoms, i_jbond_dict,
                            fingerprint_dict, edge_dict):
        """Extract the fingerprints from a molecular graph
        based on Weisfeiler-Lehman algorithm.
        """

        if (len(atoms) == 1) or (radius == 0):
            nodes = [fingerprint_dict[a] for a in atoms]

        else:
            nodes = atoms
            i_jedge_dict = i_jbond_dict

            for _ in range(radius):

                """Update each node ID considering its neighboring nodes and edges.
                The updated node IDs are the fingerprint IDs.
                """
                nodes_ = []
                for i, j_edge in i_jedge_dict.items():
                    neighbors = [(nodes[j], edge) for j, edge in j_edge]
                    fingerprint = (nodes[i], tuple(sorted(neighbors)))
                    nodes_.append(fingerprint_dict[fingerprint])

                """Also update each edge ID considering
                its two nodes on both sides.
                """
                i_jedge_dict_ = defaultdict(lambda: [])
                for i, j_edge in i_jedge_dict.items():
                    for j, edge in j_edge:
                        both_side = tuple(sorted((nodes[i], nodes[j])))
                        edge = edge_dict[(both_side, edge)]
                        i_jedge_dict_[i].append((j, edge))

                nodes = nodes_
                i_jedge_dict = i_jedge_dict_

        return np.array(nodes)

    @staticmethod
    def str_to_float(
        data: pd.DataFrame
    ) -> pd.DataFrame:
        drop_feats = ['PNS', 'PP', 'DP', 'FNS', 'FPS', 'WNS', 'WPS', 'TASA', 'TPSA', 'RASA', 'RPSA', 'MAX', 'MIN',
                      'Geom', 'GRAV', 'Mor']
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype('float')
            for drop in drop_feats:
                if drop in col:
                    data.drop([col], axis=1, inplace=True)
        return data

    
    def get_sdf(
        self,
        x
    ):
        split, idx = x.split('_')
        sdf_path = os.path.join('..','datasets',f'{split}_sdf',f'{split}_{idx}.sdf')
        sdf = [i for i in pybel.readfile('sdf',sdf_path)]
        if len(sdf)>0:
            sdf = sdf[0].calcdesc(self.sdf_loader)
            return np.array([v for k,v in sdf.items()])
        else:
            return np.nan

    
    @staticmethod
    def get_pfp(x):
        pfp = Chem.MolFromSmiles(x)
        pfp = Chem.rdmolops.PatternFingerprint(pfp)
        pfp_arr = np.zeros((0,))
        DataStructs.ConvertToNumpyArray(pfp, pfp_arr)
        return pfp_arr

    
    @staticmethod
    def get_xyz(x):
        split, idx = x.split('_')
        sdf_path = os.path.join('..','datasets',f'{split}_sdf',f'{split}_{idx}.sdf')
        sdf = [i for i in pybel.readfile('sdf',sdf_path)]
        if len(sdf)==0:
            return np.nan
        xyz_element_list = []
        xyz_data = sdf[0].write("xyz").split('\n')[2:-1]

        for xyz in xyz_data:
            xyz = re.sub('[^\d.-]', ' ', xyz).split(' ')
            xyz = ' '.join(xyz).split()
            xyz = list(map(float, xyz))
            xyz_element_list.append(xyz)
        return xyz_element_list
    
    def get_graph(
        self,
        df: pd.DataFrame,
        split: str,
    )-> pd.DataFrame:
        """
        
        """
        featurizer = dc.feat.MolGraphConvFeaturizer()
        df['graph'] = featurizer.featurize(df['SMILES'])

        atom_dict = defaultdict(lambda: len(atom_dict))
        bond_dict = defaultdict(lambda: len(bond_dict))
        fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
        edge_dict = defaultdict(lambda: len(edge_dict))
        graph_fingerprints = []
        graph_adjacency = []
        graph_molecular_size = []
        for index, row in tqdm(df.iterrows(), desc=f"{split} : add Graph feature", total=len(df)):
            mol = Chem.AddHs(Chem.MolFromSmiles(row['SMILES']))
            atoms = self.create_atoms(mol, atom_dict)
            molecular_size = len(atoms)
            i_jbond_dict = self.create_ijbonddict(mol, bond_dict)
            fingerprints = self.extract_fingerprints(self.config.data.radius, atoms, i_jbond_dict,
                                                    fingerprint_dict, edge_dict)
            adjacency = Chem.GetAdjacencyMatrix(mol)
            graph_fingerprints.append(fingerprints)
            graph_adjacency.append(adjacency)
            graph_molecular_size.append(molecular_size)

        df['graph_fingerprints'] = graph_fingerprints
        df['graph_adjacency'] = graph_adjacency
        df['graph_molecular_size'] = graph_molecular_size
        
        return df

    


class CustomDataset(Dataset):
    loader_columns = [
        'uid', 
        'sdf_features', 
        'pfp_features', 
        'mordred_features',
        'xyz_feature', 
        'graph',
        'graph_fingerprints',
        'graph_adjacency',
        'graph_molecular_size',
        'input_ids',
        'attention_mask', 
        'labels'
    ]

    def __init__(
        self, 
        df: pd.DataFrame, 
        split: str,
    ) -> None:
        """

        """
        self.split = split
        self.df = df
        self.imgs = (f'../datasets/imgs/'+df.uid+'.png').to_numpy()
            
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, i):
        assert cv2.imread(self.imgs[i]) is not None, f'imread error. i : {i}, self.imgs[i] :{self.imgs[i]}'
        img = cv2.imread(self.imgs[i]).astype(np.float32)/255
        img = np.transpose(img, (2,0,1))
        return_dict = {
            'sdf_features' : torch.tensor(self.df.loc[i,'sdf_features'], dtype=torch.float32),
            'pfp_features' : torch.tensor(self.df.loc[i,'pfp_features'], dtype=torch.float32),
            'mordred_features' : torch.tensor(self.df.loc[i,'mordred_features'], dtype=torch.float32),
            #'xyz_feature' : torch.tensor(self.df.loc[i,'xyz_feature'], dtype=torch.float32),
            
            'img' : torch.tensor(img, dtype=torch.float32),
            
            #'graph' : torch.tensor(self.df.loc[i,'graph'], dtype=torch.float32),
            #'graph_fingerprints' : torch.tensor(self.df.loc[i,'graph_fingerprints'], dtype=torch.float32),
            #'graph_adjacency' : torch.tensor(self.df.loc[i,'graph_adjacency'], dtype=torch.float32),
            #'graph_molecular_size' : torch.tensor(self.df.loc[i,'graph_molecular_size'], dtype=torch.float32),
            
            'input_ids' : torch.tensor(self.df.loc[i,'input_ids']).long(),
            'attention_mask' : torch.tensor(self.df.loc[i,'attention_mask']).long(),
        }
        
        if self.split != 'test':
            return_dict['labels'] = torch.tensor(self.df.loc[i, 'labels'], dtype=torch.float32)
            return return_dict
        else:
            return return_dict
