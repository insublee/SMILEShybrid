import os
import pickle
from tqdm import tqdm
from multiprocessing import Pool
from collections import defaultdict
import re

import pandas as pd
import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import  Draw
from openbabel import pybel
import deepchem as dc

from mordred import Calculator, descriptors

import torch
from torch.utils.data import Dataset, DataLoader
import cv2

class Datamodule(object):
    """
    TODO : config화, 주석달기, 라이브러리화
    """
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config

        
        # 0. 캐시 확인
        cache_file_name=self.config.data.cache_file_name

        if os.path.exists(cache_file_name):
            print(f"load cache from {cache_file_name}")
            cached_file = open(cache_file_name, "rb")
            self.datasets = pickle.load(cached_file)
        else:
            # 1. 데이터 로드
            train = pd.read_csv(self.config.data.train_path)
            dev = pd.read_csv(self.config.data.dev_path)
            test = pd.read_csv(self.config.data.test_path)
            train = train.iloc[:300]
            datasets = {'train':train, 'dev':dev, 'test':test}

            # 2. preprocessing
            datasets = self.add_feature(datasets)

            # 3. postprocessing
            self.datasets = self.postprocessing(datasets)

            # 4. 저장
            cached_file = open(cache_file_name, "wb")
            pickle.dump(self.datasets, cached_file)


    def generate_img(self, df, split):
        folder = f'../datasets/{split}_imgs'
        file_ = f'{split}_0.png'
        if not os.path.exists(folder):
            print(f"can not find {folder}. make new one")
            os.mkdir(folder)
            for idx, row in tqdm(df.iterrows(), desc="generate_img", total=len(df)):
                file = row['uid']
                smiles = row['SMILES']
                m = Chem.MolFromSmiles(smiles)
                if m != None:
                    img = Draw.MolToImage(m, size=(300,300))
                    img.save(f'../datasets/{split}_imgs/{file}.png')

    def add_feature(self, datasets):
        for split in datasets.keys():
            self.generate_img(datasets[split], split)
            datasets[split] = self.parallelize_dataframe(self.processing_func, datasets[split])
            datasets[split].reset_index(drop=True, inplace=True)
        return datasets

    def parallelize_dataframe(self, func, df):
        #num_cores = os.cpu_count() # 첫번째 씨피유 말고는 다 None으로 반환하는 문제가 발생. 임시로 1로 설정
        num_cores=1
        df_split = np.array_split(df, num_cores)
        pool = Pool(num_cores)
        df = pd.concat(pool.map(func, df_split))
        pool.close()
        pool.join()
        return df

    def processing_func(self, df):
        
        if df.uid.iloc[0].find('train')==0:
            split='train'
        elif df.uid.iloc[0].find('dev')==0:
            split='dev'
        else:
            split='test'

        # 1. SDF피쳐(~20개) + mordred (null제거 후 ~1300개) + PatternFingerprint(~2000개) 피쳐를 합쳐서 'features'로 배정.(TODO : axis=1 drop nan )
        # + XYZ좌표 (2차원 리스트)는 일단 제외 (주석처리)
        features=[]
        not_loaded_lst = []
        # xyz_list = []
        for index, row in tqdm(df.iterrows(), desc=f"{split} : add SDF and PatternFingerprint feature", total=len(df)):
            sdf = [i for i in pybel.readfile('sdf',self.sdf_load(index, split))]
            if len(sdf)>0:
#                 xyz_element_list = []
#                 xyz_data = sdf[0].write("xyz").split('\n')[2:-1]
#                 for xyz in xyz_data:
#                   xyz = re.sub('[^\d.-]', ' ', xyz).split(' ')
#                   xyz = ' '.join(xyz).split()
#                   xyz = list(map(float, xyz))
#                   xyz_element_list.append(xyz)
#                 xyz_list.append(xyz_element_list)
                sdf = sdf[0].calcdesc()
                sdf_arr = np.array([v for k, v in sdf.items()])
            else:
                sdf_arr = np.array([np.nan for i in range(len(sdf_arr))])
                not_loaded_lst.append(index)

            pfp = Chem.MolFromSmiles(row['SMILES'])
            pfp = Chem.rdmolops.PatternFingerprint(pfp)
            pfp_arr = np.zeros((0,))
            DataStructs.ConvertToNumpyArray(pfp, pfp_arr)

            # + mordred feature
            calc = Calculator(descriptors, ignore_3D=False)
            mordred_df = calc.pandas([Chem.MolFromSmiles(row['SMILES'])], nproc = 1)

            mordred_df['Lipinski'] = mordred_df['Lipinski'].apply(lambda x: 1 if x == True else 0)
            mordred_df['GhoseFilter'] = mordred_df['GhoseFilter'].apply(lambda x: 1 if x == True else 0)

            self.str_to_float(mordred_df)

            mordred_arr = np.array(mordred_df.values[0])
            features.append(np.concatenate((sdf_arr, pfp_arr, mordred_arr), axis=0))
        
        if len(not_loaded_lst)!=0:
            print(f'sample # {not_loaded_lst} not loaded. So, delete them!!\n')
            df.drop(not_loaded_lst)

#        xyz_df = pd.DataFrame(np.array(xyz_list))
        features_df = pd.DataFrame(features).dropna(axis = 1)
#        features_df = pd.concat([features_df, xyz_df], axis = 1)
        #print(f"{split} features_df.isna().sum():", features_df.isna().sum())
        df['features'] = pd.Series([i for i in features_df.values])
        
        # 2. 그래프 추가
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
        


        # 3. input_ids, attention_mask 생성
        tokenized_series = df.apply(lambda x:self.tokenizer(x['SMILES'], max_length=self.config.data.max_seq_length, pad_to_max_length=True, truncation=True), axis=1)
        df['input_ids'] = [i['input_ids'] for i in tokenized_series]
        df['attention_mask'] = [i['attention_mask'] for i in tokenized_series]
        df.drop(['SMILES'], axis=1)

        # 4.labels 생성
        if split!='test':
            df['labels'] = df['S1_energy(eV)'] - df['T1_energy(eV)']
            df.drop(['S1_energy(eV)', 'T1_energy(eV)'], axis=1)

        return df

    def sdf_load(self, uid, split):
        return os.path.join('..','datasets',f'{split}_sdf',f'{split}_{uid}.sdf')

    # mordred 파트 null feature 제거 및 str -> float 변환
    def str_to_float(self, data):
        drop_feats = ['PNS', 'PP', 'DP', 'FNS', 'FPS', 'WNS', 'WPS', 'TASA', 'TPSA', 'RASA', 'RPSA', 'MAX', 'MIN',
                      'Geom', 'GRAV', 'Mor']
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype('float')
            for drop in drop_feats:
                if drop in col:
                    data.drop([col], axis=1, inplace=True)

    def postprocessing(self, datasets):
        for split in datasets.keys():
            datasets[split] = CustomDataset(datasets[split], self.tokenizer, split)
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

    


class CustomDataset(Dataset):
    loader_columns = [
        'uid', 'input_ids', 'attention_mask', 'features', 'labels'
    ]

    def __init__(self, df, tokenizer, split):
        """

        """
        self.split = split
        self.df = df
        self.imgs = (f'../datasets/{self.split}_imgs/'+df.uid+'.png').to_numpy()
            
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, i):
        img = cv2.imread(self.imgs[i]).astype(np.float32)/255
        img = np.transpose(img, (2,0,1))
        return_dict = {
                'features' : torch.tensor(self.df.loc[i,'features'], dtype=torch.float32),
                'img' : torch.tensor(img, dtype=torch.float32),
                #'graph' : torch.tensor(i),
                #'graph_fingerprints' : torch.tensor(self.df.loc[i,'graph_fingerprints'], dtype=torch.float32),
                #'graph_adjacency' : torch.tensor(self.df.loc[i,'graph_adjacency'], dtype=torch.float32),
                #'graph_molecular_size' : torch.tensor(self.df.loc[i,'graph_molecular_size'], dtype=torch.float32),
                'input_ids' : torch.tensor(self.df.loc[i,'input_ids'], dtype=torch.float32),
                'attention_mask' : torch.tensor(self.df.loc[i,'attention_mask'], dtype=torch.float32),
        }
        
        if self.split != 'test':
            return_dict['labels'] = torch.tensor(self.df.loc[i, 'labels'], dtype=torch.float32)
            return return_dict
        else:
            return return_dict
