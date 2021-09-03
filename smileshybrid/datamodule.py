import os
import pickle
from tqdm import tqdm
from multiprocessing import Pool

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import  Draw

from openbabel import pybel
import deepchem as dc

import cv2

class Datamodule(object):
    """
    TODO : config화, 주석달기, 라이브러리화
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.train_batch_size = 40
        self.dev_batch_size = 40
        self.test_batch_size = 40
        self.max_seq_length = 200
        
        # 0. 캐시 확인
        cache_file_name='./data/datasets.pkl'

        if os.path.exists(cache_file_name):
            print(f"load cache from {cache_file_name}")
            cached_file = open(cache_file_name, "rb")
            self.datasets = pickle.load(cached_file)
        else:
            # 1. 데이터 로드
            train = pd.read_csv('./data/train.csv')
            dev = pd.read_csv("./data/dev.csv")
            test = pd.read_csv("./data/test.csv")
            #train = train.iloc[:2000]
            datasets = {'train':train, 'dev':dev, 'test':test}

            # 2. preprocessing
            datasets = self.add_feature(datasets)

            # 3. postprocessing
            self.datasets = self.postprocessing(datasets)

            # 4. 저장
            cached_file = open(cache_file_name, "wb")
            pickle.dump(self.datasets, cached_file)


    def generate_img(self, df, split):
        folder = f'./data/{split}_imgs'
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
                    img.save(f'./data/{split}_imgs/{file}.png')

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

        # 1. SDF피쳐(~20개)와 PatternFingerprint(~2000개) 피쳐를 합쳐서 'features'로 배정.(TODO : axis=1 drop nan )
        features=[]
        not_loaded_lst = []
        for index, row in tqdm(df.iterrows(), desc=f"{split} : add SDF and PatternFingerprint feature", total=len(df)):
            sdf = [i for i in pybel.readfile('sdf',self.sdf_load(index, split))]
            if len(sdf)>0:
                sdf = sdf[0].calcdesc()
                sdf_arr = np.array([v for k,v in sdf.items()])
            else:
                sdf_arr = np.array([np.nan for i in range(len(sdf_arr))])
                not_loaded_lst.append(index)
            
            pfp = Chem.MolFromSmiles(row['SMILES'])
            pfp = Chem.rdmolops.PatternFingerprint(pfp)
            pfp_arr = np.zeros((0,))
            DataStructs.ConvertToNumpyArray(pfp, pfp_arr)

            features.append(np.concatenate((sdf_arr, pfp_arr), axis=0))
        
        if len(not_loaded_lst)!=0:
            print(f'sample # {not_loaded_lst} not loaded. \n')

        features_df = pd.DataFrame(features).dropna(axis = 1)
        #print(f"{split} features_df.isna().sum():", features_df.isna().sum())
        df['features'] = pd.Series([i for i in features_df.values])

        # 2. 그래프 추가
        #featurizer = dc.feat.ConvMolFeaturizer()
        featurizer = dc.feat.MolGraphConvFeaturizer()
        df['graph'] = featurizer.featurize(df['SMILES'])

        # 3. input_ids, attention_mask 생성
        tokenized_series = df.apply(lambda x:self.tokenizer(x['SMILES'], max_length=self.max_seq_length, pad_to_max_length=True, truncation=True), axis=1)
        df['input_ids'] = [i['input_ids'] for i in tokenized_series]
        df['attention_mask'] = [i['attention_mask'] for i in tokenized_series]
        #df.drop(['SMILES'], axis=1)

        # 4.labels 생성
        if split!='test':
            df['labels'] = df['S1_energy(eV)'] - df['T1_energy(eV)']
            #df.drop(['S1_energy(eV)', 'T1_energy(eV)'], axis=1)

        return df

    def sdf_load(self, uid, split):
        return os.path.join('.','data',f'{split}_sdf',f'{split}_{uid}.sdf')

    def postprocessing(self, datasets):
        for split in datasets.keys():
            datasets[split] = CustomDataset(datasets[split], self.tokenizer, split)
        return datasets

    def train_dataloader(self):
        return DataLoader(self.datasets['train'], batch_size=self.train_batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.datasets['dev'], batch_size=self.dev_batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.datasets['test'], batch_size=self.test_batch_size, num_workers=4)


class CustomDataset(Dataset):
    loader_columns = [
        'uid', 'input_ids', 'attention_mask', 'features', 'graph', 'labels'
    ]

    def __init__(self, df, tokenizer, split):
        """

        """
        self.split = split
        self.df = df
        self.imgs = (f'./data/{self.split}_imgs/'+df.uid+'.png').to_numpy()
            
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, i):
        img = cv2.imread(self.imgs[i]).astype(np.float32)/255
        img = np.transpose(img, (2,0,1))
        return_dict = {
                'features' : torch.tensor(self.df.loc[i,'features'], dtype=torch.float32),
                'img' : torch.tensor(img, dtype=torch.float32),
                #'graph' : self.df.loc[i, 'graph'],
                'graph' : torch.tensor(i),
                'input_ids' : torch.tensor(self.df.loc[i,'input_ids'], dtype=torch.float32),
                'attention_mask' : torch.tensor(self.df.loc[i,'attention_mask'], dtype=torch.float32),
        }
        
        if self.split != 'test':
            return_dict['labels'] = torch.tensor(self.df.loc[i, 'labels'], dtype=torch.float32)
            return return_dict
        else:
            return return_dict
