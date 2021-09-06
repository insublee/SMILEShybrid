# SMILEShybrid
## 0. 준비
데이터는 datasets 폴더에 넣어주세요.

### use conda env
```
conda create -n smiles python=3.7
conda activate smiles
git clone git@github.com:insublee/SMILEShybrid.git
cd SMILEShybrid
pip install -e.
conda install -c conda-forge openbabel --yes
```

## 1. train & prediction
```
cd SMILEShybrid/execute
python train.py -c baseline.yaml
```
## 2. 개발
각자 브랜치 파서 개발후 풀리퀘스트 주세용
### 윤표
datamodule,  
models/MLP
### 정섭
datamodule GNN feature 추가,  
models/GNN
### 인섭
라이브러리, others
