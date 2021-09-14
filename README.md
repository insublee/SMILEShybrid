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
## 3. 업데이트 사항입니다. 
1. train+dev 한담에 셔플 후 0.05로 다시 분리해주었어요. 이제 발리데이션 로스 떨어지는거 보입니다.
2. datamodule 리뉴얼 싹 해서 성능이랑 가독성 좋아졌어요.
3. 기존에 train_img, dev_img 이런식으로 폴더 각각이였는데 train, dev, test 다 한 폴더에(imgs) 합쳐놨어요. 없으면 직접 합치기 ㄱㄱ
4. datasets.pkl 드라이브에 업로드 해놨어요.
5. 파이토치 라이트닝 모듈 상속받아서 모델 작성하시면 됩니다. init이랑 forward만 구현하시면 뚝딱 돌아갈거에요.
6. master branch로 초기화 해주세요. 이제 여기서 바꿔나가면 됩니다.(git pull origin main)
