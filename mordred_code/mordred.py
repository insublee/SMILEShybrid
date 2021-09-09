!pip install mordred

from mordred import Calculator, descriptors

# 아래의 과정을 통해 row에 해당하는 SMILES의 feature 1862개 feature dataframe 생성
calc = Calculator(descriptors, ignore_3D=False)
mordred_df = calc.pandas([Chem.MolFromSmiles(x) for x in df.SMILES])

# Boolean 데이터 변환
mordred_df['Lipinski'] = mordred_df['Lipinski'].apply(lambda x: 1 if x == True else 0)
mordred_df['GhoseFilter'] = mordred_df['GhoseFilter'].apply(lambda x: 1 if x == True else 0)

# drop할 컬럼 리스트 (path 지정 필요)
with open(path + "drop_mord_feats.pickle","rb") as fr:
    drop_mord_feats = pickle.load(fr)

mordred_df.drop(drop_mord_feats, axis = 1, inplace = True)

# str 데이터가 섞여있는 컬럼 형변환 (기존 str 데이터는 0으로 치환)
def str_to_float(data):
  for col in tqdm(data.columns):
    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype('float')

str_to_float(mordred_df)
