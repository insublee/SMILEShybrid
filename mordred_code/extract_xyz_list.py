# 기존 datamodule 코드에 3D좌표 리스트 추가
xyz_list = []

for index, row in tqdm(df.iterrows(), desc=f"{split} : add SDF and PatternFingerprint feature", total=len(df)):
    sdf = [i for i in pybel.readfile('sdf',self.sdf_load(index, split))]
    if len(sdf)>0:
        xyz_data = sdf[0].write("xyz").split('\n')
        xyz_list.append(xyz_data)
    else:
        pass
    
# pickle 저장 (경로 지정 필요)    
with open(path + "xyz_list.pickle","wb") as fw:
    pickle.dump(xyz_list, fw)
