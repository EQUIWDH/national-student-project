import h5py
import numpy as np
import csv
f=h5py.File("project\\DeepSurv.pytorch-master\\DeepSurv.pytorch-master\\data\\metabric\\metabric_IHC4_clinical_train_test.h5","r")

e = f['test']['e'][()].reshape(-1,1)
t = f['test']['t'][()].reshape(-1,1)
x = f['test']['x'][()]
np.savetxt("metabric_test.csv", np.concatenate((x,e,t),axis=1), delimiter=",")
file_path = "project\\DeepSurv.pytorch-master\\DeepSurv.pytorch-master\\data\\metabric\\metabric_IHC4_clinical_train_test.h5"
df = csv.read_csv(file_path, header=0)
df.columns = ["x1", "x2", "x3",'x4','x5','x6','x7','x8','x9','censor','days']
df.to_csv(file_path, index=False)
