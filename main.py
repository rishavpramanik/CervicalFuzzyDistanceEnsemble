import numpy as np
from utils.generate_csv import generate_csv 
from utils.k_fold_splits import k_fold_splits 
from utils.k_fold_separate import k_fold_separate 

df = generate_csv()

y = np.array([i for i in df["class"]])
x = np.array([i for i in df["path"]]) 

files_for_train_x = []
files_for_validation_x = [] 
files_for_train_y = []
files_for_validation_y = []

k_fold_splits(x,y, files_for_train_x ,  files_for_validation_x , files_for_train_y , files_for_validation_y ) # n_splits = 5 

n = len(files_for_train_x)
for i in range(0,n):

     k_fold_separate(files_for_train_x[i] , files_for_train_y[i] , files_for_validation_x[i] ,files_for_validation_y[i] , "InceptionV3" , "MobileNetV2" ,"InceptionResNetV2" ,i+1 , NUM_EPOCHS = 1) 
