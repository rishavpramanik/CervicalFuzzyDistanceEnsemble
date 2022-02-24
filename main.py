import argparse
import numpy as np
from utils.generate_csv import generate_csv
from utils.k_fold_splits import k_fold_splits
from utils.k_fold_separate import k_fold_separate

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default = 70,
                    help='Number of Epochs for training')
parser.add_argument('--path', type=str, default = './',
                    help='Path where the image data is stored')
parser.add_argument('--batch_size', type=int, default = 16,
                    help='Batch Size for Mini Batch Training')
parser.add_argument('--kfold', type=int, default = 5,
                    help='Number of folds for training')
parser.add_argument('--lr', type=float, default = 1e-4,
                    help='Learning rate for training')
args = parser.parse_args()

df = generate_csv(args.path)

y = np.array(list(df["class"]))
x = np.array(list( df["path"]))

files_for_train_x = []
files_for_validation_x = []
files_for_train_y = []
files_for_validation_y = []

k_fold_splits(x,y, files_for_train_x ,  files_for_validation_x ,
              files_for_train_y , files_for_validation_y,  n_splits = args.kfold ) # n_splits = 5

#N is the number of folds
N = len(files_for_train_x)
for i in range(0,N):
    k_fold_separate(files_for_train_x[i] , files_for_train_y[i] ,
                    files_for_validation_x[i] ,files_for_validation_y[i] ,
                    "InceptionV3" , "MobileNetV2" ,"InceptionResNetV2" ,i+1 ,
                    NUM_EPOCHS = args.num_epochs , train_batch=args.batch_size ,
                    validation_batch = args.batch_size, lr=args.lr)
