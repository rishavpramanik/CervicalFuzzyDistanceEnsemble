
from sklearn.model_selection import KFold

def k_fold_splits(x , y , files_for_train_x , files_for_validation_x , files_for_train_y , files_for_validation_y , n_splits = 5    ):

  

  kf = KFold(n_splits = n_splits)
  #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
  #fold_no = 1
  for train_index, val_index in kf.split(x):

      
     
    
      #it will split the entire data into 5 folds 
      x_train, x_val = x[train_index], x[val_index]
      y_train, y_val = y[train_index], y[val_index]
     



      # split the into 5 folds 

      files_for_train_x.append(x_train)
      files_for_validation_x.append(x_val)
      files_for_train_y.append(y_train)
      files_for_validation_y.append(y_val)
      #fold_no += 1 
