from msilib.schema import Directory
import os 
import pandas as pd 
from sklearn.utils import shuffle

def generate_csv():
    print("runing")
    uniques = ["Dyskeratotic" , "Koilocytotic" , "Metaplastic" , "Parabasal" , "SuperficialIntermediate"]
    dirs = ["train" , "test"]

    """
            +-- train
            |   +-- Dyskeratotic
            |   +-- Koilocytotic
            |   +-- Metaplastic
            |   +-- Parabasal
            |   +-- SuperficialIntermediate

            +-- test
            |   +-- Dyskeratotic
            |   +-- Koilocytotic
            |   +-- Metaplastic
            |   +-- Parabasal
            |   +-- SuperficialIntermediate

    
    """

    data = []
    for dir in dirs :
        for unique in uniques:
            directory = "data/SiPakMed/" + dir + "/" + unique    #required path 

            for filename in os.listdir(directory):
               
                path = directory + "/" + filename  #required path 
                data.append([ filename , path  , unique])

    df = pd.DataFrame(data, columns = ["filename" ,"path", "class"]) 
    df = shuffle(df)
    name = "csv_files/" + "Data-full"        #required path 
    df.to_csv(name, index = False )
    print("runing")

    return df    



