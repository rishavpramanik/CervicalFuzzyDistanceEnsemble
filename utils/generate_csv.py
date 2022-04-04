import os 
import pandas as pd 
from sklearn.utils import shuffle

def generate_csv(path):
    print("CSV being generated")
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
    #Above is the expected directory structure

    data = []
    for dir in dirs :
        for unique in uniques:
            directory = path + "/" + dir + "/" + unique    #required path 

            for filename in os.listdir(directory):
               
                paths = directory + "/" + filename  #required path 
                data.append([ filename , paths  , unique])

    df = pd.DataFrame(data, columns = ["filename" ,"path", "class"]) 
    df = shuffle(df)
    name = "csv_files/" + "Data-full"        #required path 
    df.to_csv(name, index = False)
    print("Generation Complete")
    return df
