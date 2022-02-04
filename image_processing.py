import numpy as np

#"Dyskeratotic" , "Koilocytotic" , "Metaplastic" , "Parabasal" , "SuperficialIntermediate"

# this function will encode the labels
# for five classes
def encode_y(y):
    Y = []
    for i in y:
        if(i == "Dyskeratotic"):
            Y.append(0)
        elif(i == "Koilocytotic"):
            Y.append(1)
        if(i == "Metaplastic"):
            Y.append(2)
        if(i == "Parabasal"):
            Y.append(3)
        if(i == "SuperficialIntermediate"):
            Y.append(4)

    return np.array(Y).astype("float32")

# convert file paths info nums
# then normalize


def process_x(x):
    return np.array([imread(i) for i in x]).astype("float32") / 255.0


# for two classes
def encode_y1(y):
    Y = []
    for i in y:
        if(i == "normal"):
            Y.append(0)
        if(i == "abnormal"):
            Y.append(1)

    return np.array(Y).astype("float32")
