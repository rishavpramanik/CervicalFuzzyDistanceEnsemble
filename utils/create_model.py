import tensorflow as tf
from tensorflow.keras.models import Model

def create_model(model_name,IMG_SIZE = 256, output = 5):


    IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)  # IMG_SIZE = 256
    if(model_name == "MobileNetV2" ):

        model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                            include_top=False,
                                                            weights='imagenet')
    elif(model_name == "InceptionV3"):
        model = tf.keras.applications.inception_v3.InceptionV3(input_shape=IMG_SHAPE,
                                                                        include_top=False,
                                                                        weights='imagenet')
        
    elif(model_name == "InceptionResNetV2"):
        model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(input_shape=IMG_SHAPE,
                                                                                        include_top=False,
                                                                                        weights='imagenet')
    else:
        return        

    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(model.output)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(100, activation='relu')(x)
    x = tf.keras.layers.Dense(output, activation='softmax')(x)

    model = Model(inputs=model.input, outputs=x)

    my_model = tf.keras.models.clone_model(model)
    return my_model
