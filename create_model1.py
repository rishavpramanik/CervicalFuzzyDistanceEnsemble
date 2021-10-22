"""
with only flatten layer 

"""


def create_model1(model_name):
    models = {"MobileNetV2": MobileNetV2_model, "InceptionV3": InceptionV3_model,
              "InceptionResNetV2": InceptionResNetV2_model}
    base_model = models[model_name]

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    my_model = tf.keras.models.clone_model(model)
    return my_model
