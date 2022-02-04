from image_processing import process_x,encode_y
def essemble_indi(x_val, y_val,  model_name1, model_name2, model_name3, fold_no, validation_batch=16):

    x_val, y_val = process_x(x_val), encode_y(y_val)

    test = x_val

    model_saved_name = model_name1 + "_" + str(fold_no) + ".h5"
    model1 = tf.keras.models.load_model(MODEL1_PATH)

    # Generate generalization metrics

    preds1 = model1.predict(test, batch_size=validation_batch)

    model_saved_name = model_name2 + "_" + str(fold_no) + ".h5"
    model2 = tf.keras.models.load_model(MODEL2_PATH)

    preds2 = model2.predict(test, batch_size=validation_batch)

    model_saved_name = model_name3 + "_" + str(fold_no) + ".h5"
    model3 = tf.keras.models.load_model(MODEL3_PATH)

    preds3 = model3.predict(test, batch_size=validation_batch)

    ensem_pred = fuzzy_dist(preds1, preds2, preds3)
    print('Post Enemble Accuracy Score: ', accuracy_score(y_val, ensem_pred))
    print('Post Enemble Precision Score(Class wise): ',
          precision_score(y_val, ensem_pred, average=None))
    print('Post Enemble Recall Score(Class wise): ',
          recall_score(y_val, ensem_pred, average=None))
    print('Post Enemble F1 Score(Class wise): ',
          f1_score(y_val, ensem_pred, average=None))
    print('Post Enemble Conf Matrix Score(Class wise):\n ',
          confusion_matrix(y_val, ensem_pred))
