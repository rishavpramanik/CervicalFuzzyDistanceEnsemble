
"""
k_fold function gave us run time error in google colab 
So we need to make a function that can process one validation at a time for 3 models so will not give the run time error
"""


def k_fold_separate(x_train, y_train, x_val, y_val, model_name1, model_name2, model_name3, fold_no,  NUM_EPOCHS=70, train_batch=16, validation_batch=16):

    x_train, y_train, x_val, y_val = process_x(x_train), encode_y(
        y_train), process_x(x_val), encode_y(y_val)

    train = train_datagen.flow(x_train, y_train, batch_size=train_batch)
    validation = val_datagen.flow(x_val, y_val,
                                  batch_size=validation_batch)
    test = x_val

    print('------------------------------------------------------------------------')

    y_preds = []
    print()
    print(model_name1)
    print()

    model1 = create_model(model_name1)

    # Compile the model
    model1.compile(loss='sparse_categorical_crossentropy',
                   optimizer=tf.keras.optimizers.Adam(
                       learning_rate=0.0001, beta_1=0.9, beta_2=0.999, decay=0.0001),
                   metrics=['accuracy'])

    # Generate a print

    # Fit data to model
    history = model1.fit(x=train,
                         validation_data=validation,
                         epochs=NUM_EPOCHS
                         )

    # model save..
    model_saved_name = model_name1 + "_" + str(fold_no) + ".h5"

    model1.save(
        "/content/drive/MyDrive/SiPakMed/K_cross_val_models/" + model_saved_name)

    print(f'{model_saved_name} saved')

    # Generate generalization metrics
    scores = model1.evaluate(validation)
    print(
        f'Score for fold {fold_no}: {model1.metrics_names[0]} of {scores[0]}; {model1.metrics_names[1]} of {scores[1]*100}%')
    # predictions = model.predict()
    preds1 = model1.predict(test, batch_size=validation_batch)
    for pred in preds1:
        y_preds.append(np.argmax(pred))
    print('Accuracy Score: ', accuracy_score(y_val, y_preds))
    print('Precision Score(Class wise): ',
          precision_score(y_val, y_preds, average=None))
    print('Recall Score(Class wise): ',
          recall_score(y_val, y_preds, average=None))
    print('F1 Score(Class wise): ', f1_score(y_val, y_preds, average=None))
    print('Conf Matrix Score(Class wise):\n ',
          confusion_matrix(y_val, y_preds))

    y_preds = []
    print()
    print(model_name2)
    print()

    model2 = create_model(model_name2)

    # Compile the model
    model2.compile(loss='sparse_categorical_crossentropy',
                   optimizer=tf.keras.optimizers.Adam(
                       learning_rate=0.0001, beta_1=0.9, beta_2=0.999, decay=0.0001),
                   metrics=['accuracy'])

    # Generate a print

    # Fit data to model
    history = model2.fit(x=train,
                         validation_data=validation,
                         epochs=NUM_EPOCHS
                         )

    # model save..
    model_saved_name = model_name2 + "_" + str(fold_no) + ".h5"

    model2.save(
        "/content/drive/MyDrive/SiPakMed/K_cross_val_models/" + model_saved_name)
    print(f'{model_saved_name} saved')

    # Generate generalization metrics
    scores = model2.evaluate(validation)
    print(
        f'Score for fold {fold_no}: {model2.metrics_names[0]} of {scores[0]}; {model2.metrics_names[1]} of {scores[1]*100}%')
    # predictions = model.predict()
    preds2 = model2.predict(test, batch_size=validation_batch)
    for pred in preds2:
        y_preds.append(np.argmax(pred))

    print('Accuracy Score: ', accuracy_score(y_val, y_preds))
    print('Precision Score(Class wise): ',
          precision_score(y_val, y_preds, average=None))
    print('Recall Score(Class wise): ',
          recall_score(y_val, y_preds, average=None))
    print('F1 Score(Class wise): ', f1_score(y_val, y_preds, average=None))
    print('Conf Matrix Score(Class wise):\n ',
          confusion_matrix(y_val, y_preds))

    y_preds = []
    print()
    print(model_name3)
    print()

    model3 = create_model(model_name3)

    # Compile the model
    model3.compile(loss='sparse_categorical_crossentropy',
                   optimizer=tf.keras.optimizers.Adam(
                       learning_rate=0.0001, beta_1=0.9, beta_2=0.999, decay=0.0001),
                   metrics=['accuracy'])

    # Generate a print

    # Fit data to model
    history = model3.fit(x=train,
                         validation_data=validation,
                         epochs=NUM_EPOCHS
                         )

    # model save..
    model_saved_name = model_name3 + "_" + str(fold_no) + ".h5"

    model3.save(
        "/content/drive/MyDrive/SiPakMed/K_cross_val_models/" + model_saved_name)

    print(f'{model_saved_name} saved')

    # Generate generalization metrics
    scores = model3.evaluate(validation)
    print(
        f'Score for fold {fold_no}: {model3.metrics_names[0]} of {scores[0]}; {model3.metrics_names[1]} of {scores[1]*100}%')
    # predictions = model.predict()
    preds3 = model3.predict(test, batch_size=validation_batch)
    for pred in preds3:
        y_preds.append(np.argmax(pred))

    print('Accuracy Score: ', accuracy_score(y_val, y_preds))
    print('Precision Score(Class wise): ',
          precision_score(y_val, y_preds, average=None))
    print('Recall Score(Class wise): ',
          recall_score(y_val, y_preds, average=None))
    print('F1 Score(Class wise): ', f1_score(y_val, y_preds, average=None))
    print('Conf Matrix Score(Class wise):\n ',
          confusion_matrix(y_val, y_preds))

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
