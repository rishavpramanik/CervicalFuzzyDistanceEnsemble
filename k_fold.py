

"""
The below function can do the entire crosss validation for 3 models and can do the ensemble

It can process the all data(for k folds , where k = 5 ) and give us the results .The problem with this method it
will take more than 12 hours to processs so it will give us run time error in google colab https://research.google.com/colaboratory/faq.html

"""


def k_fold(X, y, model_name1, model_name2, model_name3, files_for_train,  files_for_validation, NUM_EPOCHS=100, train_batch=16, validation_batch=16):

    acc_per_fold1 = []
    loss_per_fold1 = []
    acc_per_fold2 = []
    loss_per_fold2 = []
    acc_per_fold3 = []
    loss_per_fold3 = []

    fold_no = 1
    kf = KFold(n_splits=5)
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

    for train_index, val_index in kf.split(X):

        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')
        # it will split the entire data into 5 folds
        y_preds = []
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # in cross validation where k = 5 we split the dataset into 5 splits
        # this 2 arrays will store files_for_train ,  files_for_validation  those splits , any case if we need this .

        files_for_train.append(X_train)
        files_for_validation.append(X_val)

        # data processing .............
        train_data = create_data_batches(X_train, y_train, train_batch)
        val_data = create_data_batches(
            X_val, y_val, validation_batch, valid_data=True)
        test_data = create_data_batches(
            X_val, validation_batch,  test_data=True)

        # it will store the files in training  and validation
        # files_for_train.append(X_train)
        # files_for_validation.append(X_val)

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
        history = model1.fit(x=train_data,
                             validation_data=val_data,
                             epochs=NUM_EPOCHS
                             )

        # model save..
        model_saved_name = model_name1 + "_" + str(fold_no) + ".h5"

        model1.save(
            "/content/drive/MyDrive/SiPakMed/K_cross_val_models/" + model_saved_name)

        print(f'{model_saved_name} saved')

        # Generate generalization metrics
        scores = model1.evaluate(val_data)
        print(
            f'Score for fold {fold_no}: {model1.metrics_names[0]} of {scores[0]}; {model1.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold1.append(scores[1] * 100)
        loss_per_fold1.append(scores[0])
        # predictions = model.predict()
        preds1 = model1.predict(test_data)
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
        history = model2.fit(x=train_data,
                             validation_data=val_data,
                             epochs=NUM_EPOCHS
                             )

        # model save..
        model_saved_name = model_name2 + "_" + str(fold_no) + ".h5"

        model2.save(
            "/content/drive/MyDrive/SiPakMed/K_cross_val_models/" + model_saved_name)
        print(f'{model_saved_name} saved')

        # Generate generalization metrics
        scores = model2.evaluate(val_data)
        print(
            f'Score for fold {fold_no}: {model2.metrics_names[0]} of {scores[0]}; {model2.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold2.append(scores[1] * 100)
        loss_per_fold2.append(scores[0])
        # predictions = model.predict()
        preds2 = model2.predict(test_data)
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
        history = model3.fit(x=train_data,
                             validation_data=val_data,
                             epochs=NUM_EPOCHS
                             )

        # model save..
        model_saved_name = model_name3 + "_" + str(fold_no) + ".h5"

        model3.save(
            "/content/drive/MyDrive/SiPakMed/K_cross_val_models/" + model_saved_name)

        print(f'{model_saved_name} saved')

        # Generate generalization metrics
        scores = model3.evaluate(val_data)
        print(
            f'Score for fold {fold_no}: {model3.metrics_names[0]} of {scores[0]}; {model3.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold3.append(scores[1] * 100)
        loss_per_fold3.append(scores[0])
        # predictions = model.predict()
        preds3 = model3.predict(test_data)
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
        print('Post Enemble Accuracy Score: ',
              accuracy_score(y_val, ensem_pred))
        print('Post Enemble Precision Score(Class wise): ',
              precision_score(y_val, ensem_pred, average=None))
        print('Post Enemble Recall Score(Class wise): ',
              recall_score(y_val, ensem_pred, average=None))
        print('Post Enemble F1 Score(Class wise): ',
              f1_score(y_val, ensem_pred, average=None))
        print('Post Enemble Conf Matrix Score(Class wise):\n ',
              confusion_matrix(y_val, ensem_pred))
        # Increase fold number
        fold_no += 1

    # == Provide average scores ==
    print('------------------------------------------------------------------------')

    print('Score per fold')
    print(model_name1)
    for i in range(0, len(acc_per_fold1)):
        print('------------------------------------------------------------------------')
        print(
            f'> Fold {i+1} - Loss: {loss_per_fold1[i]} - Accuracy: {acc_per_fold1[i]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold1)} (+- {np.std(acc_per_fold1)})')
    print(f'> Loss: {np.mean(loss_per_fold1)}')
    print('------------------------------------------------------------------------')

    print('------------------------------------------------------------------------')
    print('Score per fold')
    print(model_name2)
    for i in range(0, len(acc_per_fold2)):
        print('------------------------------------------------------------------------')
        print(
            f'> Fold {i+1} - Loss: {loss_per_fold2[i]} - Accuracy: {acc_per_fold2[i]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold2)} (+- {np.std(acc_per_fold2)})')
    print(f'> Loss: {np.mean(loss_per_fold2)}')
    print('------------------------------------------------------------------------')

    print('------------------------------------------------------------------------')
    print('Score per fold')
    print(model_name3)
    for i in range(0, len(acc_per_fold3)):
        print('------------------------------------------------------------------------')
        print(
            f'> Fold {i+1} - Loss: {loss_per_fold3[i]} - Accuracy: {acc_per_fold3[i]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold3)} (+- {np.std(acc_per_fold3)})')
    print(f'> Loss: {np.mean(loss_per_fold3)}')
    print('------------------------------------------------------------------------')
