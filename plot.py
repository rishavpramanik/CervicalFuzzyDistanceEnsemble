def plot(x_train, y_train, x_val, y_val, model_name1, model_name2, model_name3, fold_no,  NUM_EPOCHS=70, train_batch=16, validation_batch=16):

    x_train, y_train, x_val, y_val = process_x(x_train), encode_y(
        y_train), process_x(x_val), encode_y(y_val)

    train = train_datagen.flow(x_train, y_train, batch_size=train_batch)
    validation = val_datagen.flow(x_val, y_val,
                                  batch_size=validation_batch)
    test = x_val

    print('------------------------------------------------------------------------')

    print()
    print(model_name1)
    print()

    model1 = create_model(model_name1)

    model1.compile(loss='sparse_categorical_crossentropy',
                   optimizer=tf.keras.optimizers.Adam(
                       learning_rate=0.0001, beta_1=0.9, beta_2=0.999, decay=0.0001),
                   metrics=['accuracy'])

    history1 = model1.fit(x=train,
                          validation_data=validation,
                          epochs=NUM_EPOCHS

                          )

    # convert the history.history dict to a pandas DataFrame:
    hist_df = pd.DataFrame(history1.history)

    # https://stackoverflow.com/questions/41061457/keras-how-to-save-the-training-history-attribute-of-the-history-object
    # it will save history as csv
    hist_csv_file = "history_" + model_name1 + "_" + str(fold_no) + ".csv"
    filepath = "/content/drive/MyDrive/SiPakMed/history/" + hist_csv_file
    with open(filepath, mode='w') as f:
        hist_df.to_csv(f)

    plot_graph(history1, model_name1, fold_no)

    print('------------------------------------------------------------------------')

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
    history2 = model2.fit(x=train,
                          validation_data=validation,
                          epochs=NUM_EPOCHS

                          )
    hist_df = pd.DataFrame(history2.history)

    hist_csv_file = "history_" + model_name2 + "_" + str(fold_no) + ".csv"
    filepath = "/content/drive/MyDrive/SiPakMed/history/" + hist_csv_file
    with open(filepath, mode='w') as f:
        hist_df.to_csv(f)

    plot_graph(history2, model_name2, fold_no)
    print('------------------------------------------------------------------------')

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
    history3 = model3.fit(x=train,
                          validation_data=validation,
                          epochs=NUM_EPOCHS

                          )

    hist_df = pd.DataFrame(history3.history)
    hist_csv_file = "history_" + model_name3 + "_" + str(fold_no) + ".csv"
    filepath = "/content/drive/MyDrive/SiPakMed/history/" + hist_csv_file
    with open(filepath, mode='w') as f:
        hist_df.to_csv(f)
    plot_graph(history3, model_name3, fold_no)
