def plot_confusion(x_val, y_val,  model_name1, model_name2, model_name3, fold_no, validation_batch=16):

    x_val, y_val = process_x(x_val), encode_y(y_val)

    test = x_val

    print("processing.... ", fold_no)

    model_saved_name = model_name1 + "_" + str(fold_no) + ".h5"
    model1 = tf.keras.models.load_model(
        "/content/drive/MyDrive/SiPakMed/K_cross_val_models/" + model_saved_name)

    # Generate generalization metrics

    preds1 = model1.predict(test, batch_size=validation_batch)

    model_saved_name = model_name2 + "_" + str(fold_no) + ".h5"
    model2 = tf.keras.models.load_model(
        "/content/drive/MyDrive/SiPakMed/K_cross_val_models/" + model_saved_name)

    preds2 = model2.predict(test, batch_size=validation_batch)

    model_saved_name = model_name3 + "_" + str(fold_no) + ".h5"
    model3 = tf.keras.models.load_model(
        "/content/drive/MyDrive/SiPakMed/K_cross_val_models/" + model_saved_name)

    preds3 = model3.predict(test, batch_size=validation_batch)

    sns.set(font_scale=1.5)

    ensem_pred = fuzzy_dist(preds1, preds2, preds3)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax = sns.heatmap(confusion_matrix(y_val, ensem_pred),
                     annot=True,
                     cmap="YlGnBu",
                     cbar=False,
                     fmt='g')
    plt.xlabel("True label ")
    plt.ylabel("Predicted label")

    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    figure = ax.get_figure()
    path = "/content/drive/MyDrive/SiPakMed/pic/" + \
        "Ensemblefold_" + str(fold_no) + ".png"
    figure.savefig(path, dpi=400)
