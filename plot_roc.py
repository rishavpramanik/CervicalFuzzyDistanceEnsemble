def plot_roc(x_val, y_val,  model_name1, model_name2, model_name3, fold_no, validation_batch=16):

    x_val, y_val = process_x(x_val), encode_y(y_val)

    test = x_val

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

    ensem_pred = fuzzy_dist(preds1, preds2, preds3)

    # one hot encoding
    ensem_pred = to_categorical(ensem_pred, num_classes=5)
    y_val = to_categorical(y_val, num_classes=5)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(5):
        fpr[i], tpr[i], _ = roc_curve(y_val[:, i], ensem_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # # Plot of a ROC curve for a specific class

    # for i in range(5):
    plt.figure(figsize=(15, 10))
    plt.plot(fpr[0], tpr[0], label='ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot(fpr[1], tpr[1], label='ROC curve (area = %0.2f)' % roc_auc[1])
    plt.plot(fpr[2], tpr[2], label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot(fpr[3], tpr[3], label='ROC curve (area = %0.2f)' % roc_auc[3])
    plt.plot(fpr[4], tpr[4], label='ROC curve (area = %0.2f)' % roc_auc[4])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.legend(loc="lower right")
    filepath = "/content/drive/MyDrive/SiPakMed/pic/roc_curve_5_class_final.png"
    plt.savefig(filepath)
    plt.show()
