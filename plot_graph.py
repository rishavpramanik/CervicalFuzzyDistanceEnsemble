def plot_graph(history, model_name, fold_no):
    # plotting the figure for accuracy
    plt.figure(figsize=(15, 10))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    tittle = "Model Accuracy " + model_name
    plt.title(tittle, fontsize=20)
    plt.ylabel('Accuracy', fontsize=15)
    plt.xlabel('Epoch', fontsize=15)
    plt.legend(['train', 'test'], loc='lower right')
    filepath = "/content/drive/MyDrive/SiPakMed/pic/" + model_name + \
        "_" + str(fold_no) + "_accuracy_train_val" + ".png"
    plt.savefig(filepath)
    plt.show()

    # plotting the figure for loss
    plt.figure(figsize=(15, 10))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    tittle = "Model Loss " + model_name
    plt.title(tittle, fontsize=20)
    plt.ylabel('Loss', fontsize=15)
    plt.xlabel('Epoch', fontsize=15)
    plt.legend(['train', 'test'], loc='upper right')
    filepath = "/content/drive/MyDrive/SiPakMed/pic/" + \
        model_name + "_" + str(fold_no) + "_loss_train_val" + ".png"
    plt.savefig(filepath)
    plt.show()
