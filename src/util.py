import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_fscore_support
import tensorflow as tf
import seaborn as sns
from tensorflow import keras
from tqdm import tqdm
from sklearn.manifold import TSNE


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def get_data(path):
    df = pd.read_pickle(path)
    labels = df["job_state"]
    df = df.drop(["job_state"], axis=1)
    return df.to_numpy(), labels.to_numpy()


def print_confusion_matrix(labels, predictions, average, ax):
    cm_train = confusion_matrix(labels, predictions, normalize='all')

    if average == 'binary':
        display_labels = ['COMPLETED', 'FAILED']
    else:
        display_labels=['COMPLETED', 'FAILED', 'OUT_OF_MEMORY', 'TIMEOUT']

    ConfusionMatrixDisplay(confusion_matrix=cm_train ,display_labels=display_labels).plot(colorbar=False, ax=ax)

def evaluate_model(model, x_train, y_train, x_val, y_val, average="binary"):
    train_pred = model.predict(x_train)
    val_pred = model.predict(x_val)

    train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(y_train, train_pred, average=average)
    val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(y_val, val_pred, average=average)
    print(f"F1:\t\ttrain={train_f1:.3f}, validation={val_f1:.3f}")
    print(f"Recall:\t\ttrain={train_rec:.3f}, validation={val_rec:.3f}")
    print(f"Precision:\ttrain={train_prec:.3f}, validation={val_prec:.3f}")

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(13,7))
    print_confusion_matrix(y_train, train_pred, average, ax1)
    ax1.set_title("Confusion Matrix on train set")
    print_confusion_matrix(y_val, val_pred, average, ax2)
    ax2.set_title("Confusion Matrix on validation set")
    plt.show()


def build_cnn_model(input_shape, output_shape, stages, convs):
    model_in = keras.Input(shape=input_shape)
    x = model_in
    filters = 128
    for s in range(stages):
        for _ in range(convs):
            x = keras.layers.Conv1D(filters=filters, kernel_size=3, activation='relu',  strides=1, padding="causal")(x)
        x = keras.layers.MaxPooling1D()(x)
        filters *= 2
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    model_out = keras.layers.Dense(output_shape, activation="sigmoid")(x)
    model = keras.Model(model_in, model_out)
    return model


def model_tuner(models, x_train, y_train, x_val, y_val, metrics):
    train_f1_scores = []
    val_f1_scores = []
    for model in tqdm(models):
        model.fit(x_train, y_train)
        train_pred = model.predict(x_train)
        val_pred = model.predict(x_val)
        train_f1 = metrics(y_train, train_pred, beta=1.5, average="binary")
        val_f1 = metrics(y_val, val_pred, beta=1.5, average="binary")
        train_f1_scores.append(train_f1)
        val_f1_scores.append(val_f1)

    return train_f1_scores, val_f1_scores, np.subtract(train_f1_scores, val_f1_scores)


def show_space(data, labels, n_comp, x_coord_conflict_area=[0,0], y_coord_conflict_area=[0,0], x_coord_good=[0,0], y_coord_good=[0,0]):
    tsne = TSNE(n_components=n_comp)
    low_dim_data = tsne.fit_transform(data)
    tsne_df = pd.DataFrame(low_dim_data)
    if n_comp == 2:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax = sns.scatterplot(data=tsne_df, x=0, y=1, hue=labels)
        ax.set_yticks(np.arange(-120,120,5))
        ax.set_xticks(np.arange(-120,120,5))
        # plt.axvspan(-40, -30, color='red', alpha=0.5)
        # plt.ayvspan(25, 40, color='red', alpha=0.5)
        x_slice = abs(x_coord_conflict_area[1] - x_coord_conflict_area[0])
        y_slice = abs(y_coord_conflict_area[1] - y_coord_conflict_area[0])
        ax.add_patch(Rectangle((x_coord_conflict_area[0], y_coord_conflict_area[0]), x_slice, y_slice, linewidth=2, edgecolor='red', facecolor='none'))
        x_slice = abs(x_coord_good[1] - x_coord_good[0])
        y_slice = abs(y_coord_good[1] - y_coord_good[0])
        ax.add_patch(Rectangle((x_coord_good[0], y_coord_good[0]), x_slice, y_slice, linewidth=2, edgecolor='green', facecolor='none'))
        plt.xticks(rotation = "vertical")
        plt.grid()
        plt.show()
    else:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax = fig.add_subplot(111, projection = '3d')
        x = tsne_df[0]
        y = tsne_df[1]
        z = tsne_df[2]
        label = np.where(labels == "COMPLETED", "green", labels)
        label = np.where(label == "FAILED", "red", label)
        label = np.where(label == "TIMEOUT", "orange", label)
        label = np.where(label == "OUT_OF_MEMORY", "blue", label)
        ax.scatter(x, y, z, c=label)
        plt.legend(labels)
        plt.grid()
        plt.show()
    return tsne_df


def rf_feature_importance(model, x_train, y_train, x_test, y_test):
    local_ranking = {}
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary")
    print("-------------------------------------------------------------------")
    print(f"F1 on test set: {f1}")
    print(f"Recall on test set: {recall}")
    print(f"Precision on test set: {precision}")
    # cm = confusion_matrix(y_test, y_pred)
    # ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    importances = pd.Series(model.feature_importances_, index=x_train.columns)
    importances.sort_values(ascending=True, inplace=True)
    i = 0
    for key in importances.index:
      local_ranking.update({key : importances[i]})
      i = i + 1
    print("Local ranking: ", local_ranking)
    print("-------------------------------------------------------------------")
    return local_ranking