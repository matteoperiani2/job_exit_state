import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_fscore_support
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm


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


def evaluate_model(model, x_test, y_test, average="binary"):
    pred = model.predict(x_test)

    prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred, beta=1.5, average=average)
    print(f"F1:\t\t{f1:.3f}")
    print(f"Recall:\t\t{rec:.3f}")
    print(f"Precision:\t{prec:.3f}")

    cm = confusion_matrix(y_test, pred)

    ConfusionMatrixDisplay(confusion_matrix=cm).plot(colorbar=False)
    plt.title("Confusion Matrix on test set")
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


def max_depth_dt_tuning(models, samples, labels, cv_fn):
    cv_scores_list = []
    cv_scores_std = []
    cv_scores_mean = []
    for model in tqdm(models):
        cv_scores = cv_fn(model, samples, labels, cv=5, scoring="f1")
        cv_scores_list.append(cv_scores)
        cv_scores_mean.append(cv_scores.mean())
        cv_scores_std.append(cv_scores.std())
    cv_scores_mean = np.array(cv_scores_mean)
    cv_scores_std = np.array(cv_scores_std)

    return cv_scores_mean, cv_scores_std

def plot_max_depth_tuning(depths, cv_scores_mean, cv_scores_std):
    fig, ax = plt.subplots(1,1, figsize=(15,5))
    ax.plot(depths, cv_scores_mean, '-o', label='mean cross-validation f1', alpha=0.9)
    ax.fill_between(depths, cv_scores_mean-2*cv_scores_std, cv_scores_mean+2*cv_scores_std, alpha=0.2)
    ylim = plt.ylim()
    ax.set_title('F1 per decision tree depth on training data', fontsize=16)
    ax.set_xlabel('Tree depth', fontsize=14)
    ax.set_ylabel('F1', fontsize=14)
    ax.set_ylim(ylim)
    ax.set_xticks(range(1, max(depths), 5))
    ax.legend()
