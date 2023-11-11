import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, precision_recall_fscore_support
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


def evaluate_model(model, x_train, y_train, x_val, y_val, average="binary"):
    train_pred = model.predict(x_train)
    val_pred = model.predict(x_val)

    train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(y_train, train_pred, beta=1.5, average=average)
    val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(y_val, val_pred, beta=1.5, average=average)
    print(f"F1:\t\ttrain={train_f1:.3f}, validation={val_f1:.3f}")
    print(f"Recall:\t\ttrain={train_rec:.3f}, validation={val_rec:.3f}")
    print(f"Precision:\ttrain={train_prec:.3f}, validation={val_prec:.3f}")

    cm_train = confusion_matrix(y_train, train_pred)
    cm_val = confusion_matrix(y_val, val_pred)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,7))
    ConfusionMatrixDisplay(confusion_matrix=cm_train).plot(colorbar=False, ax=ax1)
    ConfusionMatrixDisplay(confusion_matrix=cm_val).plot(colorbar=False, ax=ax2)
    ax1.set_title("Confusion Matrix on train set")
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


def tune_regressor(regs, samples, labels, cv_fn):
    cv_scores_list = []
    cv_scores_std = []
    cv_scores_mean = []
    for model in tqdm(regs):
        cv_scores = cv_fn(model, samples, labels, cv=5, scoring="f1")
        cv_scores_list.append(cv_scores)
        cv_scores_mean.append(cv_scores.mean())
        cv_scores_std.append(cv_scores.std())
    cv_scores_mean = np.array(cv_scores_mean)
    cv_scores_std = np.array(cv_scores_std)

    return cv_scores_mean, cv_scores_std