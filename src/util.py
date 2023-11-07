import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_fscore_support


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    # tf.random.set_seed(seed)
    # tf.keras.utils.set_random_seed(seed)
    # tf.config.experimental.enable_op_determinism()


def get_data(path):
    df = pd.read_pickle(path)
    labels = df["job_state"]
    df = df.drop(["job_state"], axis=1)
    return df.to_numpy(), labels.to_numpy()


def evaluate_model(model, x_train, x_val, y_train, y_val, average="binary"):
    train_predict = model.predict(x_train)
    val_predict = model.predict(x_val)

    t_precision, t_recall, t_f1, _ = precision_recall_fscore_support(y_train, train_predict, beta=1.5, average=average)
    v_precision, v_recall, v_f1, _ = precision_recall_fscore_support(y_val, val_predict, beta=1.5, average=average)
    print(f"F1:\n - train set:\t\t{t_f1:.3f}\n - validation set:\t{v_f1:.3f}")
    print(f"Recall:\n - train set:\t\t{t_recall:.3f}\n - validation set:\t{v_recall:.3f}")
    print(f"Precision:\n - train set:\t\t{t_precision:.3f}\n - validation set:\t{v_precision:.3f}")

    train_cm = confusion_matrix(y_train, train_predict)
    val_cm = confusion_matrix(y_val, val_predict)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ConfusionMatrixDisplay(confusion_matrix=train_cm).plot(ax=ax1, colorbar=False)
    ConfusionMatrixDisplay(confusion_matrix=val_cm).plot(ax=ax2,  colorbar=False)
    ax1.set_title("Confusion Matrix on train set")
    ax2.set_title("Confusion Matrix on val set")
    plt.show()

