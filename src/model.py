import numpy as np
from sklearn.ensemble import RandomForestClassifier

class MultiBinaryClassifier:
    def __init__(self, weights, seed):
        self.base_model_params = {
            'random_state': seed,
            'n_jobs': -1
        }


        self.weights = weights
        self.base_classifier = [RandomForestClassifier(class_weight=self.weights[i], **self.base_model_params) for i in range(4)]
        self.discriminat_model = RandomForestClassifier(**self.base_model_params)

    def fit(self, x_train, y_train):
        base_clfs_preds = []
        for i in range(4):
            self.base_classifier[i].fit(x_train, y_train[i])
            pred = self.base_classifier[i].predict(x_train)
            base_clfs_preds.append(pred)

        base_clfs_preds = np.vstack(base_clfs_preds).transpose()
        self.discriminat_model.fit(base_clfs_preds, y_train[-1])

    def predict(self, x):
        discriminat_input = []
        for i in range(4):
            pred = self.base_classifier[i].predict(x)
            discriminat_input.append(pred)

        discriminat_input = np.vstack(discriminat_input).transpose()
        final_pred = self.discriminat_model.predict(discriminat_input)

        return final_pred
