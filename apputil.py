import pandas as pd


class GroupEstimator:
    def __init__(self, estimate):
        if estimate not in ['mean', 'median']:
            raise ValueError("Invalid estimate type. Choose 'mean' or 'median'.")
        self.estimate = estimate
        self.group_estimates = None
    def fit(self, X, y):
        data = pd.concat([X, y], axis=1)
        if self.estimate == 'mean':
            self.group_estimates = data.groupby(X.columns.tolist()).mean().to_dict()
        elif self.estimate == 'median':
            self.group_estimates = data.groupby(X.columns.tolist()).median().to_dict()
    def predict(self, X):
        return None