import pandas as pd
import numpy as np


class GroupEstimate:
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
        results = []
        missing_count = 0
        for _, row in X.iterrows():
            key = tuple(row)
            if key in self.group_estimates:
                results.append(self.group_estimates[key])
            else:
                results.append(np.nan)
                missing_count += 1
        if missing_count > 0:
            print(f"{missing_count} missing groups found during prediction.")
        return np.array(results)