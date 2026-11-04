import pandas as pd
import numpy as np


class GroupEstimate:

    def __init__(self, estimate):
        """ Initialize the GroupEstimate class """
        # Validate the estimate type
        if estimate not in ['mean', 'median']:
            # Raise an error if the estimate type is invalid
            raise ValueError("Invalid estimate type. Choose 'mean' or 'median'.")
        # Store the estimate type and initialize group estimates
        self.estimate = estimate
        self.group_estimates = None

    def fit(self, X, y):
        """ Function to fit the GroupEstimate model to the data """
        # Combine X and y into a single DataFrame
        y = pd.Series(y, index=X.index, name="target")
        data = pd.concat([X, y], axis=1)
        # Group by the columns in X and calculate the specified estimate
        if self.estimate == 'mean':
            # Calculate the mean of the target variable for each group
            self.group_estimates = data.groupby(X.columns.tolist())["target"].mean()
        elif self.estimate == 'median':
            # Calculate the median of the target variable for each group
            self.group_estimates = data.groupby(X.columns.tolist())["target"].median()
        # Convert the group estimates to a dictionary with tuple keys
        self.group_estimates = self.group_estimates.to_dict()

    def predict(self, X):
        """ Predict the target values based on the fitted group estimates """
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        # Initialize a list to store the prediction results
        results = []
        # Counter for missing groups
        missing_count = 0
        # Iterate over each row in X and match keys
        for _, row in X.iterrows():
            # Create a tuple key from the row values
            key = row.iloc[0] if X.shape[1] == 1 else tuple(row)
            # Check if the key exists in the group estimates
            if key in self.group_estimates:
                # Append the corresponding estimate to the results
                results.append(self.group_estimates[key])
            # If key not found, append NaN and increment the missing count
            else:
                results.append(np.nan)
                missing_count += 1
        # Print the number of missing groups
        if missing_count > 0:
            print(f"{missing_count} missing groups found during prediction.")
        # Return the results as a NumPy array
        return np.array(results)
