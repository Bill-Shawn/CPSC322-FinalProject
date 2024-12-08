import numpy as np

class LocalRandomForestClassifier:
    def __init__(self, n_estimators=10, max_features="sqrt", random_state=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []

    def _entropy(self, y):
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    def _information_gain(self, X_column, y, split_value):
        left_indices = X_column < split_value
        right_indices = ~left_indices
        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
            return 0
        parent_entropy = self._entropy(y)
        n = len(y)
        child_entropy = (len(y[left_indices]) / n) * self._entropy(y[left_indices]) + \
                        (len(y[right_indices]) / n) * self._entropy(y[right_indices])
        return parent_entropy - child_entropy

    def _best_split(self, X, y):
        best_gain = -1
        split_value = None
        split_column = None

        for col_index in range(X.shape[1]):
            X_column = X[:, col_index]
            for value in np.unique(X_column):
                left_indices = X_column < value
                right_indices = ~left_indices

                # Skip invalid splits
                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue

                gain = self._information_gain(X_column, y, value)
                if gain > best_gain:
                    best_gain = gain
                    split_value = value
                    split_column = col_index

        return split_column, split_value


    def _build_tree(self, X, y, depth=0):
        if y is None or len(y) == 0:  # Handle empty labels
            return {"label": None}

        # If all labels are the same or no features are left
        if len(set(y)) == 1 or X.shape[0] == 0:
            return {"label": np.bincount(y).argmax() if len(y) > 0 else None}

        # Find the best split
        column, value = self._best_split(X, y)

        # If no valid split is found
        if column is None:
            return {"label": np.bincount(y).argmax() if len(y) > 0 else None}

        left_indices = X[:, column] < value
        right_indices = ~left_indices

        # If a split is invalid, return majority label
        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            return {"label": np.bincount(y).argmax() if len(y) > 0 else None}

        return {
            "column": column,
            "value": value,
            "left": self._build_tree(X[left_indices], y[left_indices], depth + 1),
            "right": self._build_tree(X[right_indices], y[right_indices], depth + 1),
        }



def fit(self, X, y):
    np.random.seed(self.random_state)
    n_samples = X.shape[0]
    for _ in range(self.n_estimators):
        indices = np.random.choice(range(n_samples), size=n_samples, replace=True)
        X_sample, y_sample = X[indices], y[indices]
        if len(np.unique(y_sample)) > 1:  # Avoid degenerate trees
            tree = self._build_tree(X_sample, y_sample)
            self.trees.append(tree)


    def _predict_tree(self, tree, row):
        if "label" in tree:
            return tree["label"]
        column = tree["column"]
        value = tree["value"]
        if row[column] < value:
            return self._predict_tree(tree["left"], row)
        else:
            return self._predict_tree(tree["right"], row)

    def predict(self, X):
        tree_predictions = np.array([[self._predict_tree(tree, row) for row in X] for tree in self.trees])
        return np.array([np.bincount(tree_predictions[:, i]).argmax() for i in range(tree_predictions.shape[1])])



class LocalDecisionTreeClassifier(LocalRandomForestClassifier):
    def __init__(self, max_depth=None, random_state=None):
        super().__init__(n_estimators=1, random_state=random_state)
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._predict_tree(self.tree, row) for row in X])


class LocalKNNClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _distance(self, row1, row2):
        return np.sqrt(np.sum((row1 - row2) ** 2))

    def _get_neighbors(self, row):
        distances = [(self._distance(row, train_row), label) for train_row, label in zip(self.X_train, self.y_train)]
        distances.sort(key=lambda x: x[0])
        return [label for _, label in distances[:self.n_neighbors]]

    def predict(self, X):
        predictions = []
        for row in X:
            neighbors = self._get_neighbors(row)
            predictions.append(np.bincount(neighbors).argmax())
        return np.array(predictions)


class LocalNaiveBayesClassifier:
    def __init__(self):
        self.classes = None
        self.means = {}
        self.variances = {}
        self.priors = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis=0)
            self.variances[c] = np.var(X_c, axis=0)
            self.priors[c] = len(X_c) / len(X)

    def _calculate_probability(self, x, mean, var):
        eps = 1e-6
        coeff = 1.0 / np.sqrt(2.0 * np.pi * var + eps)
        exponent = np.exp(-(x - mean) ** 2 / (2.0 * var + eps))
        return coeff * exponent

    def _predict_single(self, x):
        posteriors = []
        for c in self.classes:
            prior = np.log(self.priors[c])
            likelihood = np.sum(np.log(self._calculate_probability(x, self.means[c], self.variances[c])))
            posteriors.append(prior + likelihood)
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])


if __name__ == "__main__":
    # Example usage

    # Random Forest
    rf = LocalRandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)

    # Decision Tree
    dt = LocalDecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    dt_preds = dt.predict(X_test)

    # k-NN
    knn = LocalKNNClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_preds = knn.predict(X_test)

    # Naive Bayes
    nb = LocalNaiveBayesClassifier()
    nb.fit(X_train, y_train)
    nb_preds = nb.predict(X_test)

    # Evaluate Accuracy
    print("Random Forest Accuracy:", np.mean(rf_preds == y_test))
    print("Decision Tree Accuracy:", np.mean(dt_preds == y_test))
    print("k-NN Accuracy:", np.mean(knn_preds == y_test))
    print("Naive Bayes Accuracy:", np.mean(nb_preds == y_test))
