import numpy as np
from scipy.stats import multivariate_normal

class BayesianClassifier:
    def __init__(self, n_classes=5):
        self.n_classes = n_classes
        self.class_means = {}
        self.class_covs = {}
        self.class_priors = {}

        # Initialize with some default parameters for each age group
        # These are simplified parameters for demonstration
        n_features = 1000  # Expected feature vector size

        for i in range(n_classes):
            # Initialize with reasonable defaults for each age group
            self.class_means[i] = np.random.normal(0, 1, n_features)
            self.class_covs[i] = np.eye(n_features) * (i + 1)  # Different variance for each class
            self.class_priors[i] = 1.0 / n_classes  # Equal priors

    def fit(self, X, y):
        """
        Train the Bayesian classifier

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
        """
        unique_classes = np.unique(y)
        n_samples = len(y)

        # Calculate class statistics and priors
        for class_label in unique_classes:
            class_samples = X[y == class_label]

            if len(class_samples) > 0:
                # Calculate mean and covariance
                self.class_means[class_label] = np.mean(class_samples, axis=0)
                # Add small diagonal term for numerical stability
                cov = np.cov(class_samples.T) + np.eye(class_samples.shape[1]) * 1e-6
                self.class_covs[class_label] = cov

                # Calculate class prior probability
                self.class_priors[class_label] = len(class_samples) / n_samples

    def predict(self, X):
        """
        Predict age group using Bayes' theorem

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            predictions: Predicted age groups
        """
        predictions = []

        for x in X:
            # Calculate posterior probability for each class
            posteriors = {}

            for class_label in self.class_means.keys():
                try:
                    # Calculate likelihood using multivariate normal distribution
                    likelihood = multivariate_normal.pdf(
                        x,
                        mean=self.class_means[class_label],
                        cov=self.class_covs[class_label],
                        allow_singular=True
                    )

                    # Calculate posterior using Bayes' theorem
                    posterior = likelihood * self.class_priors[class_label]
                    posteriors[class_label] = posterior
                except ValueError:
                    # Handle potential numerical issues
                    posteriors[class_label] = 0.0

            # Select class with highest posterior probability
            if posteriors:
                predicted_class = max(posteriors.items(), key=lambda x: x[1])[0]
            else:
                predicted_class = 0  # Default to youngest class if calculation fails

            predictions.append(predicted_class)

        return np.array(predictions)

    def get_confidence(self, X):
        """
        Calculate confidence scores for predictions

        Args:
            X: Feature matrix

        Returns:
            confidences: Confidence scores
        """
        confidences = []

        for x in X:
            posteriors = {}

            for class_label in self.class_means.keys():
                try:
                    likelihood = multivariate_normal.pdf(
                        x,
                        mean=self.class_means[class_label],
                        cov=self.class_covs[class_label],
                        allow_singular=True
                    )
                    posterior = likelihood * self.class_priors[class_label]
                    posteriors[class_label] = posterior
                except ValueError:
                    posteriors[class_label] = 0.0

            # Calculate confidence as normalized posterior
            total = sum(posteriors.values())
            max_posterior = max(posteriors.values())
            confidence = max_posterior / total if total > 0 else 0
            confidences.append(confidence)

        return np.array(confidences)
