"""Neural network classifiers for binary classification."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import Optional, List


# Module-level constants
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_HIDDEN_SIZE = 100  # Match scikit-learn's MLPClassifier default


class BinaryClassifier(nn.Module):
    """Simple neural network for binary classification.

    This is a minimal implementation matching scikit-learn's MLPClassifier
    with a single hidden layer and ReLU activation.
    """

    def __init__(self, in_dim: int, hidden_size: int = DEFAULT_HIDDEN_SIZE) -> None:
        """Initialize the model.

        Args:
            in_dim: Input dimension
            hidden_size: Size of the hidden layer (default: 100)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, in_dim)

        Returns:
            Logits of shape (batch_size, 1)
        """
        return self.net(x)


class BaseNeuralClassifier(BaseEstimator, ClassifierMixin):
    """Base class for neural network classifiers.

    This class provides common functionality for neural network classifiers
    that are compatible with scikit-learn's API.
    """

    def __init__(
        self,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        learning_rate: float = 0.0001,
        batch_size: int = 32,
        n_epochs: int = 100,
        device: str = DEFAULT_DEVICE,
    ) -> None:
        """Initialize the classifier.

        Args:
            hidden_size: Size of the hidden layer
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            n_epochs: Number of training epochs
            device: Device to use for training ('cuda' or 'cpu')
        """
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = device
        self.model: Optional[torch.nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.training_history: List[float] = []
        self.scaler = StandardScaler()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            x: Feature matrix

        Returns:
            Predicted class labels
        """
        probas = self.predict_proba(x)
        return (probas > 0.5).astype(int)

    def _prepare_data(
        self, x: np.ndarray, y: Optional[np.ndarray] = None
    ) -> DataLoader:
        """Convert numpy arrays to PyTorch DataLoader.

        Args:
            x: Feature matrix
            y: Optional target labels

        Returns:
            DataLoader for the dataset
        """
        # Scale features
        if y is not None:  # Training mode
            x_scaled = self.scaler.fit_transform(x)
        else:  # Inference mode
            x_scaled = self.scaler.transform(x)

        x_tensor = torch.FloatTensor(x_scaled).to(self.device)
        if y is not None:
            y_tensor = torch.FloatTensor(y).to(self.device)
            dataset = TensorDataset(x_tensor, y_tensor)
            # Only shuffle during training
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        else:
            dataset = TensorDataset(x_tensor)
            # Don't shuffle during prediction
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)


class HardSigmoidClassifier(BaseEstimator, ClassifierMixin):
    """Neural network classifier using hard sigmoid activation.

    This classifier uses BCEWithLogitsLoss during training and
    applies hard sigmoid at inference time to convert logits to probabilities.
    Matches sklearn's MLPClassifier architecture with a single hidden layer.
    """

    def __init__(
        self,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        learning_rate: float = 0.001,  # Match sklearn's default
        batch_size: int = 200,  # Match sklearn's default
        n_epochs: int = 200,  # Match sklearn's default
        device: str = DEFAULT_DEVICE,
    ) -> None:
        """Initialize the classifier.

        Args:
            hidden_size: Size of the hidden layer
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            n_epochs: Number of training epochs
            device: Device to use for training ('cuda' or 'cpu')
        """
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = device
        self.model: Optional[torch.nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scaler = StandardScaler()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            x: Feature matrix

        Returns:
            Predicted class labels
        """
        probas = self.predict_proba(x)
        return (probas > 0.5).astype(int)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "HardSigmoidClassifier":
        """Fit the model using BCEWithLogitsLoss.

        Args:
            x: Feature matrix
            y: Target labels

        Returns:
            self
        """
        # Initialize model and optimizer
        self.model = BinaryClassifier(
            in_dim=x.shape[1],
            hidden_size=self.hidden_size,
        ).to(self.device)

        if self.model is None:
            raise RuntimeError("Model initialization failed")

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )
        criterion = nn.BCEWithLogitsLoss()

        # Prepare data
        train_loader = self._prepare_data(x, y)

        # Training loop
        self.model.train()
        for _epoch in range(self.n_epochs):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                if self.optimizer is None or self.model is None:
                    raise RuntimeError("Model or optimizer not initialized")

                self.optimizer.zero_grad()
                logits = self.model(batch_x).squeeze(1)
                loss = criterion(logits, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict class probabilities using hard sigmoid.

        Args:
            x: Feature matrix

        Returns:
            Predicted probabilities
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call fit() first.")

        self.model.eval()
        with torch.no_grad():
            test_loader = self._prepare_data(x)
            probas = []
            for (batch_x,) in test_loader:
                logits = self.model(batch_x).squeeze(1)
                # Use hard sigmoid (clamp) instead of regular sigmoid
                probs = torch.clamp((logits + 1) * 0.5, 0.0, 1.0)
                probas.append(probs.cpu().numpy())

        return np.concatenate(probas).reshape(-1, 1)

    def _prepare_data(
        self, x: np.ndarray, y: Optional[np.ndarray] = None
    ) -> DataLoader:
        """Convert numpy arrays to PyTorch DataLoader.

        Args:
            x: Feature matrix
            y: Optional target labels

        Returns:
            DataLoader for the dataset
        """
        # Scale features
        if y is not None:  # Training mode
            x_scaled = self.scaler.fit_transform(x)
        else:  # Inference mode
            x_scaled = self.scaler.transform(x)

        x_tensor = torch.FloatTensor(x_scaled).to(self.device)
        if y is not None:
            y_tensor = torch.FloatTensor(y).to(self.device)
            dataset = TensorDataset(x_tensor, y_tensor)
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        else:
            dataset = TensorDataset(x_tensor)
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)


class HingeClassifier(BaseEstimator, ClassifierMixin):
    """Neural network classifier using hinge loss.

    This classifier uses PyTorch's HingeEmbeddingLoss during training and
    applies hard sigmoid at inference time to convert logits to probabilities.
    """

    def __init__(
        self,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        learning_rate: float = 0.001,  # Match sklearn's default
        batch_size: int = 200,  # Match sklearn's default
        n_epochs: int = 200,  # Match sklearn's default
        device: str = DEFAULT_DEVICE,
    ) -> None:
        """Initialize the classifier.

        Args:
            hidden_size: Size of the hidden layer
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            n_epochs: Number of training epochs
            device: Device to use for training ('cuda' or 'cpu')
        """
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = device
        self.model: Optional[torch.nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scaler = StandardScaler()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            x: Feature matrix

        Returns:
            Predicted class labels
        """
        probas = self.predict_proba(x)
        return (probas > 0.5).astype(int)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "HingeClassifier":
        """Fit the model using hinge loss.

        Args:
            x: Feature matrix
            y: Target labels

        Returns:
            self
        """
        # Initialize model and optimizer
        self.model = BinaryClassifier(
            in_dim=x.shape[1],
            hidden_size=self.hidden_size,
        ).to(self.device)

        if self.model is None:
            raise RuntimeError("Model initialization failed")

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.HingeEmbeddingLoss()

        # Prepare data
        train_loader = self._prepare_data(x, y)

        # Training loop
        self.model.train()
        for _epoch in range(self.n_epochs):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                if self.optimizer is None or self.model is None:
                    raise RuntimeError("Model or optimizer not initialized")

                self.optimizer.zero_grad()
                logits = self.model(batch_x).squeeze(1)
                # For hinge loss, we want:
                # - Positive examples (y=1) to have logits > 1
                # - Negative examples (y=0) to have logits < -1
                y_signed = batch_y * 2 - 1  # Convert {0,1} to {-1,1}
                loss = criterion(logits, y_signed)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict class probabilities using hard sigmoid of logits.

        For hinge loss:
        - logits > 1 indicate positive class (y=1)
        - logits < -1 indicate negative class (y=0)
        - Values in between are uncertain

        Args:
            x: Feature matrix

        Returns:
            Predicted probabilities
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call fit() first.")

        self.model.eval()
        with torch.no_grad():
            test_loader = self._prepare_data(x)
            probas = []
            for (batch_x,) in test_loader:
                logits = self.model(batch_x).squeeze(1)
                # Convert logits to probabilities:
                # - logits > 1 -> prob ≈ 1
                # - logits < -1 -> prob ≈ 0
                # - in between -> linear interpolation
                probs = torch.clamp((logits + 1) * 0.5, 0.0, 1.0)
                # Invert probabilities since hinge loss is inverted
                probs = 1.0 - probs
                probas.append(probs.cpu().numpy())
        return np.concatenate(probas).reshape(-1, 1)

    def _prepare_data(
        self, x: np.ndarray, y: Optional[np.ndarray] = None
    ) -> DataLoader:
        """Convert numpy arrays to PyTorch DataLoader.

        Args:
            x: Feature matrix
            y: Optional target labels

        Returns:
            DataLoader for the dataset
        """
        # Scale features
        if y is not None:  # Training mode
            x_scaled = self.scaler.fit_transform(x)
        else:  # Inference mode
            x_scaled = self.scaler.transform(x)

        x_tensor = torch.FloatTensor(x_scaled).to(self.device)
        if y is not None:
            y_tensor = torch.FloatTensor(y).to(self.device)
            dataset = TensorDataset(x_tensor, y_tensor)
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        else:
            dataset = TensorDataset(x_tensor)
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
