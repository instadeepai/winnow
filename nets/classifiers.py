"""Neural network classifiers for binary classification."""

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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
    that are compatible with scikit-learn's API. The classifiers predict the
    probability that a given prediction is correct (1) or incorrect (0).
    """

    def __init__(
        self,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        learning_rate: float = 0.0001,
        batch_size: int = 32,
        n_epochs: int = 100,
        device: str = DEFAULT_DEVICE,
        random_state: Optional[int] = None,  # Random state for reproducibility
    ) -> None:
        """Initialize the classifier.

        Args:
            hidden_size: Size of the hidden layer
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            n_epochs: Number of training epochs
            device: Device to use for training ('cuda' or 'cpu')
            random_state: Random state for reproducibility
            calibrate_probs: Whether to calibrate probabilities
        """
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = device
        self.random_state = random_state
        self.model: Optional[torch.nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.training_history: List[float] = []
        self.scaler = StandardScaler()
        self.calibrator: Optional[LogisticRegression] = None
        self.calibrate_probs: bool = True

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


class SteepSigmoidClassifier(BaseNeuralClassifier):
    """Neural network classifier using a custom steep sigmoid activation."""

    def __init__(
        self,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        learning_rate: float = 0.001,
        batch_size: int = 200,
        n_epochs: int = 200,
        device: str = DEFAULT_DEVICE,
        alpha: float = 1.0,
        random_state: Optional[int] = None,  # Random state for reproducibility
        calibrate_probs: bool = True,  # Whether to calibrate probabilities
    ) -> None:
        """Initialize the classifier.

        Args:
            hidden_size: Size of the hidden layer
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            n_epochs: Number of training epochs
            device: Device to use for training ('cuda' or 'cpu')
            alpha: Steepness parameter for sigmoid
            random_state: Random state for reproducibility
            calibrate_probs: Whether to calibrate probabilities
        """
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = device
        self.random_state = random_state
        self.model: Optional[torch.nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scaler = StandardScaler()
        self.calibrator: Optional[LogisticRegression] = None
        self.alpha: float = alpha
        self.calibrate_probs: bool = calibrate_probs

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "SteepSigmoidClassifier":
        """Fit the model using BCEWithLogitsLoss and calibrate probabilities.

        Args:
            x_train: Training features
            y_train: Training labels
            x_val: Validation features for calibration
            y_val: Validation labels for calibration

        Returns:
            self
        """
        # Initialize model and optimizer
        self.model = BinaryClassifier(
            in_dim=x_train.shape[1],
            hidden_size=self.hidden_size,
        ).to(self.device)

        if self.model is None:
            raise RuntimeError("Model initialization failed")

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )
        criterion = nn.BCEWithLogitsLoss()

        # Prepare training data
        train_loader = self._prepare_data(x_train, y_train)

        # Training loop
        self.model.train()
        for _epoch in range(self.n_epochs):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                if self.optimizer is None or self.model is None:
                    raise RuntimeError("Model or optimizer not initialized")

                self.optimizer.zero_grad()
                logits = self.model(batch_x).squeeze(1)
                loss = criterion(self.alpha * logits, batch_y) - 0.01 * torch.mean(
                    logits**2
                )
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

        # Calibrate probabilities if desired
        if self.calibrate_probs:
            self.model.eval()
            with torch.no_grad():
                val_loader = self._prepare_data(x_val)
                raw_probs = []
                for (batch_x,) in val_loader:
                    logits = self.model(batch_x).squeeze(1)
                    probs = self._steep_sigmoid(logits)
                    raw_probs.append(probs.cpu().numpy())
            raw_probs = np.concatenate(raw_probs).reshape(-1, 1)

            # Fit calibrator on validation set
            self.calibrator = LogisticRegression()
            self.calibrator.fit(raw_probs, y_val)

        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict class probabilities using hard sigmoid and calibration.

        Args:
            x: Feature matrix

        Returns:
            Predicted probabilities
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call fit() first.")

        # Get raw probabilities
        self.model.eval()
        with torch.no_grad():
            test_loader = self._prepare_data(x)
            raw_probs = []
            for (batch_x,) in test_loader:
                logits = self.model(batch_x).squeeze(1)
                probs = self._steep_sigmoid(logits)
                raw_probs.append(probs.cpu().numpy())

        if self.calibrate_probs and self.calibrator is not None:
            return self.calibrator.predict_proba(
                np.concatenate(raw_probs).reshape(-1, 1)
            )
        else:
            return np.concatenate(raw_probs).reshape(-1, 1)

    def _steep_sigmoid(self, z: torch.Tensor) -> torch.Tensor:
        """Steep sigmoid activation.

        Args:
            z: Input tensor

        Returns:
            Steep sigmoid output in (0,1)
        """
        return nn.Sigmoid()(self.alpha * z)

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


class HingeClassifier(BaseNeuralClassifier):
    """Neural network classifier using hinge loss.

    This classifier uses PyTorch's HingeEmbeddingLoss during training and
    applies a softened hard sigmoid at inference time to convert logits to probabilities.
    The output probabilities represent the probability that a given prediction
    is correct (1) or incorrect (0). For hinge loss, positive logits correspond
    to correct predictions and negative logits to incorrect predictions.
    """

    def __init__(
        self,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        learning_rate: float = 0.001,
        batch_size: int = 200,
        n_epochs: int = 200,
        device: str = DEFAULT_DEVICE,
        random_state: Optional[int] = None,  # Random state for reproducibility
        alpha: float = 1.0,
        calibrate_probs: bool = True,
    ) -> None:
        """Initialize the classifier.

        Args:
            hidden_size: Size of the hidden layer
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            n_epochs: Number of training epochs
            device: Device to use for training ('cuda' or 'cpu')
            random_state: Random state for reproducibility
            alpha: Stretch factor for hard sigmoid
            calibrate_probs: Whether to calibrate probabilities
        """
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = device
        self.random_state = random_state
        self.model: Optional[torch.nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scaler = StandardScaler()
        self.calibrator: Optional[LogisticRegression] = None
        self.calibrate_probs: bool = True
        self.alpha: float = alpha

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "HingeClassifier":
        """Fit the model using hinge loss and calibrate probabilities.

        Args:
            x_train: Training features
            y_train: Training labels (0 for negative class, 1 for positive class)
            x_val: Validation features for calibration
            y_val: Validation labels for calibration

        Returns:
            self
        """
        # Initialize model and optimizer
        self.model = BinaryClassifier(
            in_dim=x_train.shape[1],
            hidden_size=self.hidden_size,
        ).to(self.device)

        if self.model is None:
            raise RuntimeError("Model initialization failed")

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.HingeEmbeddingLoss()

        # Prepare training data
        train_loader = self._prepare_data(x_train, y_train)

        # Training loop
        self.model.train()
        for _epoch in range(self.n_epochs):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                if self.optimizer is None or self.model is None:
                    raise RuntimeError("Model or optimizer not initialized")

                self.optimizer.zero_grad()
                logits = self.model(batch_x).squeeze(1)
                probs = self._hard_sigmoid(logits)
                # Convert {0,1} to {-1,1} for hinge loss
                y_signed = batch_y * 2 - 1
                loss = criterion(probs, y_signed)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

        # Calibrate probabilities if desired
        if self.calibrate_probs:
            self.model.eval()
            with torch.no_grad():
                val_loader = self._prepare_data(x_val)
                raw_probs = []
                for (batch_x,) in val_loader:
                    logits = self.model(batch_x).squeeze(1)
                    probs = self._hard_sigmoid(logits)
                    raw_probs.append(probs.cpu().numpy())
            raw_probs = np.concatenate(raw_probs).reshape(-1, 1)

            # Fit calibrator on validation set
            self.calibrator = LogisticRegression()
            self.calibrator.fit(raw_probs, y_val)

        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict class probabilities using softened hard sigmoid and calibration.

        Args:
            x: Feature matrix

        Returns:
            Predicted probabilities where higher values indicate class 1
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call fit() first.")

        # Get raw probabilities
        self.model.eval()
        with torch.no_grad():
            test_loader = self._prepare_data(x)
            raw_probs = []
            for (batch_x,) in test_loader:
                logits = self.model(batch_x).squeeze(1)
                probs = self._hard_sigmoid(logits)
                raw_probs.append(probs.cpu().numpy())

        if self.calibrate_probs and self.calibrator is not None:
            return self.calibrator.predict_proba(
                np.concatenate(raw_probs).reshape(-1, 1)
            )
        else:
            return np.concatenate(raw_probs).reshape(-1, 1)

    def _hard_sigmoid(self, z: torch.Tensor) -> torch.Tensor:
        """Hard sigmoid activation, stretched by alpha.

        Args:
            z: Input tensor

        Returns:
            Hard sigmoid output in (0,1)
        """
        # Apply hard sigmoid to the scaled input
        return f.hardsigmoid(self.alpha * z)

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
