import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score, roc_curve, mean_absolute_error, mean_squared_error, log_loss, r2_score, brier_score_loss
from sklearn.calibration import calibration_curve

class DeepNeuralNetwork:
    def __init__(
        self,
        layer_dims: List[int],
        activations: List[str] = [],
        lr: float = 0.001,
        epochs: int = 100,
        batch_size: int = 64,
        loss_function: str = "cross_entropy",
        optimizer_type: str = "adam",
        dropout_rates: List[float] = [],
        weight_decay: float = 0.0001,
        early_stopping: bool = True,
        patience: int = 10,
        print_every: int = -1,
        plot_loss: bool = False,
        seed: int = 42
    ):
        """
        Initialize the Deep Neural Network

        Parameters:
        layer_dims : List[int]
            List containing the number of units in each layer
        activations : Optional[List[str]]
            List of activation functions for each layer
        lr : float
            Learning rate for optimization
        epochs : int
            Number of training epochs
        batch_size : int
            Size of each training batch
        loss_function : str
            Loss function to use ("cross_entropy" or "mse")
        optimizer_type : str
            Optimizer to use ("sgd", "momentum", or "adam")
        dropout_rates : Optional[List[float]]
            Dropout rates for each layer
        weight_decay : float
            L2 regularization coefficient
        early_stopping : bool
            Whether to use early stopping (default: True)
        patience : int
            Patience for early stopping 
        print_every : int
            Frequency of printing training progress (-1 means no printing)
        plot_loss : bool
            Whether to plot loss curves after training (default: False)
        seed : int
            Random seed for reproducibility (default: 42)

        Returns:
        """
        np.random.seed(seed)
        self.layer_dims = layer_dims
        self.activations = ["relu"] * (len(layer_dims) - 2) if activations==[] else activations
        self.activations.append("softmax")
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.optimizer_type = optimizer_type
        self.dropout_rates = [0.1] * (len(layer_dims) - 2) if dropout_rates==[] else dropout_rates
        self.weight_decay = weight_decay
        self.early_stopping = early_stopping
        self.patience = patience
        self.print_every = print_every
        self.plot_loss = plot_loss

        self.weights, self.biases = self._initialize_parameters()
        self.velocity, (self.m, self.v) = None, (None, None)

    # Activation Functions
    def _sigmoid(
        self, 
        x: np.ndarray, 
        derivative: bool = False
    ) -> np.ndarray:
        """
        Sigmoid activation function and its derivative
        
        Parameters:
        x : np.ndarray
            Input array
        derivative : bool
            Whether to compute the derivative
            
        Returns:
        np.ndarray
            Activated output or its derivative
        """
        x = np.clip(x, -500, 500)
        s = 1 / (1 + np.exp(-x))
        return s * (1 - s) if derivative else s

    def _tanh(
        self, 
        x: np.ndarray, 
        derivative: bool = False
    ) -> np.ndarray:
        """
        Tanh activation function and its derivative

        Parameters:
        x : np.ndarray
            Input array
        derivative : bool
            Whether to compute the derivative

        Returns:
        np.ndarray
            Activated output or its derivative
        """
        t = np.tanh(x)
        return 1 - t**2 if derivative else t

    def _relu(
        self, 
        x: np.ndarray, 
        derivative: bool = False
    ) -> np.ndarray:
        """
        ReLU activation function and its derivative

        Parameters:
        x : np.ndarray
            Input array
        derivative : bool
            Whether to compute the derivative

        Returns:
        np.ndarray
            Activated output or its derivative
        """
        return (x > 0).astype(float) if derivative else np.maximum(0, x)

    def _leaky_relu(
        self,
        x: np.ndarray, 
        derivative: bool = False,
        alpha: float = 0.01
    ) -> np.ndarray:
        """
        Leaky ReLU activation function and its derivative

        Parameters:
        x : np.ndarray
            Input array
        derivative : bool
            Whether to compute the derivative
        alpha : float
            Slope for negative inputs

        Returns:
        np.ndarray
            Activated output or its derivative
        """
        if derivative:
            grad = np.ones_like(x)
            grad[x < 0] = alpha
            return grad
        return np.where(x > 0, x, alpha * x)

    def _softmax(
        self, 
        x: np.ndarray
    ) -> np.ndarray:
        """
        Softmax activation function

        Parameters:
        x : np.ndarray
            Input array

        Returns:
        np.ndarray
            Activated output
        """
        exp_shifted = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_shifted / np.sum(exp_shifted, axis=0, keepdims=True)

    # Loss Functions
    def _mse_loss(
        self, 
        Y : np.ndarray, 
        Y_pred : np.ndarray
    ) -> float:
        """
        Mean Squared Error loss function

        Parameters:
        Y : np.ndarray
            True labels
        Y_pred : np.ndarray
            Predicted labels

        Returns:
        float
            Computed MSE loss
        """
        diff = Y - Y_pred.T
        return 0.5 * np.mean(diff * diff)

    def _cross_entropy_loss(
        self, 
        Y: np.ndarray, 
        Y_pred: np.ndarray,
        eps : float = 1e-9
    ) -> float:
        """
        Cross-Entropy loss function

        Parameters:
        Y : np.ndarray
            True labels (label encoded)
        Y_pred : np.ndarray
            Predicted probabilities
        eps : float
            Small value to avoid log(0)

        Returns:
        float
            Computed Cross-Entropy loss
        """
        Y_pred = np.clip(Y_pred.T, eps, 1 - eps)
        return -np.mean(np.sum(Y * np.log(Y_pred), axis=1))

    # Initialization
    def _initialize_parameters(
        self
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Initialize weights and biases using He or Xavier initialization

        Parameters:
        
        Returns:
        weights : List[np.ndarray]
            List of weight matrices for each layer
        biases : List[np.ndarray]
            List of bias vectors for each layer
        """
        standard_normal = np.random.standard_normal
        weights, biases = [], []
        for i in range(1, len(self.layer_dims)):
            fan_in = self.layer_dims[i - 1]
            fan_out = self.layer_dims[i]
            act = self.activations[i - 1]
            scale = np.sqrt(2.0 / fan_in) if act in ("relu", "leaky_relu") else np.sqrt(1.0 / fan_in)
            W = standard_normal((fan_out, fan_in)).astype(np.float32) * scale
            b = np.zeros((fan_out, 1), dtype=np.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    # Forward Propagation
    def _forward(
        self, 
        X, 
        training=True
    ) -> Tuple[np.ndarray, Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]]:
        """
        Perform forward propagation through the network
        
        Parameters:
        X : np.ndarray
            Input data of shape (num_samples, num_features)
        training : bool
            Flag indicating whether in training mode (to apply dropout)
        
        Returns:
        A_out : np.ndarray
            Output of the network after forward propagation
        cache_data : Tuple
            Tuple containing caches needed for backpropagation
        """
        A = X.T
        cache = [A]  # store activations for backprop
        Zs = []      # store pre-activations (Z values)
        masks = []   # store dropout masks
        act_funcs = {
            "sigmoid": self._sigmoid,
            "relu": self._relu,
            "tanh": self._tanh,
            "leaky_relu": self._leaky_relu,
            "softmax": self._softmax,
        }

        for i, (W, b, act_name) in enumerate(zip(self.weights, self.biases, self.activations)):
            Z = W @ A + b
            Zs.append(Z)
            A = act_funcs[act_name](Z)
            if training and self.dropout_rates and i < len(self.dropout_rates):
                rate = self.dropout_rates[i]
                mask = (np.random.rand(*A.shape) > rate).astype(float)
                A *= mask
                A /= (1 - rate)
                masks.append(mask)
            else:
                masks.append(None)
            cache.append(A)
        return A, (cache, Zs, masks)

    # Backward Propagation
    def _backward(
        self, 
        Y: np.ndarray, 
        cache_data: Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Perform backward propagation through the network

        Parameters:
        Y : np.ndarray
            True labels (one-hot encoded)
        cache_data : Tuple
            Tuple containing caches from forward propagation

        Returns:
        dWs : List[np.ndarray]
            Gradients of weights for each layer
        dbs : List[np.ndarray]
            Gradients of biases for each layer
        """
        cache, Zs, masks = cache_data
        m = Y.shape[0]
        L = len(self.weights)
        dWs, dbs = [None] * L, [None] * L
        Y = Y.T
        dZ = cache[-1] - Y  # for softmax-crossentropy

        act_derivs = {
            "sigmoid": lambda Z: self._sigmoid(Z, True),
            "relu": lambda Z: self._relu(Z, True),
            "tanh": lambda Z: self._tanh(Z, True),
            "leaky_relu": lambda Z: self._leaky_relu(Z, True),
        }

        for i in reversed(range(L)):
            A_prev = cache[i]
            W = self.weights[i]
            dWs[i] = (dZ @ A_prev.T) / m + self.weight_decay * W
            dbs[i] = np.sum(dZ, axis=1, keepdims=True) / m
            if i > 0:
                dA_prev = W.T @ dZ
                if masks[i - 1] is not None:
                    dA_prev *= masks[i - 1]
                    dA_prev /= (1 - self.dropout_rates[i - 1])
                dZ = dA_prev * act_derivs[self.activations[i - 1]](Zs[i - 1])
        return dWs, dbs

    # Parameter Update through Optimizers
    def _update_parameters(
        self,
        dWs: List[np.ndarray],
        dbs: List[np.ndarray],
        epoch: int
    ):
        """
        Update network parameters using the selected optimizer

        Parameters:
        dWs : List[np.ndarray]
            Gradients of weights for each layer
        dbs : List[np.ndarray]
            Gradients of biases for each layer
        epoch : int
            Current epoch number (for Adam bias correction)

        Returns:
        """
        L = len(self.weights)
        
        if self.optimizer_type == "sgd":
            """ Stochatic Gradient Descent (SGD):
                θ^(t+1) <- θ^t - η∇L(y, ŷ)
            """
            for i in range(L):
                self.weights[i] -= self.lr * dWs[i]
                self.biases[i] -= self.lr * dbs[i]
            
        elif self.optimizer_type == "momentum":
            """ Momentum:
                v^(t+1) <- βv^t + (1-β)∇L(y, ŷ)^t
                θ^(t+1) <- θ^t - ηv^(t+1)
            """
            if self.velocity is None:
                self.velocity = {"W": [np.zeros_like(W) for W in self.weights], "b": [np.zeros_like(b) for b in self.biases]}
            beta = 0.9
            for i in range(L):
                self.velocity["W"][i] = beta * self.velocity["W"][i] + (1 - beta) * dWs[i]
                self.velocity["b"][i] = beta * self.velocity["b"][i] + (1 - beta) * dbs[i]
                self.weights[i] -= self.lr * self.velocity["W"][i]
                self.biases[i] -= self.lr * self.velocity["b"][i]
        
        elif self.optimizer_type == "adam":
            """ Adam:
                m^(t+1) <- β1 m^t + (1-β1)∇L(y, ŷ)^t
                v^(t+1) <- β2 v^t + (1-β2)(∇L(y, ŷ)^t)^2
                m̂^(t+1) <- m^(t+1) / (1 - β1^(t+1))
                v̂^(t+1) <- v^(t+1) / (1 - β2^(t+1))
                θ^(t+1) <- θ^t - η m̂^(t+1) / (√(v̂^(t+1)) + ε)
            """
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            if self.m is None:
                self.m = {"W": [np.zeros_like(W) for W in self.weights], "b": [np.zeros_like(b) for b in self.biases]}
                self.v = {"W": [np.zeros_like(W) for W in self.weights], "b": [np.zeros_like(b) for b in self.biases]}
            for i in range(L):
                self.m["W"][i] = beta1 * self.m["W"][i] + (1 - beta1) * dWs[i]
                self.m["b"][i] = beta1 * self.m["b"][i] + (1 - beta1) * dbs[i]
                self.v["W"][i] = beta2 * self.v["W"][i] + (1 - beta2) * (dWs[i] ** 2)
                self.v["b"][i] = beta2 * self.v["b"][i] + (1 - beta2) * (dbs[i] ** 2)

                m_hat_W = self.m["W"][i] / (1 - beta1 ** (epoch + 1))
                m_hat_b = self.m["b"][i] / (1 - beta1 ** (epoch + 1))
                v_hat_W = self.v["W"][i] / (1 - beta2 ** (epoch + 1))
                v_hat_b = self.v["b"][i] / (1 - beta2 ** (epoch + 1))

                self.weights[i] -= self.lr * m_hat_W / (np.sqrt(v_hat_W) + eps)
                self.biases[i] -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + eps)

    # Training Loop
    def fit(
        self, 
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray
    ):
        """
        Train the Deep Neural Network

        Parameters:
        X_train : np.ndarray
            Training input data
        Y_train : np.ndarray
            Training labels
        X_val : np.ndarray
            Validation input data
        Y_val : np.ndarray
            Validation labels

        Returns:
        """
        best_loss, patience_counter = np.inf, 0
        train_losses, val_losses = [], []
        self.velocity, (self.m, self.v) = None, (None, None)

        def batch_iterator(X, Y, batch_size):
            for i in range(0, X.shape[0], batch_size):
                yield X[i:i + batch_size], Y[i:i + batch_size]

        for epoch in range(self.epochs):
            perm = np.random.permutation(X_train.shape[0]) # Shuffle training data
            X_train, Y_train = X_train[perm], Y_train[perm]
            epoch_loss, num_batches = 0.0, 0

            for X_batch, Y_batch in batch_iterator(X_train, Y_train, self.batch_size): # Training by batches
                A_out, cache_data = self._forward(X_batch, training=True)
                loss = self._cross_entropy_loss(Y_batch, A_out) if self.loss_function == "cross_entropy" else self._mse_loss(Y_batch, A_out)
                dWs, dbs = self._backward(Y_batch, cache_data)
                self._update_parameters(dWs, dbs, epoch)
                epoch_loss += loss
                num_batches += 1

            # Validation
            A_val, _ = self._forward(X_val, training=False)
            val_loss = self._cross_entropy_loss(Y_val, A_val) if self.loss_function == "cross_entropy" else self._mse_loss(Y_val, A_val)
            avg_loss = epoch_loss / num_batches
            train_losses.append(avg_loss)
            val_losses.append(val_loss)

            if self.print_every > 0 and epoch % self.print_every == 0:
                print(f"Epoch {epoch}/{self.epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Early stopping
            if self.early_stopping:
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_weights = [W.copy() for W in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if self.print_every > 0:
                            print("Early stopping triggered.")
                        self.weights, self.biases = best_weights, best_biases
                        break

        if self.plot_loss:
            plt.plot(train_losses, label="Train")
            plt.plot(val_losses, label="Val")
            plt.legend()
            plt.title("Loss over Epochs")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.show()

        return self.weights, self.biases

    # Prediction
    def predict(
        self, 
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ 
        Make predictions with the trained network

        Parameters:
        X : np.ndarray
            Input data for prediction

        Returns:
        Tuple[np.ndarray, np.ndarray]
            Predicted class labels and probabilities
        """
        A_out, _ = self._forward(X, training=False)
        return np.argmax(A_out, axis=0), A_out.T

    # Metrics
    def accuracy(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Calculate accuracy of predictions

        Parameters:
        y_true : np.ndarray
            True class labels
        y_pred : np.ndarray
            Predicted class labels

        Returns:
        float
            Accuracy percentage
        """
        return np.mean(y_true == y_pred)
    
    def classification_report(
        self,
        y_true : np.ndarray,
        y_pred : np.ndarray
    ) -> str:
        """
        Generate classification report
        
        Parameters:
        y_true : np.ndarray
            True class labels
        y_pred : np.ndarray
            Predicted class labels  

        Returns:
        str
            Text summary of the precision, recall, F1-score for each class
        """
        return classification_report(y_true, y_pred)
    
    def confusion_matrix(
        self,
        y_true : np.ndarray,
        y_pred : np.ndarray
    ) -> np.ndarray:
        """
        Generate confusion matrix

        Parameters:
        y_true : np.ndarray
            True class labels
        y_pred : np.ndarray
            Predicted class labels

        Returns:
        np.ndarray
            Confusion matrix
        """
        return confusion_matrix(y_true, y_pred)
    
    def roc_auc_score(
        self, 
        y_true : np.ndarray,
        y_scores : np.ndarray
    ) -> float:
        """
        Calculate ROC AUC score
        
        Parameters:
        y_true : np.ndarray
            True class labels
        y_scores : np.ndarray
            Predicted probabilities for positive class
            
        Returns:
        float
            ROC AUC score
        """
        return roc_auc_score(y_true, y_scores)
    
    def f1_score(
        self,
        y_true : np.ndarray,
        y_pred : np.ndarray
    ) -> float:
        """
        Calculate F1 score

        Parameters:
        y_true : np.ndarray
            True class labels
        y_pred : np.ndarray
            Predicted class labels

        Returns:
        float
            F1 score
        """
        return f1_score(y_true, y_pred, average='weighted')
    
    def precision(
        self,
        y_true : np.ndarray,
        y_pred : np.ndarray
    ) -> float:
        """
        Calculate precision score
        
        Parameters:
        y_true : np.ndarray
            True class labels
        y_pred : np.ndarray
            Predicted class labels
        
        Returns:
        float
            Precision score
        """
        return precision_score(y_true, y_pred, average='weighted')
    
    def recall(
        self,
        y_true : np.ndarray,
        y_pred : np.ndarray
    ) -> float:
        """
        Calculate recall score  

        Parameters:
        y_true : np.ndarray
            True class labels
        y_pred : np.ndarray
            Predicted class labels

        Returns:
        float
            Recall score
        """
        return recall_score(y_true, y_pred, average='weighted')
    
    def roc_curve(
        self,
        y_true : np.ndarray,
        y_scores : np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate ROC curve

        Parameters:
        y_true : np.ndarray
            True class labels
        y_scores : np.ndarray
            Predicted probabilities for positive class

        Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            False positive rates, true positive rates, thresholds
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        return fpr, tpr, thresholds
    
    def mean_absolute_error(
        self,
        y_true : np.ndarray,
        y_pred : np.ndarray
    ) -> float:
        """
        Calculate Mean Absolute Error (MAE)

        Parameters:
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels

        Returns:
        float
            Mean Absolute Error
        """
        return mean_absolute_error(y_true, y_pred)
    
    def mean_squared_error(
        self,
        y_true : np.ndarray,
        y_pred : np.ndarray
    ) -> float:
        """
        Calculate Mean Squared Error (MSE)

        Parameters:
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels

        Returns:
        float
            Mean Squared Error
        """
        return mean_squared_error(y_true, y_pred)
    
    def r2_score(
        self,
        y_true : np.ndarray,
        y_pred : np.ndarray
    ) -> float:
        """
        Calculate R2 Score  

        Parameters:
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels

        Returns:
        float
            R2 Score
        """
        return r2_score(y_true, y_pred)
    
    def brier_score_loss(
        self,
        y_true : np.ndarray,
        y_pred : np.ndarray
    ) -> float:
        """
        Calculate Brier Score Loss

        Parameters:
        y_true : np.ndarray
            True class labels
        y_pred : np.ndarray
            Predicted probabilities for positive class

        Returns:
        float
            Brier Score Loss
        """
        return brier_score_loss(y_true, y_pred)
    
    def feature_importances(
        self
    ) -> List[np.ndarray]:
        """
        Calculate feature importances based on absolute weights

        Parameters:

        Returns:
        List[np.ndarray]
            List of feature importances for each layer
        """
        importances = [np.sum(np.abs(W), axis=0) for W in self.weights]
        return importances
    
    def calibration_curve(
        self, 
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate calibration curve

        Parameters:
        y_true : np.ndarray
            True class labels
        y_prob : np.ndarray
            Predicted probabilities for positive class
        n_bins : int
            Number of bins for calibration curve    

        Returns:        
        Tuple[np.ndarray, np.ndarray]
            True probabilities, predicted probabilities
        """
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        return prob_true, prob_pred

