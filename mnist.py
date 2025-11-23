# =======================================
# ML Pipeline 
# Dataset: MNIST (Handwritten Digits for multi-class classification)
# =======================================


# =======================================
# 0. Import Libraries
# =======================================
import numpy as np
import argparse
import os
os.makedirs("model_configs/mnist", exist_ok=True)
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from deepneuralnetwork import DeepNeuralNetwork
from functools import partial
import optuna
import yaml


# =======================================
# 1. Argument Parsing
# =======================================
parser = argparse.ArgumentParser(description='MNIST (Handwritten Digits) Dataset Classification with DNN from Scratch')
# parameters needed for both optuna and non-optuna runs
parser.add_argument('-o', '--optuna', action='store_true', help='Enable hyperparameter optimization with Optuna')
parser.add_argument('-c', '--config', type=str, help='Path to YAML config file specifying hyperparameters') # can be done via config file
# otherwise they can be hyperparameters set by the user 

# epochs is needed for both optuna and non-optuna runs and can be set by user
parser.add_argument('-e', '--epochs', type=int, default=40)
# only if --optuna is set 
parser.add_argument('-t', '--trials', type=int, default=20, help='Number of Optuna trials for hyperparameter optimization')

# User-defined hyperparameters (used if Optuna is NOT enabled), can be one by one specifically default values otherwise
parser.add_argument('-l', '--layer_dims', nargs='+', type=int, default=[128, 128])
parser.add_argument('-a', '--activations', nargs='+', type=str, default=["relu", "relu"])
parser.add_argument('-r', '--learning_rate', type=float, default=0.001)
parser.add_argument('-b', '--batch_size', type=int, default=64)
parser.add_argument('-m', '--optimizer_method', type=str, default="adam")
parser.add_argument('-d', '--dropout_rates', nargs='+', type=float, default=[0.2, 0.2])
parser.add_argument('-w', '--weight_decay', type=float, default=0.0001)
parser.add_argument('-s', '--early_stopping', type=bool, default=True)
parser.add_argument('-p', '--patience', type=int, default=5)
parser.add_argument('-f', '--print_every', type=int, default=-1)
args = parser.parse_args()


# =======================================
# 2. Load and Preprocess Data
# =======================================
mnist_data = fetch_openml("mnist_784", version=1, as_frame=False)
X = mnist_data["data"]
y = mnist_data["target"]
nb_classes = len(np.unique(y))
y = y.astype(np.int64)
y = y.reshape(-1, 1)

def train_val_test_split(X, y, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Sizes must sum to 1"
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(1 - train_size), random_state=random_state, stratify=y)
    val_prop = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(1 - val_prop), random_state=random_state, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test

X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

scalers = {}
for col in range(X.shape[1]):
    scalers[col] = RobustScaler()
    X_train[:, col:col+1] = scalers[col].fit_transform(X_train[:, col:col+1])
    X_val[:, col:col+1] = scalers[col].transform(X_val[:, col:col+1])
    X_test[:, col:col+1] = scalers[col].transform(X_test[:, col:col+1])

encoder = OneHotEncoder(sparse_output=False)
y_train = encoder.fit_transform(y_train)
y_val = encoder.transform(y_val)
y_test = encoder.transform(y_test)


# =======================================
# 3. Model training with hyperparameter optimization with Optuna if in arguments
# =======================================
if args.optuna:
    # look for optuna config 
    if args.config:
        with open(args.config, "r") as file:
            config = yaml.safe_load(file)
        n_trials = config.get("n_trials", args.trials)
        epochs = config.get("epochs", args.epochs)
        loss_function = config.get("loss_function", "cross_entropy")
    else: # otherwise use args command line
        n_trials = args.trials
        epochs = args.epochs
        loss_function = "cross_entropy"
    early_stopping = True
    print_every = 5

    def objective(
        trial: optuna.trial.Trial,
        epochs: int=40,
        loss_function: str="cross_entropy"
    )-> float:
        """
        Objective function for Optuna hyperparameter optimization

        Parameters:
        trial : optuna.trial.Trial
            Optuna trial object
        epochs : int
            Number of training epochs
        loss_function : str
            Loss function to use    

        Returns:
        float
            Validation loss to minimize
        """

        # Hyperparameters to search
        n_layers = trial.suggest_int("n_layers", 2, 5)
        hidden_layers = [trial.suggest_int(f"n_units_l{i}", 64, 384) for i in range(n_layers)]
        layer_dims = [X.shape[1]] + hidden_layers + [nb_classes]
        # Activation functions per layer
        possible_activations = ["relu", "sigmoid", "tanh", "leaky_relu"]
        activations = [trial.suggest_categorical(f"activation_l{i}", possible_activations) for i in range(n_layers)]
        lr = trial.suggest_float("learning_rate", 1e-4, 0.1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        optimizer_type = trial.suggest_categorical("optimizer_type", ["sgd", "momentum", "adam"])
        dropout_rates = [trial.suggest_float(f"dropout_l{i}", 0.0, 0.5) for i in range(n_layers)]
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        patience = trial.suggest_int("patience", 3, 10)

        # Build and train model
        dnn = DeepNeuralNetwork(
            layer_dims=layer_dims,
            activations=activations,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            loss_function=loss_function,
            optimizer_type=optimizer_type,
            dropout_rates=dropout_rates,
            weight_decay=weight_decay,
            early_stopping=early_stopping,
            patience=patience,
            print_every=print_every,
            plot_loss=False
        )
        _, _ = dnn.fit(X_train, y_train, X_val, y_val)
        A_val_out, _ = dnn._forward(X_val, training=False)
        return dnn._cross_entropy_loss(y_val, A_val_out)
    
    obj = partial(objective, epochs=epochs, loss_function=loss_function)
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(obj, n_trials=n_trials)
    print("Best Trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    # Extract best hyperparameters
    best_params = trial.params
    layer_dims = [X.shape[1]] + [best_params[f"n_units_l{i}"] for i in range(best_params["n_layers"])] + [nb_classes]
    activations = [best_params[f"activation_l{i}"] for i in range(best_params["n_layers"])]
    lr = best_params["learning_rate"]
    batch_size = best_params["batch_size"]
    optimizer_type = "adam"
    dropout_rates = [best_params[f"dropout_l{i}"] for i in range(best_params["n_layers"])]
    weight_decay = best_params["weight_decay"]
    patience = best_params["patience"]

else:
    # Load hyperparameters from config file if provided
    if args.config:
        with open(args.config, "r") as file:
            config = yaml.safe_load(file)
        layer_dims = config.get("layer_dims", args.layer_dims)
        layer_dims = [X.shape[1]] + layer_dims + [nb_classes]
        activations = config.get("activations", args.activations)
        epochs = config.get("epochs", args.epochs)
        lr = config.get("learning_rate", args.learning_rate)
        batch_size = config.get("batch_size", args.batch_size)
        loss_function = config.get("loss_function", "cross_entropy")
        optimizer_type = config.get("optimizer_method", args.optimizer_method)
        dropout_rates = config.get("dropout_rates", args.dropout_rates)
        weight_decay = config.get("weight_decay", args.weight_decay)
        early_stopping = config.get("early_stopping", args.early_stopping)
        patience = config.get("patience", args.patience)
        print_every = config.get("print_every", args.print_every)
    else:
        layer_dims = [X.shape[1]] + args.layer_dims + [nb_classes]
        activations = args.activations
        lr = args.lr
        epochs = args.epochs
        batch_size = args.batch_size
        optimizer_type = args.optimizer_method
        dropout_rates = args.dropout_rates 
        weight_decay = args.weight_decay
        early_stopping = args.early_stopping
        patience = args.patience
        print_every = args.print_every

dnn = DeepNeuralNetwork(
    layer_dims=layer_dims,
    activations=activations,
    lr=lr,
    epochs=epochs*(5 if args.optuna else 1),
    batch_size=batch_size,
    loss_function=loss_function,
    optimizer_type=optimizer_type,
    dropout_rates=dropout_rates,
    weight_decay=weight_decay,
    early_stopping=early_stopping,
    patience=patience,
    print_every=print_every,
)
weights, biases = dnn.fit(X_train, y_train, X_val, y_val)


# =======================================
# 4. Evaluation metrics
# =======================================
y_pred, y_prob = dnn.predict(X_test)
y_pred_encode = encoder.transform(y_pred.reshape(-1, 1))
accuracy = dnn.accuracy(y_test, y_pred_encode)
log_loss = dnn._cross_entropy_loss(y_test, y_prob.T)
printing = False
if printing:
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print(f"Test Log Loss: {log_loss:.4f}")


# =======================================
# 5. Export network configuration and results
# =======================================
# save test metrics
with open("model_configs/mnist/metrics.yaml", "w") as file:
    results = {
        "test_accuracy": float(accuracy),
        "test_log_loss": float(log_loss)
    }
    yaml.dump(results, file)    
# save model configuration

with open("model_configs/mnist/model_layout.yaml", "w") as file:
    config = {
        "layer_dims": layer_dims,
        "activations": activations,
        "learning_rate": lr,
        "epochs": epochs*(5 if args.optuna else 1),
        "batch_size": batch_size,
        "loss_function": loss_function,
        "optimizer_method": optimizer_type,
        "dropout_rates": dropout_rates,
        "weight_decay": weight_decay,
        "early_stopping": early_stopping,
        "patience": patience
    }
    yaml.dump(config, file)

