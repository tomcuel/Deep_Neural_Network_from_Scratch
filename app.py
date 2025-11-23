# Save this as iris_dnn_streamlit.py
import streamlit as st
import streamlit.components.v1 as components
import yaml
import subprocess
import pandas as pd
st.set_page_config(layout="wide")


# -----------------------------
# Page setup
# -----------------------------
st.title("Deep Neural Network Explorer")


# -----------------------------
# Sidebar: Hyperparameters
# -----------------------------
st.sidebar.header("DNN Hyperparameters")
dataset = st.sidebar.selectbox("Dataset", ["Iris", "MNIST"], index=0)
dataset = dataset.lower().replace(" ", "_")
use_optuna = st.sidebar.checkbox("Use Optuna Optimization", value=False)

# Define dataset-specific defaults
DATASET_DEFAULTS = {
    "iris": {
        "num_layers": 2,
        "layer_dims": 10,
        "activations": "relu",
        "dropouts": 0.1,
        "learning_rate": 0.001,
        "epochs": 500,
        "batch_size_index": 2,
        "weight_decay": 0.0001,
        "patience": 100
    },
    "mnist": {
        "num_layers": 2,
        "layer_dims": 128,
        "activations": "relu",
        "dropouts": 0.2,
        "learning_rate": 0.001,
        "epochs": 40,
        "batch_size_index": 3,
        "weight_decay": 0.0001,
        "patience": 5
    }
}
defaults = DATASET_DEFAULTS[dataset]

# Reset all defaults when dataset changes
if st.session_state.get("dataset_selected") != dataset:
    st.session_state.dataset_selected = dataset
    # Remove dynamic keys from previous dataset
    for k in list(st.session_state.keys()):
        if k.startswith(("units_", "act_", "drop_")) or k in [
            "num_layers", "layer_dims", "activations", "dropouts",
            "learning_rate", "epochs", "batch_size_index",
            "weight_decay", "patience"
        ]:
            st.session_state.pop(k)

    st.session_state.num_layers       = defaults["num_layers"]
    st.session_state.layer_dims       = [defaults["layer_dims"]] * defaults["num_layers"]
    st.session_state.activations      = [defaults["activations"]] * defaults["num_layers"]
    st.session_state.dropouts         = [defaults["dropouts"]] * defaults["num_layers"]
    st.session_state.learning_rate    = defaults["learning_rate"]
    st.session_state.epochs           = defaults["epochs"]
    st.session_state.batch_size_index = defaults["batch_size_index"]
    st.session_state.weight_decay     = defaults["weight_decay"]
    st.session_state.patience         = defaults["patience"]

# optuna mode
if use_optuna:
    st.sidebar.info("Optuna optimization selected, hyperparameter inputs will be ignored")
    loss_function = "cross_entropy"
    n_trials_default = 50 if dataset == "iris" else 20
    n_epochs_default = 500 if dataset == "iris" else 40
    n_trials = st.sidebar.slider("Number of Optuna Trials", 1, 200, value=n_trials_default)
    epochs = st.sidebar.slider("Epochs", 1, 5000, value=n_epochs_default)

    with open(f"model_configs/{dataset}/config_optuna.yaml", "w") as f:
        config = {
            "n_trials": n_trials,
            "epochs": epochs,
            "loss_function": loss_function,
        }
        yaml.dump(config, f)

# manual mode
else:
    st.sidebar.info("Manual hyperparameter configuration selected")

    # Buttons to add/remove layers
    col1, col2 = st.sidebar.columns(2)
    if col1.button("Add Layer"):
        st.session_state.num_layers += 1
        st.session_state.layer_dims.append(defaults["layer_dims"])
        st.session_state.activations.append(defaults["activations"])
        st.session_state.dropouts.append(defaults["dropouts"])
    if col2.button("Remove Layer") and st.session_state.num_layers > 1:
        st.session_state.num_layers -= 1
        st.session_state.layer_dims.pop()
        st.session_state.activations.pop()
        st.session_state.dropouts.pop()

    # Render layers dynamically
    layer_dims, activations, dropout_rates = [], [], []
    for i in range(st.session_state.num_layers):
        st.sidebar.markdown(f"**Layer {i+1}**")

        # Units
        u_key = f"units_{i}"
        if u_key not in st.session_state:
            st.session_state[u_key] = st.session_state.layer_dims[i]
        units = st.sidebar.number_input("Units", 1, 512, key=u_key)
        st.session_state.layer_dims[i] = units
        layer_dims.append(units)

        # Activation
        a_key = f"act_{i}"
        if a_key not in st.session_state:
            st.session_state[a_key] = st.session_state.activations[i]
        act = st.sidebar.selectbox("Activation", ["relu", "sigmoid", "tanh", "leaky_relu"], index=["relu","sigmoid","tanh","leaky_relu"].index(st.session_state[a_key]), key=a_key)
        st.session_state.activations[i] = act
        activations.append(act)

        # Dropout
        d_key = f"drop_{i}"
        if d_key not in st.session_state:
            st.session_state[d_key] = st.session_state.dropouts[i]
        dropout = st.sidebar.number_input("Dropout", 0.0, 0.9, step=0.02, key=d_key)
        st.session_state.dropouts[i] = dropout
        dropout_rates.append(dropout)
       
    # Learning rate (float > 0)
    learning_rate = st.sidebar.slider("Learning Rate", 0.00001, 0.1, value=defaults["learning_rate"], step=0.0001, format="%.4f")
    if not isinstance(learning_rate, float) or learning_rate <= 0:
        st.sidebar.error("Learning rate must be a positive float!")

    # Epochs (int > 0)
    epochs = st.sidebar.slider("Epochs", 1, 5000, value=defaults["epochs"])
    if not isinstance(epochs, int) or epochs <= 0:
        st.sidebar.error("Number of epochs must be a positive integer!")

    # Batch size (int > 0)
    batch_size = st.sidebar.selectbox("Batch Size", [8, 16, 32, 64, 128], index=defaults["batch_size_index"])
    if not isinstance(batch_size, int) or batch_size <= 0:
        st.sidebar.error("Batch size must be a positive integer!")

    # Optimizer
    optimizer_method = st.sidebar.selectbox("Optimizer", ["sgd", "momentum", "adam"], index=2)

    # Weight decay (float >= 0)
    weight_decay = st.sidebar.slider("Weight Decay", 0.000001, 0.01, value=defaults["weight_decay"], step=0.000001, format="%.6f")
    if not isinstance(weight_decay, float) or weight_decay < 0:
        st.sidebar.error("Weight decay must be a non-negative float!")

    # Early stopping
    early_stopping = st.sidebar.checkbox("Early Stopping", value=True)

    # Patience (int > 0)
    patience = st.sidebar.slider("Early Stopping Patience", 1, 500, value=defaults["patience"])
    if not isinstance(patience, int) or patience < 1:
        st.sidebar.error("Patience must be a positive integer!")

    loss_function = "cross_entropy"
    print_every = -1  # Not used in this script

    with open(f"model_configs/{dataset}/config.yaml", "w") as f:
        config = {
            "layer_dims": layer_dims,
            "activations": activations,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "loss_function": loss_function,
            "optimizer_method": optimizer_method,
            "dropout_rates": dropout_rates,
            "weight_decay": weight_decay,
            "early_stopping": early_stopping,
            "patience": patience,
            "print_every": print_every
        }
        yaml.dump(config, f)


# -----------------------------
# Pipeline results content
# -----------------------------
if st.button("Run DNN Pipeline"):
    run_msg = st.info("Running the DNN pipeline... Please wait")

    # build command
    cmd = ["python", f"{dataset}.py"]
    if use_optuna:
        cmd.append("-o")
        cmd.append("-c")
        cmd.append(f"model_configs/{dataset}/config_optuna.yaml")
    else:
        cmd.append("-c")
        cmd.append(f"model_configs/{dataset}/config.yaml")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        run_msg.empty()
        st.error(f"Pipeline failed: {e}")
        st.stop()
    
    run_msg.empty() # remove the running info

    # Load results
    with open(f"model_configs/{dataset}/metrics.yaml", "r") as f:
        metrics = yaml.safe_load(f)
    with open(f"model_configs/{dataset}/model_layout.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Centered layout for results
    with st.container():
        st.subheader("DNN Metrics")
        col1, col2 = st.columns(2)
        col1.metric("Test Accuracy", f"{metrics['test_accuracy']*100:.2f}%")
        col2.metric("Test Log Loss", f"{metrics['test_log_loss']:.4f}")

        # Neural Network Diagram
        st.subheader("Neural Network Architecture")
        layer_dims = config["layer_dims"]
        activations = config["activations"]
        dropout = config["dropout_rates"]
        L = len(layer_dims)

        # Adaptive sizing
        max_width = 1200
        max_height = 600
        page_width_param = st.query_params.get("page_width", ["900"])
        svg_width = min(int(page_width_param[0]), max_width)
        svg_height = min(int(svg_width * 0.5), max_height)

        left_margin, right_margin, top_margin, bottom_margin = 40, 40, 40, 40
        usable_w = svg_width - left_margin - right_margin
        usable_h = svg_height - top_margin - bottom_margin

        x_spacing = usable_w / max(1, L-1)
        neuron_radius = 10
        neuron_gap = 22

        svg_elems = [f'<rect x="0" y="0" width="{svg_width}" height="{svg_height}" fill="#ffffff" rx="8" />']

        # Compute neuron centers
        layer_centers = []
        for i, n_units in enumerate(layer_dims):
            layer_h = max((n_units-1)*neuron_gap, 0)
            x = left_margin + i*x_spacing
            y0 = top_margin + (usable_h - layer_h)/2
            centers = [(x, y0+j*neuron_gap) for j in range(n_units)]
            layer_centers.append(centers)

        # Draw connectors
        for i in range(1, L):
            for cx, cy in layer_centers[i]:
                for px, py in layer_centers[i-1]:
                    svg_elems.append(f'<line x1="{px+neuron_radius}" y1="{py}" x2="{cx-neuron_radius}" y2="{cy}" stroke="#bdbdbd" stroke-width="1" />')

        # Draw neurons + labels
        for i, centers in enumerate(layer_centers):
            fill, stroke = ("#4A90E2","#1F5BAA") if i==0 else ("#66BB6A","#338a3e") if i==L-1 else ("#FFD54F","#F9A825")
            for cx, cy in centers:
                svg_elems.append(f'<circle cx="{cx}" cy="{cy}" r="{neuron_radius}" fill="{fill}" stroke="{stroke}" stroke-width="2" />')
            layer_type = "Input" if i==0 else "Output" if i==L-1 else f"Hidden {i}"
            svg_elems.append(f'<text x="{centers[0][0]-30}" y="30" font-family="sans-serif" font-size="14" fill="#333">{layer_type}</text>')
            svg_elems.append(f'<text x="{centers[0][0]-30}" y="50" font-family="sans-serif" font-size="12" fill="#666">{len(centers)} units</text>')
            if 0<i<L-1:
                svg_elems.append(f'<text x="{centers[0][0]-30}" y="70" font-family="sans-serif" font-size="12" fill="#aa0000">dropout: {dropout[i-1]}</text>')
            if i>0:
                svg_elems.append(f'<text x="{centers[0][0]-30}" y="90" font-family="sans-serif" font-size="12" fill="#666">{activations[i-1]}</text>')

        svg = f'<svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg">{"".join(svg_elems)}</svg>'
        components.html(svg, width=svg_width, height=svg_height)

        # Training hyperparameters table
        st.subheader("Training Hyperparameters")
        hp = {
            "Learning Rate": config.get("learning_rate"),
            "Epochs": config["epochs"],
            "Batch Size": config["batch_size"],
            "Optimizer": config["optimizer_method"],
            "Loss Function": config["loss_function"],
            "Weight Decay": config["weight_decay"],
            "Early Stopping": config["early_stopping"],
            "Patience": config["patience"]
        }
        hp_df = pd.DataFrame({"Hyperparameter": list(hp.keys()), "Value": [str(v) for v in hp.values()]})
        st.table(hp_df)
    
