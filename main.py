from fastapi import FastAPI
import mindspore as ms
from mindspore import nn, ops
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import numpy as np
import pickle

# Load your encoders and scaler
with open("soilcolor_encoder.pkl", "rb") as f:
    soilcolor_encoder = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

num_classes = len(label_encoder.classes_)


# Define model architecture (should match training)
class MLP(nn.Cell):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Dense(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Dense(hidden_dim, num_classes)

    def construct(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Define the columns used at inference
categorical_cols = ["Soilcolor"]
continuous_cols = ["T2M_MAX-W", "T2M_MIN-W", "QV2M-W", "PRECTOTCORR-W"]

input_dim = len(categorical_cols) + len(continuous_cols)
hidden_dim = 64
net = MLP(input_dim, hidden_dim, num_classes)

# Load model checkpoint
param_dict = load_checkpoint("trained_model_reduced.ckpt")
load_param_into_net(net, param_dict)
net.set_train(False)  # inference mode

app = FastAPI()


@app.get("/predict")
def predict(
        Soilcolor: str,
        T2M_MAX_W: float,
        T2M_MIN_W: float,
        QV2M_W: float,
        PRECTOTCORR_W: float
):
    # Encode categorical feature
    soilcolor_val = soilcolor_encoder.transform([Soilcolor])[0]

    # Prepare feature vector
    features = np.array([[soilcolor_val, T2M_MAX_W, T2M_MIN_W, QV2M_W, PRECTOTCORR_W]], dtype=np.float32)

    # Scale continuous features
    features[:, len(categorical_cols):] = scaler.transform(features[:, len(categorical_cols):])

    # Convert to MindSpore Tensor
    input_tensor = ms.Tensor(features, ms.float32)

    # Inference
    logits = net(input_tensor)
    predicted_class = ops.Argmax()(logits).asnumpy()[0]
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

    return {"predicted_crop": predicted_label}
