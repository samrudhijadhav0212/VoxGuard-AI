import csv
import numpy as np

FILE = "dataset_features.csv"
MODEL_FILE = "voxguard_model.npz"

print("=" * 50)
print("       🛡️ VOXGUARD AI")
print("       ML MODEL TRAINING")
print("=" * 50)

# Load dataset
data = []

with open(FILE, "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        data.append([
            float(row["duration"]),
            float(row["rms"]),
            float(row["peak"]),
            float(row["zcr"]),
            float(row["spectral_centroid"]),
            float(row["dominant_frequency"]),
            int(row["label"])
        ])

data = np.array(data)

X = data[:, :-1]
y = data[:, -1]

print("\n📊 Dataset information")
print("----------------------")
print("Total samples:", len(X))
print("Real samples:", np.sum(y == 0))
print("AI samples:", np.sum(y == 1))

# Normalize features
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)

std[std == 0] = 1

X_scaled = (X - mean) / std


# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -50, 50)))


# Logistic regression
weights = np.zeros(X_scaled.shape[1])
bias = 0.0

learning_rate = 0.05
epochs = 3000

print("\n🧠 Training model...")

for epoch in range(epochs):

    predictions = sigmoid(
        np.dot(X_scaled, weights) + bias
    )

    error = predictions - y

    dw = np.dot(X_scaled.T, error) / len(X)
    db = np.mean(error)

    weights -= learning_rate * dw
    bias -= learning_rate * db

    if epoch % 500 == 0:
        loss = -np.mean(
            y * np.log(predictions + 1e-10)
            + (1 - y) * np.log(1 - predictions + 1e-10)
        )

        print(
            "Epoch:",
            epoch,
            "Loss:",
            round(loss, 4)
        )


# Training predictions
probabilities = sigmoid(
    np.dot(X_scaled, weights) + bias
)

predicted = (probabilities >= 0.5).astype(int)

accuracy = np.mean(predicted == y) * 100

print("\n" + "=" * 50)
print("✅ TRAINING COMPLETE")
print("=" * 50)

print("Training accuracy:",
      round(accuracy, 2), "%")


# Save model
np.savez(
    MODEL_FILE,
    weights=weights,
    bias=bias,
    mean=mean,
    std=std
)

print("\n💾 Model saved as:")
print(MODEL_FILE)

print("\n🛡️ VoxGuard ML Engine: READY")
