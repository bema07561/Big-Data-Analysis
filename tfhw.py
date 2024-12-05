import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Function to load data from the file
def load_data(filename):
    with open(filename, 'r', encoding='utf-16') as file:
        content = file.read().replace('"', '')
    from io import StringIO
    data = np.loadtxt(StringIO(content), delimiter=',')
    features = data[:, :5]
    targets = data[:, 5]
    train_mask = data[:, -1] == 1
    test_mask = data[:, -1] == 2

    X_train = features[train_mask]
    y_train = targets[train_mask]

    X_test = features[test_mask]
    y_test = targets[test_mask]
    return X_train, y_train, X_test, y_test

# Function to build and train the model
def train_model(X_train, y_train, epochs=200):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),  # Input layer
        tf.keras.layers.experimental.preprocessing.Normalization(),  # Normalization layer
        tf.keras.layers.Dense(128, activation='relu'),  # Hidden layer
        tf.keras.layers.Dense(64, activation='relu'),  # Hidden layer
        tf.keras.layers.Dropout(0.2),  # Dropout layer
        tf.keras.layers.Dense(1)  # Output layer
    ])

    model.compile(optimizer='adam', loss='mse')
    
    # Introduce early stopping to avoid overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        validation_split=0.2,
        callbacks=[early_stopping]
    )
    
    # Get minimum validation loss
    min_val_loss = min(history.history['val_loss'])
    
    return model, min_val_loss

# Function to evaluate the model on the test set and visualize results
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Save results to result.txt file
    with open('result.txt', 'w') as file:
        for pred, true in zip(y_pred, y_test):
            file.write(f"Predicted: {pred[0]:.2f}, Actual: {true:.2f}\n")

    # Plot and save visualization
    plt.scatter(y_test, y_pred, color='blue', label='Predictions')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='y=x line')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.title('True vs Predicted Values')
    plt.savefig('results_plot.png')
    plt.show()

# Load data
X_train, y_train, X_test, y_test = load_data('bupa.txt')

# Train the model
model, min_val_loss = train_model(X_train, y_train)

# Print key intermediate results
print(f"Number of samples in the training set: {len(X_train)}")
print(f"Number of samples in the test set: {len(X_test)}")
print(f"Minimum validation loss during training: {min_val_loss}")

# Evaluate the model on the test set and visualize results
evaluate_model(model, X_test, y_test)
