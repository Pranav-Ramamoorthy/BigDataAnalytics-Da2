import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from minisom import MiniSom
import matplotlib.pyplot as plt

# Load the dataset (adjust the file path as needed)
data = pd.read_csv('financial_transactions.csv')

# Drop unnecessary columns
# We'll use 'Amount', 'TransactionType', and 'Location' for anomaly detection
X = data[['Amount', 'TransactionType', 'Location']]

# Convert categorical columns into numeric values using LabelEncoder
le_transaction_type = LabelEncoder()
le_location = LabelEncoder()

X['TransactionType'] = le_transaction_type.fit_transform(X['TransactionType'])
X['Location'] = le_location.fit_transform(X['Location'])

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the SOM with a grid size (10x10) and number of input features (3)
som = MiniSom(x=10, y=10, input_len=X_scaled.shape[1], sigma=1.0, learning_rate=0.5)

# Randomly initialize the weights of the SOM
som.random_weights_init(X_scaled)

# Train the SOM with 100 iterations
som.train_random(X_scaled, 100)

# Visualize the distance map (anomalies are likely in areas with high distances)
plt.figure(figsize=(10, 10))
plt.pcolor(som.distance_map().T, cmap='coolwarm')  # The distance map as a heatmap
plt.colorbar()

# Add markers for each transaction to the distance map
for i, x in enumerate(X_scaled):
    w = som.winner(x)  # Get the winning node for the transaction
    plt.plot(w[0] + 0.5, w[1] + 0.5, 'o', markerfacecolor='None', markeredgecolor='k', markersize=12, markeredgewidth=2)

plt.title('Financial Anomaly Detection using SOM')
plt.show()

# Get the winning nodes for each transaction
winners = np.array([som.winner(x) for x in X_scaled])

# You can identify anomalies by analyzing the distance map and comparing the winners
# Transactions far from dense clusters may indicate anomalies.
