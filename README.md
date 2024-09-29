# BigDataAnalytics-Da2
Step 1: Understand the Dataset
Dataset Overview: This dataset contains details about financial transactions. Each row represents one transaction, with the following columns:

Amount: The amount of the transaction.
TransactionType: The type of transaction (e.g., credit, debit, transfer).
Location: The geographical location of the transaction.
Dataset Structure:
Timestamp	TransactionID	AccountID	Amount	Merchant	TransactionType	Location
2024-09-18 10:45:12	123456789	1001	150.75	Shop A	debit	New York
2024-09-18 10:45:13	123456790	1002	5000.00	Shop B	credit	London
Step 2: Install Necessary Libraries
You need the following libraries to run the project:

pandas: for data manipulation.
numpy: for numerical computations.
minisom: for the SOM algorithm.
scikit-learn: for data preprocessing.
matplotlib: for data visualization.
Install them using the following command:

bash
Copy code
pip install pandas numpy scikit-learn matplotlib minisom
Step 3: Load and Preprocess the Dataset
Load the dataset:

Load the dataset into a pandas DataFrame.
Drop unnecessary columns:

We focus on the features Amount, TransactionType, and Location for anomaly detection.
Encode categorical variables:

Convert categorical columns (TransactionType, Location) into numeric values using LabelEncoder.
Normalize the data:

Use MinMaxScaler to normalize all features between 0 and 1.
python
Copy code
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load the dataset (adjust the file path)
data = pd.read_csv('/path_to_file/financial_anomaly_data.csv')

# Drop unnecessary columns and use 'Amount', 'TransactionType', 'Location'
X = data[['Amount', 'TransactionType', 'Location']]

# Convert categorical columns into numeric values using LabelEncoder
le_transaction_type = LabelEncoder()
le_location = LabelEncoder()

X['TransactionType'] = le_transaction_type.fit_transform(X['TransactionType'])
X['Location'] = le_location.fit_transform(X['Location'])

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
Step 4: Train the Self-Organizing Map (SOM)
Initialize SOM:

Define the SOM with a grid size (e.g., 10x10) and input length equal to the number of features (in this case, 3 features).
Train the SOM:

SOM is trained using the preprocessed data with 100 iterations.
python
Copy code
from minisom import MiniSom

# Initialize the SOM with a 10x10 grid and 3 input features (Amount, TransactionType, Location)
som = MiniSom(x=10, y=10, input_len=X_scaled.shape[1], sigma=1.0, learning_rate=0.5)

# Randomly initialize SOM weights
som.random_weights_init(X_scaled)

# Train the SOM with 100 iterations
som.train_random(X_scaled, 100)
Step 5: Visualize the SOM and Detect Anomalies
Visualize the U-Matrix:
The U-matrix shows distances between neurons. Large distances may indicate anomalies.
Mark each transaction:
Overlay each transaction on the U-matrix.
python
Copy code
import matplotlib.pyplot as plt
import numpy as np

# Visualize the distance map
plt.figure(figsize=(10, 10))
plt.pcolor(som.distance_map().T, cmap='coolwarm')  # U-matrix
plt.colorbar()

# Mark transactions on the SOM
for i, x in enumerate(X_scaled):
    w = som.winner(x)  # Get the winning node
    plt.plot(w[0] + 0.5, w[1] + 0.5, 'o', markerfacecolor='None', markeredgecolor='k', markersize=12, markeredgewidth=2)

plt.title('Financial Anomaly Detection using SOM')
plt.show()
Step 6: Anomaly Detection Based on SOM Output
Winning nodes:

Identify the winning node for each transaction.
Set threshold:

Define a threshold for anomalies based on the mean and standard deviation of the SOM distance map.
Classify anomalies:

Transactions with distances above the threshold are marked as anomalies.
python
Copy code
# Get the winning nodes for each transaction
winners = np.array([som.winner(x) for x in X_scaled])

# Set a threshold for anomaly detection
threshold = np.mean(som.distance_map()) + np.std(som.distance_map())

# Classify each transaction as normal (0) or anomaly (1) based on the threshold
anomalies = np.zeros(len(X_scaled), dtype=int)
for i, x in enumerate(X_scaled):
    w = som.winner(x)
    distance = som.distance_map()[w]
    if distance > threshold:
        anomalies[i] = 1

# Print detected anomalies
print("\nAnomalies detected (0 = normal, 1 = anomaly):")
print(anomalies)
Step 7: Save the Model
Save the trained SOM model and results:

python
Copy code
import pickle

# Save the SOM model
with open('som_model.pkl', 'wb') as f:
    pickle.dump(som, f)
