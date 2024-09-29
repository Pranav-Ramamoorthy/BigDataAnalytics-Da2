# BigDataAnalytics-Da2
 

---
# **Financial Anomaly Detection Using SOM**

## **Project Description**
This project implements a **Self-Organizing Map (SOM)** to detect anomalies in financial transactions. The model analyzes transaction attributes such as `Amount`, `TransactionType`, and `Location` to identify potentially fraudulent or unusual transactions.

## **Dataset**
The dataset consists of financial transaction records with the following columns:
- **Amount**: The value of the transaction.
- **TransactionType**: The type of transaction (e.g., debit, credit, transfer).
- **Location**: The location where the transaction occurred.

The dataset is preprocessed by normalizing numerical data and encoding categorical variables (`TransactionType`, `Location`).

## **Requirements**

### **Software**
- **Python Version**: 3.8 or higher
- **Libraries**:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `minisom`

### **Hardware**
- A standard computer with:
  - **CPU**: Quad-core processor
  - **RAM**: 4GB or more
  - **Storage**: 200MB for dataset and scripts

## **Installation Instructions**
1. Install the necessary libraries using pip:
   ```bash
   pip install pandas numpy scikit-learn matplotlib minisom
   ```

2. Clone or download the repository to your local machine.

3. Place your dataset (e.g., `financial_anomaly_data.csv`) in the same folder as the script or adjust the file path in the script.

## **Execution Steps**

1. **Preprocess the data**: 
   - Drop unnecessary columns and normalize the data using `MinMaxScaler`.
   - Encode categorical variables (`TransactionType` and `Location`) using `LabelEncoder`.

2. **Train the SOM model**: 
   - Initialize the SOM with a grid size (e.g., 10x10) and train it with the preprocessed data.

3. **Visualize the results**: 
   - Display the SOM's U-matrix using a heatmap to identify areas of high anomaly likelihood.

4. **Detect anomalies**: 
   - Transactions in regions with high SOM distances (above the anomaly threshold) are classified as anomalies.

### **Running the Script**
Run the Python script in your terminal or IDE:
```bash
python som_anomaly_detection.py
```

### **Output**
- A heatmap visualization of the SOM, with markers for each transaction.
- An array indicating which transactions are classified as anomalies (1) or normal (0).

## **Files**
- `som_anomaly_detection.py`: The main script for training the SOM and detecting anomalies.
- `financial_anomaly_data.csv`: Sample dataset (you need to provide your own dataset or adjust the file path).
- `som_model.pkl`: The saved SOM model after training.

## **Sample Dataset Format**

| Amount | TransactionType | Location |
|--------|-----------------|----------|
| 150.75 | debit           | New York |
| 5000.00| credit          | London   |

---





## **About the Dataset**

This dataset contains **217,441** financial transactions, with details such as the transaction amount, type of transaction, and location. It is designed for detecting anomalies in financial activity, such as unusually large transactions or transactions happening in unexpected locations.

The dataset includes the following columns:

- **Timestamp**: The date and time of the transaction.
- **TransactionID**: A unique identifier for each transaction.
- **AccountID**: The identifier of the account associated with the transaction.
- **Amount**: The monetary value of the transaction. The amounts vary significantly, with a minimum of 10.51 and a maximum of 978,942.26.
- **Merchant**: The merchant involved in the transaction.
- **TransactionType**: The type of transaction (e.g., debit, credit).
- **Location**: The geographical location of the transaction.

#### **Dataset Statistics**:
- **Total Transactions**: 217,441
- **Average Transaction Amount**: 50,090.03
- **Minimum Transaction Amount**: 10.51
- **Maximum Transaction Amount**: 978,942.26

The dataset is ideal for financial anomaly detection, where abnormal patterns in transaction behavior can be identified based on features like the amount, location, and transaction type.

--- 
# Explanation of the code
This code implements a **Self-Organizing Map (SOM)** to detect anomalies in financial transactions. The SOM is an unsupervised learning algorithm that maps high-dimensional data into a 2D grid, helping to visualize clusters and detect outliers. Here’s a detailed breakdown of what each part of the code does:

---

### **1. Loading the Dataset**

```python
data = pd.read_csv('financial_transactions.csv')
```

- The dataset, assumed to be in a CSV file named `financial_transactions.csv`, is loaded into a Pandas DataFrame.
  
### **2. Selecting Relevant Columns**

```python
X = data[['Amount', 'TransactionType', 'Location']]
```

- The dataset contains many columns, but only `Amount`, `TransactionType`, and `Location` are selected for anomaly detection. These features represent important aspects of each financial transaction.

### **3. Encoding Categorical Variables**

```python
le_transaction_type = LabelEncoder()
le_location = LabelEncoder()

X['TransactionType'] = le_transaction_type.fit_transform(X['TransactionType'])
X['Location'] = le_location.fit_transform(X['Location'])
```

- Categorical data, such as `TransactionType` and `Location`, must be converted into numerical form for the SOM to process.
- **LabelEncoder** is used to convert these categorical values into integers (e.g., "debit" may become 0 and "credit" may become 1).

### **4. Normalizing the Data**

```python
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

- The SOM performs better when input data is normalized. 
- **MinMaxScaler** scales each feature between 0 and 1, ensuring that the features like `Amount`, `TransactionType`, and `Location` are on the same scale.

### **5. Initializing and Training the SOM**

```python
som = MiniSom(x=10, y=10, input_len=X_scaled.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(X_scaled)
som.train_random(X_scaled, 100)
```

- **MiniSom** is used to create a SOM with a **10x10 grid** (100 nodes).
  - The grid size `x=10, y=10` defines the size of the map.
  - `input_len=X_scaled.shape[1]` means the input data has 3 features (Amount, TransactionType, and Location).
  - `sigma=1.0` controls the radius of the influence of nodes during training, and `learning_rate=0.5` determines how much the weights are updated at each step.
  
- The **weights of the SOM** are initialized randomly using the `random_weights_init` method, based on the scaled data.

- **Training**: The SOM is trained for **100 iterations** using `train_random`, which adjusts the weights of the SOM nodes to fit the input data.

### **6. Visualizing the SOM (Distance Map)**

```python
plt.figure(figsize=(10, 10))
plt.pcolor(som.distance_map().T, cmap='coolwarm')  # The distance map as a heatmap
plt.colorbar()
```

- After training, a **distance map** is generated. This is a 2D heatmap that shows the **distance** between SOM nodes. High distances between nodes often indicate areas with **outliers** or **anomalies**.
  - Areas of **low distance** (closely related data points) are cool colors (like blue), while areas of **high distance** (potential anomalies) are warmer colors (like red).

### **7. Plotting Markers for Each Transaction**

```python
for i, x in enumerate(X_scaled):
    w = som.winner(x)  # Get the winning node for the transaction
    plt.plot(w[0] + 0.5, w[1] + 0.5, 'o', markerfacecolor='None', markeredgecolor='k', markersize=12, markeredgewidth=2)
```

- For each transaction in the dataset, the **winning node** on the SOM is determined.
- Each transaction is plotted as a marker (circles) on the SOM, overlaid on the heatmap.
  - The `som.winner(x)` method returns the grid position (node) that best matches each transaction. These nodes are plotted as black markers on the heatmap.

### **8. Identifying Anomalies (Winning Nodes)**

```python
winners = np.array([som.winner(x) for x in X_scaled])
```

- This line calculates the **winning node** for each transaction in the dataset, which can help identify **clusters** of similar transactions and potential anomalies (outliers far from dense clusters).
  
Transactions that map to areas of high distance or sparse clusters are considered potential anomalies.

---

### **Key Points:**
- **Self-Organizing Maps (SOMs)** are used here to identify unusual patterns in transactions. SOM creates a 2D grid where similar transactions are grouped together, while outliers are more distant.
- The distance map gives a visual clue about **clusters** and **anomalies** in the data.
- The black circles in the visualization represent individual transactions, allowing you to visually inspect areas where anomalies may be present. 

In this context, financial transactions that are mapped to regions of the SOM with higher distances might be flagged as **anomalous** transactions, possibly indicating fraud.



---

## **How to Run the Financial Anomaly Detection Script**

### **Step 1: Install Required Libraries**

Before running the script, ensure that the required Python libraries are installed. You can install them using `pip`:

```bash
pip install numpy pandas scikit-learn matplotlib minisom
```

### **Step 2: Prepare the Dataset**

1. Ensure your dataset (`financial_transactions.csv`) is available in the same directory as your Python script. The dataset should have the columns: `Amount`, `TransactionType`, and `Location`.

2. If your dataset has a different name, modify the code to match the dataset file name:
   
```python
data = pd.read_csv('your_dataset.csv')
```

### **Step 3: Run the Script**

You can run the Python script using a terminal or any Python-compatible IDE (like **VS Code**, **PyCharm**, or **Jupyter Notebook**).

#### **Terminal**:

1. Open a terminal (Command Prompt or Terminal on macOS/Linux).
2. Navigate to the directory where your script is located:
   
```bash
cd path_to_your_script
```

3. Run the script:

```bash
python your_script_name.py
```

#### **Jupyter Notebook**:

If you're using Jupyter Notebook:
1. Copy and paste the script code into a Jupyter cell.
2. Run the cell.

### **Step 4: Interpret the Results**

- After running the script, a heatmap will be displayed. This **SOM distance map** will highlight regions of dense transaction clusters and outliers (anomalies).
  - Cool colors (blue) represent dense clusters of normal transactions.
  - Warm colors (red) represent sparse areas where anomalies are likely.
  
- Each transaction is marked with a black circle. The location of these circles in the distance map can help identify anomalous transactions.

### **Step 5: Review Anomalies**

1. The **winning nodes** for each transaction are printed. These nodes indicate where each transaction maps onto the SOM grid.
   
2. By reviewing the distance map, you can identify clusters of normal transactions and potential outliers (far from the dense clusters).

---

This should guide you through running the SOM anomaly detection script successfully. 

---
# OUTPUTS
# HeatMap
![WhatsApp Image 2024-09-29 at 22 32 30_ca012cff](https://github.com/user-attachments/assets/dc4b535f-b8ce-4b30-b99f-920b8d37b4da)

# Final Graph
![WhatsApp Image 2024-09-29 at 23 12 46_f610966f](https://github.com/user-attachments/assets/c223e655-b128-4491-b777-a0605e9ecceb)

