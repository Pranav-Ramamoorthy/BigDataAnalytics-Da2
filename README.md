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


