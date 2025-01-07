import pickle
import pandas as pd
import math
import argparse 
import warnings


def extract_fees(fee_str):
    """Extract percent and pence fees from a fee string."""
    percent, pence = fee_str.split('+')
    return [float(percent.strip().strip('%')), float(pence.strip().strip('p'))]


def apply_fee_adjustment(df, fee_column, transaction_column):
    """Calculate the adjusted fee based on percent and pence values."""
    return df.apply(
        lambda row: row[fee_column][0] + (row[fee_column][1] / row[transaction_column]), axis=1
    )


def calculate_weighted_fees(test, weights):
    """Calculate the weighted fees based on probabilities."""
    test["Fees (%)"] = (
        (test["Mastercard Debit"] * weights["mastercard"] * weights["debit"]) +
        (test["Visa Debit"] * weights["visa"] * weights["debit"]) +
        (test["Mastercard Credit"] * weights["mastercard"] * weights["credit"]) +
        (test["Visa Credit"] * weights["visa"] * weights["credit"]) +
        (test["Mastercard Business Debit"] * weights["mastercard"] * weights["business_debit"]) +
        (test["Visa Business Debit"] * weights["visa"] * weights["business_debit"])
    )


def preprocess_categorical_columns(test):
    """Preprocess categorical columns."""
    test["Is Registered"] = test["Is Registered"].apply(lambda x: 1 if x == 'Yes' else 0)
    test["Accepts Card"] = test["Accepts Card"].apply(lambda x: 1 if x == 'Yes' else 0)
    test["Current Provider"] = test["Current Provider"].apply(lambda x: 0 if pd.isna(x) else x)


def apply_log_transform(test, columns):
    """Apply log transformation to specified columns."""
    for column in columns:
        test[f"{column} (log)"] = test[column].apply(lambda x: math.log10(x))
    test.drop(columns=columns, inplace=True)


def preprocess_data(test_file):
    """Preprocess data."""
    # Define probabilities
    weights = {
        "mastercard": 0.4,
        "visa": 0.6,
        "debit": 0.9,
        "credit": 0.08,
        "business_debit": 0.02,
    }

    # Read data
    test = pd.read_csv(test_file)

    # Process fee columns
    fee_columns = [
        "Mastercard Debit", "Visa Debit", "Mastercard Credit", "Visa Credit",
        "Mastercard Business Debit", "Visa Business Debit"
    ]
    for fee_column in fee_columns:
        test[fee_column] = test[fee_column].apply(extract_fees)

    # Convert flat fee to percentage
    for fee_column in fee_columns:
        test[fee_column] = apply_fee_adjustment(test, fee_column, "Average Transaction Amount")

    # Calculate weighted fees
    calculate_weighted_fees(test, weights)

    # Drop processed fee columns
    test.drop(columns=fee_columns, inplace=True)

    # Preprocess categorical columns
    preprocess_categorical_columns(test)

    # Apply log transformation
    apply_log_transform(test, ["Annual Card Turnover", "Average Transaction Amount"])

    return test


def scale_data(test):
    """Apply standard scaler"""

    X = test.drop(columns=["Fees (%)", "Current Provider"])

    warnings.filterwarnings(
        "ignore",
        message="Trying to unpickle estimator .* from version .* when using version .*",
    )
    with open('./scaler.pkl', 'rb') as f:
        sc = pickle.load(f)
        X_scaled = sc.transform(X)

    return X_scaled


def classify_fees(train_cluster_stats, test_dataset):
    """Gets classification labels"""
    train_cluster_stats.reset_index(drop = True, inplace=True)

    test_dataset = test_dataset.merge(train_cluster_stats[["cluster", "25%", "75%"]], left_on="cluster", right_on="cluster", how="left")

    for idx, price in enumerate(test_dataset["Fees (%)"]):
        if price < test_dataset.loc[idx, '25%']:
            test_dataset.loc[idx, "Fees Classification"] = 'competitive'
        elif price > test_dataset.loc[idx, "75%"]:
            test_dataset.loc[idx, "Fees Classification"] = 'non-competitive'
        else:
            test_dataset.loc[idx, "Fees Classification"] = 'normal'

    return test_dataset


def get_results():
    parser = argparse.ArgumentParser(description="Process transaction data.")
    parser.add_argument("file", type=str, help="Path to the CSV file to process")
    args = parser.parse_args()

    # Preprocess the test data
    processed_test_data = preprocess_data(args.file)

    # Scale the data
    X_scaled = scale_data(processed_test_data)

    # Load the KMeans model and assign clusters
    
    with open('./kmeans.pkl', 'rb') as f:
        kmeans = pickle.load(f)
        processed_test_data["cluster"] = kmeans.predict(X_scaled)

    # Load cluster statistics and classify fees
    cluster_stats = pd.read_csv("./cluster_stats.csv")
    results = classify_fees(cluster_stats, processed_test_data)

    # Save the results to a CSV file
    results.to_csv("./results.csv", index=False)
    print("Results saved to ./results.csv")

    # Print classification counts
    print(results["Fees Classification"].value_counts())


# Run the script
if __name__ == "__main__":
    get_results()
