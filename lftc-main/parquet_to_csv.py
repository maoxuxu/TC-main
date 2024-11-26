import pandas as pd

def parquet_to_csv(parquet_file, csv_file):
    # Read Parquet file
    df = pd.read_parquet(parquet_file)

    # Save data as CSV file
    df.to_csv(csv_file, index=False)

    print(f"Parquet file has been converted to {csv_file}")



if __name__ == '__main__':
    parquet_file = 'parquet/R8/train-00000-of-00001.parquet'
    csv_file = 'csv/R8/train.csv'
    parquet_to_csv(parquet_file, csv_file)
