import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())
    return df

if __name__ == "__main__":
    df = pd.read_csv("Mendeley_ae.csv", sep=';')

print(df.shape)
print(df.columns)
