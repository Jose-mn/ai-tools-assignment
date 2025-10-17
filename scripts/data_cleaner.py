import pandas as pd

def clean_text_data(file_path):
    df = pd.read_csv(file_path)
    df['cleaned'] = df['review'].str.lower().str.replace('[^a-z ]', '', regex=True)
    print("✅ Data cleaned successfully!")
    return df

if __name__ == "__main__":
    print("Running data cleaner script...")
    clean_text_data("../assets/sample_reviews.csv")
