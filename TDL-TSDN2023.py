import streamlit as st
import pandas as pd
import torch
import hashlib
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertModel
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# Function to load your model and tokenizer
def load_model():
    model_path = './Model/distilbert_model'
    tokenizer_path = './Model/distilbert_tokenizer'
    label_encoder_path = '.Model/label_encoder/label_encoder.pkl'
    loaded_tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
    loaded_model = DistilBertModel.from_pretrained(model_path, num_labels=2)
    #loaded_label_encoder = joblib.load(label_encoder_path)
    return loaded_model, loaded_tokenizer

# Custom hashing functions
def hash_sha256(cell_value):
    return hashlib.sha256(str(cell_value).encode()).hexdigest()

def hash_md5(cell_value):
    try:
        # Check if cell_value meets specific criteria
        if (isinstance(cell_value, str) and 
            len(cell_value) == 16 and 
            11 <= int(cell_value[:2]) <= 95 and
            1 <= int(cell_value[2:4]) <= 79 and
            1 <= int(cell_value[4:6]) <= 55 and
            1 <= int(cell_value[6:8]) <= 71 and
            1 <= int(cell_value[8:10]) <= 12 and
            1900 <= int(cell_value[10:12]) <= 2024):
            # Apply MD5 hashing
            return hashlib.md5(cell_value.encode()).hexdigest()
    except ValueError:
        # If conversion to integer fails, do not hash
        pass
    # Return original value if it doesn't meet criteria
    return cell_value

# Custom hashing function for anonymization
def hash_condition(cell_value):
    try:
        if (isinstance(cell_value, str) and 
            len(cell_value) == 16 and 
            11 <= int(cell_value[:2]) <= 95 and
            1 <= int(cell_value[2:4]) <= 79 and
            1 <= int(cell_value[4:6]) <= 55 and
            1 <= int(cell_value[6:8]) <= 71 and
            1 <= int(cell_value[8:10]) <= 12 and
            1900 <= int(cell_value[10:12]) <= 2024):
            return hashlib.md5(cell_value.encode()).hexdigest()
    except ValueError:
        pass  # Handle the case where int conversion fails
    return cell_value

# Load the model (adjust as necessary)
model, tokenizer = load_model()

# Streamlit interface
st.title("Auto-Anonymizer")

# File upload section
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # Read the file
    if uploaded_file.name.lower().endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.lower().endswith(('.xlsx', '.xls')):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format. Please provide a CSV or Excel file.")

# Anonymize data
if st.button('Anonymize Data'):
    if uploaded_file is not None:
        # Apply your anonymization logic here
        # For example:
        anonymized_df = df.applymap(hash_condition)
        # Display anonymized data
        st.dataframe(anonymized_df)

        # Convert DataFrame to CSV for download
        csv = anonymized_df.to_csv(index=False)
        st.download_button(
            label="Download Anonymized Data",
            data=csv,
            file_name='anonymized_data.csv',
            mime='text/csv',
        )
