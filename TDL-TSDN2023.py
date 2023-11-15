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
    loaded_label_encoder = joblib.load(label_encoder_path)
    return loaded_model, loaded_tokenizer

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
        # Transform the data
        melted_table = pd.melt(df, value_vars=df.columns, var_name='Column', value_name='Data')
        
        # Tokenize and encode the text data for DistilBert
        encodings = loaded_tokenizer(list(melted_table['Data'].astype(str)), truncation=True, padding=True, max_length=64)
        dataset = TensorDataset(torch.tensor(encodings['input_ids']), torch.tensor(encodings['attention_mask']))
        loader = DataLoader(dataset, batch_size=8, shuffle=False)
        
        # Predict with DistilBert
        loaded_model.eval()
        all_preds = []
        with torch.no_grad():
            for batch in loader:
                input_ids, attention_mask = batch
                outputs = loaded_model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
        
        # Add DistilBert predicted labels to the DataFrame
        melted_table['Sensitive_DB'] = loaded_label_encoder.inverse_transform(all_preds)
        
        # Predict with logistic regression
        text_data = melted_table['Data'].astype(str).tolist()
        logistic_preds = loaded_logistic_model.predict(text_data)
        
        # Add logistic regression predicted labels to the DataFrame
        melted_table['Sensitive_LR'] = loaded_label_encoder.inverse_transform(logistic_preds)

        # Apply SHA-256 hashing
        melted_table.loc[melted_table['Predicted_Sensitive'] == 1, 'Data'] = \
            melted_table.loc[melted_table['Predicted_Sensitive'] == 1, 'Data'].apply(
                lambda x: hashlib.sha256(str(x).encode()).hexdigest())
        # Apply custom MD5 hashing function
        reverted_table = melted_table.applymap(hash_condition)

        # Determine the file format for download based on the uploaded file
        file_format = uploaded_file.name.split('.')[-1].lower()
        if file_format == 'csv':
            csv = anonymized_df.to_csv(index=False)
            st.download_button(label="Download Anonymized Data as CSV", data=csv, file_name='anonymized_data.csv', mime='text/csv')
        elif file_format in ['xlsx', 'xls']:
            towrite = io.BytesIO()
            anonymized_df.to_excel(towrite, index=False)
            towrite.seek(0)
            st.download_button(label=f"Download Anonymized Data as {file_format.upper()}", data=towrite, file_name=f'anonymized_data.{file_format}', mime='application/vnd.ms-excel')
        else:
            st.error("Unsupported file format. Please provide a CSV or Excel file.")

This code first applies the anonymization logic, then checks the file extension of the uploaded file to determine the appropriate format for the download button. It handles CSV, XLSX, and XLS formats.
