import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import re

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=49)  # Adjust num_labels as needed
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load label classes
label_classes = np.load('label_classes.npy', allow_pickle=True)

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Streamlit app
st.title("Text Classification")

user_input = st.text_input("Enter text to classify:")

if user_input:
    # Clean and tokenize input
    user_input_cleaned = clean_text(user_input)
    user_input_tokenized = tokenizer(user_input_cleaned, padding='max_length', truncation=True, return_tensors='pt')

    # Make prediction
    with torch.no_grad():
        outputs = model(**user_input_tokenized)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()

    # Decode prediction
    predicted_class = label_classes[predicted_class_id]
    st.write(f"Predicted Class: {predicted_class}")

