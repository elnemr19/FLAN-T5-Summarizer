import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import streamlit as st

# Path to your fine-tuned model folder
model_path = "flan-t5-summarizer"

# Function to load and initialize the model
def load_model():
    # Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    
    # Set device to CPU (or CUDA if available)
    device = torch.device("cpu")  # Change to 'cuda' if you want to use GPU
    
    # Now move the model to the device
    model = model.to(device)
    
    return model, tokenizer, device

# Initialize the model and tokenizer once for each interaction
model, tokenizer, device = load_model()

# Streamlit interface setup
st.title("FLAN-T5 Summarizer")
st.write("Enter text for summarization:")

# Input text from the user
input_text = st.text_area("Text to summarize", height=200)

if st.button("Summarize"):
    if input_text:
        # Add task prefix
        input_text_with_prefix = "summarize: " + input_text

        # Tokenize and move inputs to the correct device
        inputs = tokenizer(input_text_with_prefix, return_tensors="pt", truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate the summary
        summary_ids = model.generate(
            **inputs,
            max_length=100,
            num_beams=4,
            early_stopping=True
        )

        # Decode and display the summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.write("Please enter some text to summarize.")
