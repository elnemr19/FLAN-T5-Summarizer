# 📚 FLAN-T5 Summarization on CNN/DailyMail

This project fine-tunes [Google's FLAN-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5) model on the [CNN/DailyMail dataset](https://huggingface.co/datasets/abisee/cnn_dailymail) for abstractive text summarization. The model is trained using the Hugging Face `transformers` library and evaluated using ROUGE metrics.


## 🔍 Project Overview

- **Model:** `google/flan-t5-base`

- **Dataset:** CNN/DailyMail (via Hugging Face Datasets)

- **Task:** Text Summarization

- **Frameworks:** PyTorch, Hugging Face Transformers & Datasets

- **Evaluation Metric:** ROUGE (ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum)


## 📈 Results
After 5 epochs of training on a subset of 20,000 training samples and 2,000 samples for both validation and testing:

✅ **Validation Set Performance:**

- **ROUGE-1:** `24.86`

- **ROUGE-2:** `11.48`

- **ROUGE-L:** `20.18`

- **ROUGE-Lsum:** `23.28`

🧪 **Test Set Performance:**

- **ROUGE-1:** `25.33`

- **ROUGE-2:** `11.96`

- **ROUGE-L:** `20.68`

- **ROUGE-Lsum:** `23.81`


## 🧠 Training Details

- **Epochs:** 5

- **Learning Rate:** 3e-4

- **Batch Size:** 4 (Train), 2 (Eval)

- **Weight Decay:** 0.01

- **Evaluation Strategy:** Every 500 steps

- **Tokenizer:** T5Tokenizer

- **Data Collator:** `DataCollatorForSeq2Seq`

- **Model Saving Path:** `./flan-t5-summarizer`

## 🛠️ Setup Instructions

**1. Install Dependencies**

```bash
pip install nltk datasets transformers[torch] tokenizers evaluate rouge_score sentencepiece huggingface_hub -q

```


**2. Download NLTK Tokenizer**

```python
import nltk
nltk.download("punkt")

```


## 🧪 Evaluate the Model

To evaluate the model on a test dataset:

```python
trainer.evaluate(tokenized_dataset["test"])
```


## 💾 Model Saving

The model and tokenizer are saved locally after training:

```bash
./flan-t5-summarizer

```

You can reload it using:

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained("./flan-t5-summarizer")
tokenizer = T5Tokenizer.from_pretrained("./flan-t5-summarizer")
```
## You can check my model from [Hugging Face](https://huggingface.co/)

[flan-t5-summarizer](https://huggingface.co/AbdullahAlnemr1/flan-t5-summarizer)


