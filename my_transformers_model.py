import torch
if torch.cuda.is_available():
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU.")

import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset

# Load the dataset from the CSV file
data = pd.read_csv('synthetic_sentiment_dataset.csv')

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the dataset
inputs = tokenizer(data['text'].tolist(), padding=True, truncation=True, return_tensors="pt")

# Convert the dataset to a Hugging Face Dataset
dataset = Dataset.from_dict({
    'input_ids': inputs['input_ids'],
    'attention_mask': inputs['attention_mask'],
    'labels': torch.tensor(data['label'].tolist())
})

# Load pre-trained BERT for Sequence Classification (binary classification)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir="./logs",            # directory for storing logs
    logging_steps=10,                # log every 10 steps
)

# Define Trainer
trainer = Trainer(
    model=model,                         # the model to train
    args=training_args,                  # training arguments
    train_dataset=dataset,               # training dataset
    eval_dataset=dataset                  # evaluation dataset (can use a separate dataset)
)

# Train the model
trainer.train()

# Evaluate the model (optional)
results = trainer.evaluate()
print(f"Evaluation Results: {results}")
