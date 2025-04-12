from preprocessing import train_df, val_df
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Use BioBERT or ClinicalBERT for better performance on medical text
model_name = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

def preprocess_data(df):
    """Prepare data for transformer model."""
    texts = []
    labels = []
    
    for _, row in df.iterrows():
        question = row['clinician_question']
        sentence = row['sentence_text']
        # Combine question and sentence for classification
        text = f"Question: {question} Sentence: {sentence}"
        texts.append(text)
        
        # Convert relevance to binary label
        label = 1 if row['relevance'] == 'essential' else 0
        labels.append(label)
    
    # Tokenize
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
    
    # Create dataset
    dataset = Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': labels
    })
    
    return dataset

# Prepare datasets
train_dataset = preprocess_data(train_df)
val_dataset = preprocess_data(val_df)

# Define metrics computation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    # Use these parameters for older versions
    eval_steps=500,
    save_steps=500,
    # Remove parameters that aren't supported
    # No evaluation_strategy or save_strategy
    # No load_best_model_at_end
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()