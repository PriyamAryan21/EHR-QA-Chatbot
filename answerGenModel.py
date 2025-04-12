from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from preprocessing import merged_data 
from sklearn.model_selection import train_test_split
from datasets import Dataset
# Use a medical language model with generation capabilities
model_name = "google/flan-t5-base"  # Can also use clinical-specific models
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def prepare_qa_data(merged_data):
    """Prepare question-answer pairs for training."""
    qa_pairs = []
    
    for case in merged_data:
        # Get question
        question = case['clinician_question']
        
        # Get relevant sentences for the answer
        relevant_sentences = []
        for sentence in case['sentences']:
            if sentence.get('relevance') == 'essential':
                relevant_sentences.append(sentence['text'])
        
        # Create answer by joining relevant sentences
        answer = " ".join(relevant_sentences)
        
        # Include the clinical notes context
        context = case['note_excerpt']
        
        qa_pairs.append({
            'case_id': case['case_id'],
            'question': question,
            'context': context,
            'answer': answer
        })
    
    return qa_pairs

qa_data = prepare_qa_data(merged_data)

# Split into train/val
train_qa, val_qa = train_test_split(qa_data, test_size=0.2, random_state=42)

def preprocess_qa_data(qa_pairs):
    """Prepare QA data for seq2seq model."""
    inputs = []
    targets = []
    
    for pair in qa_pairs:
        # Format input with context and question
        input_text = f"Context: {pair['context']} Question: {pair['question']}"
        inputs.append(input_text)
        targets.append(pair['answer'])
    
    # Tokenize
    input_encodings = tokenizer(inputs, truncation=True, padding=True, max_length=1024)
    target_encodings = tokenizer(targets, truncation=True, padding=True, max_length=512)
    
    # Create dataset
    dataset = Dataset.from_dict({
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': target_encodings['input_ids']
    })
    
    return dataset

# Prepare datasets
train_qa_dataset = preprocess_qa_data(train_qa)
val_qa_dataset = preprocess_qa_data(val_qa)

# Set up training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir='./results_qa',
    num_train_epochs=4,
    per_device_train_batch_size=1,           # Smaller batch size to prevent memory errors
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,           # Simulates 4 effective batch size
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs_qa',
    predict_with_generate=True,
    save_strategy="epoch",
    generation_max_length=150,
    fp16=True,
)

# Initialize trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_qa_dataset,
    eval_dataset=val_qa_dataset,
)

# Train the model
trainer.train()