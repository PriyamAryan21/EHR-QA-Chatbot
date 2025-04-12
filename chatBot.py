import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import json

# Load the tokenizers and models
relevance_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
relevance_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

qa_tokenizer = AutoTokenizer.from_pretrained("t5-small")
qa_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

def predict_answer(question, clinical_note):
    """Generate an answer based on the question and clinical note."""
    # Tokenize and prepare the sentences for relevance classification
    sentences = [s.strip() for s in clinical_note.split('.') if s.strip()]
    
    # Classify each sentence for relevance
    relevant_sentences = []
    for sentence in sentences:
        # Combine question and sentence for the relevance model
        inputs = relevance_tokenizer(f"Question: {question} Sentence: {sentence}", 
                                     return_tensors="pt", truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = relevance_model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()
        
        # If predicted as relevant, add to relevant sentences
        if prediction == 1:
            relevant_sentences.append(sentence)
    
    # Generate the answer using the seq2seq model
    context = " ".join(relevant_sentences)
    input_text = f"Context: {context} Question: {question}"
    
    inputs = qa_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=1024)
    
    with torch.no_grad():
        outputs = qa_model.generate(
            inputs["input_ids"],
            max_length=150,
            num_beams=4,
            early_stopping=True
        )
    
    # Decode and return the answer
    answer = qa_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Format for submission
    result = {
        "answer": answer
    }
    
    return answer, json.dumps(result, indent=4)

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Electronic Health Record Question Answering")
    
    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(label="Question", placeholder="Enter your clinical question here...")
            note_input = gr.Textbox(label="Clinical Note", placeholder="Paste the clinical note here...", lines=10)
            submit_btn = gr.Button("Generate Answer")
        
        with gr.Column():
            answer_output = gr.Textbox(label="Answer")
            json_output = gr.JSON(label="JSON Output (for submission)")

    submit_btn.click(
        fn=predict_answer,
        inputs=[question_input, note_input],
        outputs=[answer_output, json_output]
    )

# Launch the interface
demo.launch()
