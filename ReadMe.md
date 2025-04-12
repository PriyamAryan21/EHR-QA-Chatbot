
# Electronic Health Record (EHR) Question Answering System

## Project Overview

The **Electronic Health Record (EHR) Question Answering System** assists healthcare professionals by providing automated answers to clinical queries based on patient records. It uses NLP techniques to extract relevant data from clinical notes and generate answers grounded in that information.

## Features

- **Clinical Query Answering**: Answers a wide range of clinical questions related to diagnoses, treatments, lab results, etc.
- **Context-Aware**: Provides answers based on the most relevant information from the patient’s medical record.
- **Interactive Interface**: A Gradio-powered interface allows users to input questions and clinical data for quick responses.

## Technologies Used

- **Python**
- **Gradio** for UI
- **Transformers (Hugging Face)** for NLP models (e.g., BERT, T5)
- **Torch** for deep learning

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/ehr-qa-system.git
   cd ehr-qa-system
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download necessary pre-trained models from Hugging Face (if not included).

## Model Training Instructions

To train the models and set up the system, follow these steps:

### 1. Run Preprocessing

```bash
python preprocessing.py
```

### 2. Train the Relevance Classifier

```bash
python relevanceClassifier.py
```

### 3. Train the Answer Generation Model

```bash
python answerGenModel.py
```

### 4. Run the Chatbot

```bash
python chatBot.py
```

This will launch a Gradio interface for interacting with the model.

## Usage

After running the chatbot, enter clinical questions and clinical notes in the Gradio interface and click **Generate Answer**.

### Example Questions:

- What are the likely causes of the patient's shortness of breath?
- Does the patient show signs of fluid overload?
- What is the probable diagnosis?

## Model Details

1. **Relevance Classification Model**: Identifies relevant sentences from clinical notes.
2. **QA Model**: Generates answers using the relevant sentences from the clinical notes.
3. **Gradio Interface**: Provides an easy-to-use interface for real-time interaction.

## Example Output

- **Question**: "What is the probable diagnosis?"
- **Clinical Note**: "The patient is a 62-year-old male presenting with shortness of breath..."

- **Answer**: "The probable diagnosis is congestive heart failure."

## Contribution

Contributions are welcome! Please open issues or submit pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to Hugging Face for transformers and pre-trained models.
- Gradio for the interface.
