import xml.etree.ElementTree as ET
import json
import pandas as pd
from sklearn.model_selection import train_test_split

def parse_xml_data(xml_file_path):
    """Parse the XML data into a structured format."""
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    cases = []
    for case in root.findall('.//case'):
        case_id = case.get('id')
        
        # Extract patient narrative and questions
        patient_narrative = case.find('patient_narrative').text.strip()
        
        patient_question_elem = case.find('.//patient_question/phrase')
        patient_question = patient_question_elem.text.strip() if patient_question_elem is not None else ""
        
        clinician_question = case.find('clinician_question').text.strip()
        
        # Extract note excerpts and sentences
        note_excerpt = case.find('note_excerpt').text.strip()
        
        sentences = []
        for sentence in case.findall('.//note_excerpt_sentences/sentence'):
            sentence_id = sentence.get('id')
            sentence_text = sentence.text.strip()
            sentences.append({
                'sentence_id': sentence_id,
                'text': sentence_text
            })
        
        cases.append({
            'case_id': case_id,
            'patient_narrative': patient_narrative,
            'patient_question': patient_question,
            'clinician_question': clinician_question,
            'note_excerpt': note_excerpt,
            'sentences': sentences
        })
    
    return cases

def load_key_file(key_file_path):
    """Load the key file containing sentence relevance."""
    with open(key_file_path, 'r') as f:
        keys = json.load(f)
    return keys

def merge_data(cases, keys):
    """Merge case data with relevance information."""
    case_key_map = {item['case_id']: item['answers'] for item in keys}
    
    for case in cases:
        case_id = case['case_id']
        if case_id in case_key_map:
            relevance_map = {item['sentence_id']: item['relevance'] for item in case_key_map[case_id]}
            
            for sentence in case['sentences']:
                sentence_id = sentence['sentence_id']
                if sentence_id in relevance_map:
                    sentence['relevance'] = relevance_map[sentence_id]
                else:
                    sentence['relevance'] = "unknown"
    
    return cases

# Example usage
cases = parse_xml_data('dataset/dev/archehr-qa.xml')
keys = load_key_file('dataset/dev/archehr-qa_key.json')
merged_data = merge_data(cases, keys)

# Convert to dataframe for easier manipulation
sentences_data = []
for case in merged_data:
    for sentence in case['sentences']:
        sentences_data.append({
            'case_id': case['case_id'],
            'clinician_question': case['clinician_question'],
            'sentence_id': sentence['sentence_id'],
            'sentence_text': sentence['text'],
            'relevance': sentence.get('relevance', 'unknown')
        })

df = pd.DataFrame(sentences_data)

# Split data for training
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)