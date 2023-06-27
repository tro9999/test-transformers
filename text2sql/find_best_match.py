import tensorflow as tf
from transformers import TFDistilBertForQuestionAnswering, DistilBertTokenizer

model = TFDistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")

def find_best_match(question, text_list, model, tokenizer):
    best_match = None
    best_score = float('-inf')
    best_match_index = -1

    # Encode the question
    encoded_question = tokenizer.encode_plus(question, padding=True, truncation=True, return_tensors="tf")

    # Iterate over the text list
    for i, text in enumerate(text_list):
        # Encode the text
        encoded_text = tokenizer.encode_plus(text, padding=True, truncation=True, return_tensors="tf")

        # Forward pass through the model
        outputs = model(encoded_question['input_ids'], encoded_text['input_ids'])

        # Get the start and end logits from the model's output
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        # Compute the score using the softmax probabilities
        start_probs = tf.nn.softmax(start_logits, axis=1).numpy()[0]
        end_probs = tf.nn.softmax(end_logits, axis=1).numpy()[0]
        score = tf.reduce_max(start_probs) + tf.reduce_max(end_probs)

        # Check if the current text has a higher score
        if score > best_score:
            best_score = score
            best_match = text
            best_match_index = i

    return best_match, best_score, best_match_index

