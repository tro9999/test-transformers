from transformers import pipeline
model = "tro9999/my_test_big_ner_model"
ner_pipeline = pipeline("ner", model=model)

def get_labels(text):
    #text = "how did i sleep last night?"
    entities = ner_pipeline(text)
    #print(result)

    # Given list of dictionaries
    """
    entities = [{'entity': 'B-SCONJ', 'score': 0.5785049, 'index': 1, 'word': 'how', 'start': 0, 'end': 3},
                {'entity': 'B-PRON', 'score': 0.99429685, 'index': 3, 'word': 'i', 'start': 8, 'end': 9},
                {'entity': 'B-VERB', 'score': 0.9882465, 'index': 4, 'word': 'sleep', 'start': 10, 'end': 15},
                {'entity': 'B-TIME', 'score': 0.95321864, 'index': 5, 'word': 'last', 'start': 16, 'end': 20},
                {'entity': 'I-TIME', 'score': 0.97035754, 'index': 6, 'word': 'night', 'start': 21, 'end': 26}]
    """
    
    # Initialize variables
    current_entity = None
    current_word = ""
    entity_labels = []

    # Loop through the list of dictionaries
    for entity in entities:
        # Extract the entity label and word
        entity_label = entity['entity'][2:]  # Remove the "B-" or "I-" prefix
        word = entity['word']

        # Check if the current word belongs to the same entity as the previous word
        if entity_label == current_entity:
            current_word += " " + word  # Concatenate the word
        else:
            # If a new entity is encountered, add the previous entity and word to the list
            if current_entity:
                entity_labels.append((current_entity, current_word))
            current_entity = entity_label
            current_word = word

    # Add the last entity and word to the list
    if current_entity:
        entity_labels.append((current_entity, current_word))

    # Print the entity labels
    #for entity_label, word in entity_labels:
    #    print(entity_label, ":", word)

    return entity_labels