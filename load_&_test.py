from pymongo import MongoClient
import numpy as np
from tensorflow.keras.models import load_model
import json

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["cicids2017_db"]
seq_collection = db["preprocessed_sequences"]

# Load vocabulary from JSON
with open("vocab.json", "r") as f:
    vocab_data = json.load(f)
label_to_index = vocab_data["label_to_index"]
index_to_label = {int(k): v for k, v in vocab_data["index_to_label"].items()}
label_vocab = sorted(label_to_index.keys())
print("Label Vocabulary:", label_vocab)

# Load trained model
model = load_model("seq2seq_enhanced_model.keras")
sequence_length = 50

# Test prediction with diverse sequence
skip_offset = 2750000 
test_doc = seq_collection.find_one({"_id": seq_collection.find()[skip_offset]["_id"]})
test_labels = test_doc["X"]["labels"]
test_num = np.array(test_doc["X"]["flow_durations"] + test_doc["X"]["fwd_packets"] + 
                    test_doc["X"]["bwd_packets"] + test_doc["X"]["dest_ports"]).reshape(1, sequence_length-1, 4)
test_input_labels = np.array([test_labels])
decoder_input = test_input_labels
prediction = model.predict([test_input_labels, test_num, decoder_input])

predicted_seq = [index_to_label[np.argmax(token)] for token in prediction[0]]
print("Input sequence:\n", [index_to_label[l] for l in test_labels])
print("Predicted next sequence:\n", predicted_seq)
print("Actual next sequence:\n", [index_to_label[l] for l in test_doc["y"]["labels"]])

client.close()
