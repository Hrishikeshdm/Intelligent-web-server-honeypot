from pymongo import MongoClient
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Concatenate
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import gc
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

# Step 1: Define enhanced model
def build_model(sequence_length, label_vocab_size):
    label_input = Input(shape=(sequence_length-1,))
    label_embed = Embedding(label_vocab_size, 10)(label_input)
    num_input = Input(shape=(sequence_length-1, 4))
    combined_input = Concatenate()([label_embed, num_input])
    encoder_lstm = LSTM(128, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(combined_input)
    encoder_states = [state_h, state_c]
    decoder_input = Input(shape=(sequence_length-1,))
    decoder_embed = Embedding(label_vocab_size, 10)(decoder_input)
    decoder_lstm = LSTM(128, return_sequences=True)
    decoder_outputs = decoder_lstm(decoder_embed, initial_state=encoder_states)
    decoder_dense = Dense(label_vocab_size, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)
    return Model([label_input, num_input, decoder_input], decoder_outputs)

sequence_length = 50
model = build_model(sequence_length, len(label_vocab))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Step 2: Train model in batches from MongoDB
num_sequences = seq_collection.count_documents({})
print(f"Total sequences to train: {num_sequences}")
batch_size_fetch = 100000
num_batches = (num_sequences + batch_size_fetch - 1) // batch_size_fetch
batch_size_train = 32
epochs = 5 

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        skip = batch_idx * batch_size_fetch
        limit = min(batch_size_fetch, num_sequences - skip)
        cursor = seq_collection.find().skip(skip).limit(limit)
        
        X_labels_list = []
        X_num_list = []
        y_labels_list = []
        for seq in cursor:
            X_labels_list.append(seq["X"]["labels"])
            X_num_list.append(seq["X"]["flow_durations"] + seq["X"]["fwd_packets"] + 
                              seq["X"]["bwd_packets"] + seq["X"]["dest_ports"])
            y_labels_list.append(seq["y"]["labels"])
        
        X_labels = np.array(X_labels_list)
        X_num = np.array(X_num_list).reshape(-1, sequence_length-1, 4)
        y_labels = to_categorical(y_labels_list, num_classes=len(label_vocab))
        
        num_batches_train = (limit + batch_size_train - 1) // batch_size_train
        for i in range(num_batches_train):
            start_idx = i * batch_size_train
            end_idx = min((i + 1) * batch_size_train, limit)
            batch_X_labels = X_labels[start_idx:end_idx]
            batch_X_num = X_num[start_idx:end_idx]
            batch_y_labels = y_labels[start_idx:end_idx]
            batch_decoder_input = batch_X_labels
            model.train_on_batch([batch_X_labels, batch_X_num, batch_decoder_input], batch_y_labels)
        
        del X_labels, X_num, y_labels, X_labels_list, X_num_list, y_labels_list
        gc.collect()

model.save("seq2seq_enhanced_model.keras")

# Step 3: Test prediction
test_doc = seq_collection.find_one({"_id": seq_collection.find()[1000000]["_id"]})
test_labels = test_doc["X"]["labels"]
test_num = np.array(test_doc["X"]["flow_durations"] + test_doc["X"]["fwd_packets"] + 
                    test_doc["X"]["bwd_packets"] + test_doc["X"]["dest_ports"]).reshape(1, sequence_length-1, 4)
test_input_labels = np.array([test_labels])
decoder_input = test_input_labels
prediction = model.predict([test_input_labels, test_num, decoder_input])

predicted_seq = [index_to_label[np.argmax(token)] for token in prediction[0]]
print("Input sequence:", [index_to_label[l] for l in test_labels])
print("Predicted next sequence:", predicted_seq)
print("Actual next sequence:", [index_to_label[l] for l in test_doc["y"]["labels"]])

client.close()