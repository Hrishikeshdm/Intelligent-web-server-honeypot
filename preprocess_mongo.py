from pymongo import MongoClient
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import gc
import json

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["cicids2017_db"]
seq_collection = db["preprocessed_sequences"]

seq_collection.drop()

csv_folder = "C:/Users/Lenovo/Desktop/IWSH/prog/MachineLearningCVE" 

# Step 1: Load and preprocess CSV files
all_labels = []
all_flow_durations = []
all_fwd_packets = []
all_bwd_packets = []
all_dest_ports = []

print("Loading CSV files...")
for csv_file in os.listdir(csv_folder):
    if csv_file.endswith(".csv"):
        file_path = os.path.join(csv_folder, csv_file)
        print(f"Processing {csv_file}")
        df = pd.read_csv(file_path, encoding="ISO-8859-1", low_memory=False)
        
        df.columns = [col.strip() for col in df.columns]
        
        labels = df["Label"].str.replace("ï¿½", "-").tolist()
        flow_durations = df["Flow Duration"].fillna(0).astype(float).tolist()
        fwd_packets = df["Total Fwd Packets"].fillna(0).astype(float).tolist()
        bwd_packets = df["Total Backward Packets"].fillna(0).astype(float).tolist()
        dest_ports = df["Destination Port"].fillna(0).astype(float).tolist()
        
        all_labels.extend(labels)
        all_flow_durations.extend(flow_durations)
        all_fwd_packets.extend(fwd_packets)
        all_bwd_packets.extend(bwd_packets)
        all_dest_ports.extend(dest_ports)

total_records = len(all_labels)
assert total_records == len(all_flow_durations) == len(all_fwd_packets) == len(all_bwd_packets) == len(all_dest_ports)
print(f"Total records processed: {total_records}")

# Step 2: Vocabulary and encoding
label_vocab = sorted(set(all_labels))
label_to_index = {label: idx for idx, label in enumerate(label_vocab)}
index_to_label = {idx: label for label, idx in label_to_index.items()}
print("Label Vocabulary:", label_vocab)

# Save vocabulary to JSON
with open("vocab.json", "w") as f:
    json.dump({"label_to_index": label_to_index, "index_to_label": index_to_label}, f)

# Step 3: Normalize numerical features
max_flow_duration = max(all_flow_durations)
max_fwd_packets = max(all_fwd_packets)
max_bwd_packets = max(all_bwd_packets)
max_dest_port = max(all_dest_ports)
flow_durations = [fd / max_flow_duration for fd in all_flow_durations]
fwd_packets = [fp / max_fwd_packets for fp in all_fwd_packets]
bwd_packets = [bp / max_bwd_packets for bp in all_bwd_packets]
dest_ports = [dp / max_dest_port for dp in all_dest_ports]

# Step 4: Create and store sequences
sequence_length = 50
sequences = []
for i in tqdm(range(total_records - sequence_length + 1), desc="Creating and storing sequences"):
    seq = {
        "X": { #input features
            "labels": [label_to_index[all_labels[j]] for j in range(i, i + sequence_length - 1)],
            "flow_durations": flow_durations[i:i + sequence_length - 1],
            "fwd_packets": fwd_packets[i:i + sequence_length - 1],
            "bwd_packets": bwd_packets[i:i + sequence_length - 1],
            "dest_ports": dest_ports[i:i + sequence_length - 1]
        },
        "y": { #target lables
            "labels": [label_to_index[all_labels[j]] for j in range(i + 1, i + sequence_length)]
        }
    }
    sequences.append(seq)

batch_size = 10000
for i in tqdm(range(0, len(sequences), batch_size), desc="Inserting into MongoDB"):
    seq_collection.insert_many(sequences[i:i + batch_size])

print(f"Stored {len(sequences)} sequences in MongoDB")
client.close()
