import pickle

# --- Save generated records to disk ---
with open("embedded_records.pkl", "wb") as f:
    pickle.dump(records, f)