import time 
import redis 
import torch
import numpy as np
from joblib import load
from fastapi import HTTPException

# ==============
# Prepare Device
# ==============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============
# Prepare Redis
# =============
pool = redis.ConnectionPool(host="redis", port=6379, db=0, decode_responses=True)
r = redis.Redis(connection_pool=pool)

# ==========
# Load Tools
# ==========
event_weights = load("storage/store/event_weights.pkl")
mapping_item_idx_to_vec = load("storage/store/item_idx_to_vec.pkl")

# ============================
# Compute Interacted Items Vec
# ============================
def compute_interacted_items_vec(user_idx, decay_rate: float = 0.2):
    try:
        # === Prepare Configs ===
        key_interacted_items = f"saver_interaction:{user_idx}:interacted_items"
        key_event_type = f"saver_interaction:{user_idx}:event_type"
        key_timestamps = f"saver_interaction:{user_idx}:timestamp"

        # === Retrieve Values ===
        pipe = r.pipeline()
        pipe.lrange(key_interacted_items, 0, 9)
        pipe.lrange(key_event_type, 0, 9)
        pipe.lrange(key_timestamps, 0, 9)

        interacted_items, event_types, timestamps = pipe.execute()
    
        if not interacted_items:
            return None 
        print(f"interacted_items: {interacted_items}")
    
        # === Convert Values to "int"
        interacted_items_int = [int(i) for i in interacted_items]
        timestamps_int = [int(t) for t in timestamps]

        # === Validate Missing Values ===
        if not len(interacted_items_int) == len(timestamps_int) == len(event_types):
            pipe = r.pipeline()
            pipe.expire(key_interacted_items, 0)
            pipe.expire(key_event_type, 0)
            pipe.expire(key_timestamps, 0)
            pipe.execute()
            return None

        # === Compute Weights ===
        # Setting Event Weights
        weights_of_events = [event_weights.get(e, 0.1) for e in event_types]

        # Prepare Weights
        weights = [] 
        vectors = []
        curr_time = time.time()

        for event_weight, timestamp, item in zip(weights_of_events, timestamps_int, interacted_items_int):
            # === Retrieve Item Vec ===
            item_vec = mapping_item_idx_to_vec.get(item, None)
            if item_vec is None:
                continue 
            vectors.append(item_vec)

            # == Compute Time Decay ==
            duration = curr_time - timestamp
            duration_per_day = duration / (3600 * 24)
            time_weight = np.exp(-decay_rate * duration_per_day)

            # == Combining Weights ==
            combined_weights = float(event_weight) * time_weight
            weights.append(combined_weights)

        if not weights or not vectors:
            return None

        # === Convert to Tensors ===
        weights_tens = torch.tensor(weights, dtype=torch.float32, device=device).unsqueeze(1)
        print(f"weights_tens: {weights_tens.shape}")
        items_vecs_tens = torch.tensor(vectors, dtype=torch.float32, device=device)
        print(f"items_vecs_tens: {items_vecs_tens.shape}")

        # === Compute Weighted Vector ===
        weighted_vecs = items_vecs_tens * weights_tens
        print(f"weighted_vecs: {weighted_vecs.shape}")

        final_item_vec = torch.sum(weighted_vecs, dim=0).cpu().numpy().tolist()

        return final_item_vec
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to Compute Interacted Items Vector: {e}")
