import time
import torch
import redis
from joblib import load
from fastapi import HTTPException

from services.encoding_user_item import encoding_user_id
from services.vectors.old_user_vector import retrieve_old_user_vec
from services.vectors.interactions import compute_interacted_items_vec
from services.vectors.context_vector import compute_context_vec
from services.strategies.cold_start import retrieve_common_items
from services.strategies.recommendation import providing_recommendation 
from model.model_structure import TwoTowerModel


# ==============
# Prepare Device
# ==============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============
# Prepare Redis
# =============
pool = redis.ConnectionPool(host="redis", port=6379, db=0, decode_responses=True)
r = redis.Redis(connection_pool=pool)


# =====================
# Recommendation Engine
# =====================
def recommendation_engine(user_id, k: int = 10):
    try:
        # === Config ===
        user_idx = encoding_user_id(user_id)
        curr_time = time.time()
        vectors = []
        weights = []

        # ====== Update User Vector =======
        # === Retrieve Old User Vector ===
        st_1 = time.perf_counter()
        old_user_vec_tens = retrieve_old_user_vec(user_idx)
        et_1 = time.perf_counter()
        print(f"old_user_vec_tens: {et_1 - st_1}")
        if old_user_vec_tens is not None:
            vectors.append(old_user_vec_tens)
            weights.append(0.1)

        # === Retrieve Base Vector === 
        st_1 = time.perf_counter()
        base_user_vec = None
        et_1 = time.perf_counter()
        print(f"base_user_vec: {et_1 - st_1}")
        if base_user_vec is not None:
            vectors.append(base_user_vec)
            weights.append(0.1)
    
        # === Compute Context Vec ===
        st_1 = time.perf_counter()
        context_vec_tens = compute_context_vec(curr_time)
        et_1 = time.perf_counter()
        print(f"context_vec_tens: {et_1 - st_1}")
        if context_vec_tens is not None:
            vectors.append(context_vec_tens)
            weights.append(0.3)

        # === Compute Interacted Items Vec ===
        st_1 = time.perf_counter()
        interacted_items_vec_tens = compute_interacted_items_vec(user_idx)
        et_1 = time.perf_counter()
        print(f"interacted_items_vec_tens: {et_1 - st_1}")
        if interacted_items_vec_tens is not None:
            vectors.append(interacted_items_vec_tens)
            weights.append(0.5)

        # === Recommend Popular Items ===
        if interacted_items_vec_tens is None and old_user_vec_tens is None:
            popular_items = retrieve_common_items(user_idx, k)
            return popular_items
        
        # === Integrate Vectors with Weights ===
        weights_tens = torch.tensor(weights, dtype=torch.float32, device=device).unsqueeze(1)
        vectors_tens = torch.tensor(vectors).to(device)

        weighted_vectors = vectors_tens * weights_tens

        # === Create New User Vec ===
        with torch.no_grad():
            new_user_vec = torch.sum(weighted_vectors, dim=0).cpu().numpy()
            print(f"new_user_vec: {new_user_vec.shape}")

        # === Storing New User Vec === 
        key = f"old_user_vec:{user_idx}"
        r.set(key, new_user_vec.tobytes())

        # === Providing Recommendations ===
        st_1 = time.perf_counter()
        recommendations = providing_recommendation(user_idx, new_user_vec, k)
        et_1 = time.perf_counter()
        print(f"recommendations: {et_1 - st_1}")
    
        return recommendations
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to Provide Recommendations: {e}")
