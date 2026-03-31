import random
import faiss
import redis
from joblib import load
from fastapi import HTTPException
import numpy as np

from services.mmr import mmr_ranker_fast
from services.strategies.cold_start import retrieve_common_items
# ==========
# Load Faiss 
# ==========
index = faiss.read_index("storage/store/faiss_index.bin")

# =============
# Prepare Redis
# =============
pool = redis.ConnectionPool(host="localhost", port=6379, db=0, decode_responses=False)
r = redis.Redis(connection_pool=pool)

# =====================
# Mapping Items Vectors
# =====================
mapping_item_idx_to_vec = load("storage/store/item_idx_to_vec.pkl")

# ============
# Load Mapping
# ============
mapping_item_idx_to_title = load("storage/store/item_idx_to_title.pkl")
mapping_item_idx_to_image = load("storage/store/book_idx_to_images_links.pkl")
mapping_items_idx_to_id = load("storage/store/items_idx_to_id.pkl")

# =========================
# Providing Recommendations 
# =========================
def providing_recommendation(user_idx, new_user_vec, k):
    try:
        # === Extract Similar Items ===
        if new_user_vec.ndim == 1:
            new_user_vec = np.array([new_user_vec])
        s, total_indices = index.search(new_user_vec, k * 4)

        # === Extract Interacted Items === 
        key_interacted_items = f"saver_interaction:{user_idx}:interacted_items"
        key_watched_items = f"saver_interaction:{user_idx}:watched_items"

        # === Retrieve Values ===
        pipe = r.pipeline()
        pipe.lrange(key_interacted_items, 0, -1)
        pipe.lrange(key_watched_items, 0, -1)
        interacted_items, watched_items = pipe.execute()

        # === Filter recommendations ===
        if interacted_items:
            interacted_items_int = [int(i) for i in interacted_items]
        if watched_items:
            interacted_items_int = [int(i) for i in watched_items] + interacted_items_int

        total_indices_int = [int(i) for i in total_indices[0]]
        candidate_idx = list(set(total_indices_int) - set(interacted_items_int))
        if not candidate_idx:
            return retrieve_common_items(user_idx, k)
    
        candidate_vecs = np.array([mapping_item_idx_to_vec[i] for i in candidate_idx], dtype=np.float32)

        # === Re-ranking ===
        print(new_user_vec.shape)
        print(candidate_vecs.shape)
        reranked_items = mmr_ranker_fast(new_user_vec[0], candidate_vecs, candidate_idx, k)
        
        if len(reranked_items) < k:
            missing_num = k - len(reranked_items)
            all_items = set(mapping_items_idx_to_id.keys())
            filtered_items = list(all_items - set(interacted_items_int))
            if filtered_items >= missing_num:
                reranked_items.extend(random.sample(filtered_items, k=missing_num))
            else:
                reranked_items.extend(filtered_items)

        # === Save Recommended Items ===
        pipe = r.pipeline()
        pipe.lpush(key_watched_items, *reranked_items)
        pipe.ltrim(key_watched_items, 0, 60)
        pipe.execute()

        # === Providing Recommendations ===
        recommendations = []
        for idx in reranked_items:
            img_dict = mapping_item_idx_to_image.get(idx, {})
            id = mapping_items_idx_to_id[idx]
            recommendations.append({
                    "item_id": id,
                    "title": mapping_item_idx_to_title.get(idx, "not-found"),
                    "image_s": img_dict.get("image_url_s", "not-found"),
                    "image_m": img_dict.get("image_url_m", "not-found"),
                    "image_l": img_dict.get("image_url_l", "not-found")
                })

        return recommendations
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to Compute Recommendations: {e}")