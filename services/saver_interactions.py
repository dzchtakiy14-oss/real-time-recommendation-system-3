import redis
import time
from joblib import load
from fastapi import HTTPException

from services.encoding_user_item import encoding_user_id
from services.encoding_user_item import encoding_item_id

# =============
# Prepare Redis
# =============
pool = redis.ConnectionPool(host="redis", port=6379, db=0, decode_responses=True)
r = redis.Redis(connection_pool=pool)

# ===================
# Saving Interactions
# ===================
def saving_interactions(user_id, item_id, event_type):
    try:
        # === Prepare Configs ===
        curr_time = time.time()
        user_idx = encoding_user_id(user_id)
        item_idx = encoding_item_id(item_id)
        key_interacted_items = f"saver_interaction:{user_idx}:interacted_items"
        key_event_type = f"saver_interaction:{user_idx}:event_type"
        key_timestamps = f"saver_interaction:{user_idx}:timestamp"

        pipe = r.pipeline()
        # === Save Common Items ===
        if item_idx == f"{item_id}-unknown":
            return {"msg": "Unknown-Item"}
    
        key_common_items = "common_items"
        pipe.zincrby(key_common_items, 1, item_idx)
        pipe.zremrangebyrank(key_common_items, 0, -255)

        # === Save Interaction === 
        item_idx = int(item_idx) or -1
        pipe.lpush(key_interacted_items, item_idx)
        pipe.ltrim(key_interacted_items, 0, 200)

        event = event_type or "click"
        pipe.lpush(key_event_type, event)
        pipe.ltrim(key_event_type, 0, 200)

        pipe.lpush(key_timestamps, int(curr_time))
        pipe.ltrim(key_timestamps, 0, 200)
        pipe.execute()

        return {"msg": "success"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to Save Interactions: {e}")
