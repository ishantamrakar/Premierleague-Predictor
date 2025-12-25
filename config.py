"""
Configuration for the Premier League Predictor App

To add a new model:
1. Create a folder in models/ (e.g., models/your_model_v2/)
2. Add: saved_model/best_model.pth, saved_model/team_to_id.json, architecture.py, features.csv
3. Update ACTIVE_MODEL below to point to your new model
"""

# Which model to use (just change this to switch models!)
ACTIVE_MODEL = "win_draw_loss_v1"

# Model paths (automatically constructed)
MODEL_DIR = f"models/{ACTIVE_MODEL}"
MODEL_PATH = f"{MODEL_DIR}/saved_model/best_model.pth"
TEAM_MAPPING_PATH = f"{MODEL_DIR}/saved_model/team_to_id.json"
FEATURES_PATH = f"{MODEL_DIR}/features.csv"
ARCHITECTURE_MODULE = f"models.{ACTIVE_MODEL}.architecture"

# Model metadata
MODEL_INFO = {
    "win_draw_loss_v1": {
        "name": "Win-Draw-Loss Neural Network",
        "version": "1.0",
        "accuracy": "55-60%",
        "description": "Neural network with team embeddings and 26 engineered features",
        "embedding_dim": 10,
        "num_classes": 3,
    },
    # Add future models here
    # "better_model_v2": {
    #     "name": "Improved Model",
    #     "version": "2.0",
    #     "accuracy": "65%",
    #     ...
    # }
}

def get_model_info():
    """Get info about the currently active model"""
    return MODEL_INFO.get(ACTIVE_MODEL, {})
