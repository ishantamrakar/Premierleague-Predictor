import torch
import json
import pandas as pd
from models import MatchPredictor
from data_loader import PremierLeagueDataLoader

def predict_match(home_team_name, away_team_name, data_loader, model, team_to_id):
    """
    Predicts the outcome of a single, hypothetical future match by running the full
    preprocessing pipeline on the historical data plus the new match.
    """
    # 1. Create a hypothetical future match row
    future_date = pd.to_datetime('2099-01-01')
    
    # Try to infer the 'season' for the new match, or default to the latest known
    try:
        latest_season = sorted(data_loader.raw_data['Season'].unique())[-1]
    except (AttributeError, IndexError):
        latest_season = "2025/26" # A sensible default if raw_data isn't loaded

    new_match_df = pd.DataFrame([{
        'HomeTeam': home_team_name,
        'AwayTeam': away_team_name,
        'MatchDate': future_date,
        'Season': latest_season
    }])
    
    # 2. Append to the raw data and re-preprocess
    # This is not performant but guarantees feature consistency
    if data_loader.raw_data is not None:
        data_loader.raw_data = pd.concat([data_loader.raw_data, new_match_df], ignore_index=True)
    else:
        print("Error: Raw data not loaded in data_loader.")
        return
        
    print("Calculating features for the new match...")
    data_loader.preprocess_data()
    print("Feature calculation complete.")
    
    # 3. Extract the features for our new match (the last row)
    match_features = data_loader.processed_data.iloc[-1]
    
    feature_cols = [
        'HomeShots', 'AwayShots', 'HomeShotsOnTarget', 'AwayShotsOnTarget', 'HomeCorners', 'AwayCorners',
        'HomeFouls', 'AwayFouls', 'HomeYellowCards', 'AwayYellowCards', 'HomeRedCards', 'AwayRedCards',
        'HomeTeam_Form', 'AwayTeam_Form', 'Match_of_the_Season', 'H2H_HomeWins_Pct', 'H2H_Draws_Pct',
        'H2H_AwayWins_Pct', 'H2H_Home_AvgGoals', 'H2H_Away_AvgGoals', 'Home_Attack_Strength',
        'Home_Defense_Strength', 'Away_Attack_Strength', 'Away_Defense_Strength', 'Home_Weighted_Form',
        'Away_Weighted_Form'
    ]
    
    # Fill any potential NaNs in the feature vector with 0
    features_tensor = torch.tensor(match_features[feature_cols].fillna(0).values, dtype=torch.float32)
    
    # 4. Get team IDs
    home_id = team_to_id.get(home_team_name)
    away_id = team_to_id.get(away_team_name)
    
    if home_id is None or away_id is None:
        return f"Error: One of the teams not found in team mapping: {home_team_name} or {away_team_name}"

    home_id_tensor = torch.tensor([home_id], dtype=torch.long)
    away_id_tensor = torch.tensor([away_id], dtype=torch.long)
    
    # 5. Make the prediction
    model.eval()
    with torch.no_grad():
        output = model(features_tensor.unsqueeze(0), home_id_tensor, away_id_tensor)
        probs = torch.softmax(output, dim=1).squeeze()
        prediction_idx = torch.argmax(probs).item()
        
        idx_to_outcome = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
        
        return {
            "prediction": idx_to_outcome[prediction_idx],
            "probabilities": {
                "Home Win": f"{probs[2]:.2%}",
                "Draw": f"{probs[1]:.2%}",
                "Away Win": f"{probs[0]:.2%}"
            }
        }

if __name__ == '__main__':
    MODEL_PATH = "saved_model/best_model.pth"
    TEAM_MAP_PATH = "saved_model/team_to_id.json"

    print("Loading model and team mapping...")
    with open(TEAM_MAP_PATH, 'r') as f:
        team_to_id = json.load(f)
    
    # Infer model parameters from the saved state_dict
    state_dict = torch.load(MODEL_PATH)
    EMBEDDING_DIM = 10 # This must match the model training
    input_dim_from_model = state_dict['fc1.weight'].shape[1] - (2 * EMBEDDING_DIM)
    num_teams_from_map = len(team_to_id)

    model = MatchPredictor(input_dim=input_dim_from_model, num_teams=num_teams_from_map, embedding_dim=EMBEDDING_DIM)
    model.load_state_dict(state_dict)
    print("Model loaded successfully.")

    # Instantiate and load data into the data loader
    # This is necessary to access the feature engineering pipeline
    data_loader = PremierLeagueDataLoader(data_path='..')
    data_loader.load_data()

    if data_loader.raw_data is not None:
        print("\n--- Live Prediction Example ---")
        home_team = "Arsenal"
        away_team = "Man United"
        
        result = predict_match(home_team, away_team, data_loader, model, team_to_id)
        
        print(f"Prediction for {home_team} vs. {away_team}:")
        print(result)
