import torch
import json
import pandas as pd
import numpy as np
import argparse
import sys
from models import MatchPredictor

def predict_match(home_team_name, away_team_name, model, team_to_id, features_df):
    """
    Predicts the outcome of a single match using pre-calculated feature vectors.
    """
    # Look up the pre-calculated feature vector for the match
    match_row = features_df[(features_df['HomeTeam'] == home_team_name) & (features_df['AwayTeam'] == away_team_name)]
    
    if match_row.empty:
        valid_teams = sorted(list(team_to_id.keys()))
        return (f"Error: Fixture {home_team_name} vs. {away_team_name} not found in pre-calculated features.\n"
                f"Valid team names are: {valid_teams}")

    match_features = match_row.iloc[0]
    
    feature_cols = [
        'HomeShots', 'AwayShots', 'HomeShotsOnTarget', 'AwayShotsOnTarget', 'HomeCorners', 'AwayCorners',
        'HomeFouls', 'AwayFouls', 'HomeYellowCards', 'AwayYellowCards', 'HomeRedCards', 'AwayRedCards',
        'HomeTeam_Form', 'AwayTeam_Form', 'Match_of_the_Season', 'H2H_HomeWins_Pct', 'H2H_Draws_Pct',
        'H2H_AwayWins_Pct', 'H2H_Home_AvgGoals', 'H2H_Away_AvgGoals', 'Home_Attack_Strength',
        'Home_Defense_Strength', 'Away_Attack_Strength', 'Away_Defense_Strength', 'Home_Weighted_Form',
        'Away_Weighted_Form'
    ]
    
    features_tensor = torch.tensor(match_features[feature_cols].values.astype(np.float32), dtype=torch.float32)
    
    home_id = team_to_id.get(home_team_name)
    away_id = team_to_id.get(away_team_name)
    
    if home_id is None or away_id is None:
        invalid_team = home_team_name if home_id is None else away_team_name
        valid_teams = sorted(list(team_to_id.keys()))
        return (f"Error: Team '{invalid_team}' not found.\n"
                f"Valid team names are: {valid_teams}")

    home_id_tensor = torch.tensor([home_id], dtype=torch.long)
    away_id_tensor = torch.tensor([away_id], dtype=torch.long)
    
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
    parser = argparse.ArgumentParser(description="Predict the outcome of a Premier League match.")
    parser.add_argument("--home", type=str, help="Name of the home team.")
    parser.add_argument("--away", type=str, help="Name of the away team.")
    args = parser.parse_args()

    MODEL_PATH = "saved_model/best_model.pth"
    TEAM_MAP_PATH = "saved_model/team_to_id.json"
    PRECOMPUTED_FEATURES_PATH = "features_2025-26.csv"

    print("Loading model, team mapping, and pre-computed features...")
    with open(TEAM_MAP_PATH, 'r') as f:
        team_to_id = json.load(f)
    
    precomputed_features_df = pd.read_csv(PRECOMPUTED_FEATURES_PATH)
    
    state_dict = torch.load(MODEL_PATH)
    EMBEDDING_DIM = 10
    input_dim_from_model = state_dict['fc1.weight'].shape[1] - (2 * EMBEDDING_DIM)
    num_teams_from_map = len(team_to_id)
    model = MatchPredictor(input_dim=input_dim_from_model, num_teams=num_teams_from_map, embedding_dim=EMBEDDING_DIM)
    model.load_state_dict(state_dict)
    print("All artifacts loaded successfully.")

    if args.home and args.away:
        print(f"\n--- Predicting: {args.home} vs. {args.away} ---")
        result = predict_match(args.home, args.away, model, team_to_id, precomputed_features_df)
        
        if isinstance(result, dict):
            print(f"Prediction: {result['prediction']}")
            print(f"Probabilities: {result['probabilities']}")
        else:
            print(result) # Print error message
    else:
        print("\n--- Running Demonstration Examples (no --home or --away args provided) ---")
        print("\n--- Prediction Example: Man United vs. Bournemouth ---")
        home_team = "Man United"
        away_team = "Bournemouth"
        
        result = predict_match(home_team, away_team, model, team_to_id, precomputed_features_df)
        
        print(f"Prediction for {home_team} vs. {away_team}:")
        print(result)

        print("\n--- Example with Invalid Team ---")
        home_team_invalid = "Real Madrid"
        away_team_invalid = "Arsenal"
        
        result_invalid = predict_match(home_team_invalid, away_team_invalid, model, team_to_id, precomputed_features_df)
        print(f"Prediction for {home_team_invalid} vs. {away_team_invalid}:")
        print(result_invalid)

        print("\n--- Predicting First 20 Fixtures of 2025-26 Season ---")
        fixtures_df = pd.read_csv("fixtures_2025-26.csv")
        for index, row in fixtures_df.head(20).iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            
            prediction_result = predict_match(home_team, away_team, model, team_to_id, precomputed_features_df)
            
            if isinstance(prediction_result, dict):
                print(f"{home_team} vs. {away_team}: {prediction_result['prediction']} "
                      f"(H: {prediction_result['probabilities']['Home Win']}, "
                      f"D: {prediction_result['probabilities']['Draw']}, "
                      f"A: {prediction_result['probabilities']['Away Win']})")
            else:
                print(f"{home_team} vs. {away_team}: {prediction_result}")