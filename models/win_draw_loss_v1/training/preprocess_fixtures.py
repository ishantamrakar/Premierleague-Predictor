import pandas as pd
from data_loader import PremierLeagueDataLoader

def preprocess_fixtures(historical_data_path, fixtures_path, output_path):
    """
    Loads historical data and a future fixture list, calculates features for
    the future fixtures by running the full preprocessing pipeline, and saves
    the results.
    """
    # 1. Load historical data and fixtures
    try:
        historical_df = pd.read_csv(historical_data_path, encoding='latin1')
        fixtures_df = pd.read_csv(fixtures_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find data file: {e}")
        return

    # 2. Instantiate data loader and prepare the combined dataframe
    data_loader = PremierLeagueDataLoader(data_path='..')
    
    # Ensure date formats are consistent before combining
    historical_df['MatchDate'] = pd.to_datetime(historical_df['MatchDate'], dayfirst=True, errors='coerce')
    fixtures_df['MatchDate'] = pd.to_datetime(fixtures_df['Date']) # 'Date' is correct for fixtures_df
    
    # Combine historical data with new fixtures
    data_loader.raw_data = pd.concat([historical_df, fixtures_df], ignore_index=True)
    
    # This is crucial: we need to create the team_id mapping based on the combined data
    data_loader._create_team_id_mapping()
    
    # 3. Run the full preprocessing pipeline
    print("Calculating features for all fixtures...")
    data_loader.preprocess_data()
    print("Feature calculation complete.")
    
    # 4. Extract and save the processed fixture rows
    num_fixtures = len(fixtures_df)
    processed_fixtures_df = data_loader.processed_data.tail(num_fixtures)
    
    # Ensure all feature columns are numeric before saving
    feature_cols = [
        'HomeShots', 'AwayShots', 'HomeShotsOnTarget', 'AwayShotsOnTarget', 'HomeCorners', 'AwayCorners',
        'HomeFouls', 'AwayFouls', 'HomeYellowCards', 'AwayYellowCards', 'HomeRedCards', 'AwayRedCards',
        'HomeTeam_Form', 'AwayTeam_Form', 'Match_of_the_Season', 'H2H_HomeWins_Pct', 'H2H_Draws_Pct',
        'H2H_AwayWins_Pct', 'H2H_Home_AvgGoals', 'H2H_Away_AvgGoals', 'Home_Attack_Strength',
        'Home_Defense_Strength', 'Away_Attack_Strength', 'Away_Defense_Strength', 'Home_Weighted_Form',
        'Away_Weighted_Form'
    ]
    # Ensure all feature columns are present and numeric, filling NaNs
    for col in feature_cols:
        if col not in processed_fixtures_df.columns:
            processed_fixtures_df[col] = 0 # Add missing columns
        processed_fixtures_df[col] = pd.to_numeric(processed_fixtures_df[col], errors='coerce').fillna(0)

    processed_fixtures_df.to_csv(output_path, index=False)
    print(f"Processed fixtures with features saved to {output_path}")

if __name__ == '__main__':
    HISTORICAL_DATA_PATH = "../dataset/2000-2025 team stats/epl_final.csv"
    FIXTURES_PATH = "fixtures_2025-26.csv"
    OUTPUT_PATH = "features_2025-26.csv"
    
    preprocess_fixtures(HISTORICAL_DATA_PATH, FIXTURES_PATH, OUTPUT_PATH)
