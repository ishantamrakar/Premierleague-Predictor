import re
import pandas as pd
from datetime import datetime

# text from espn.com
fixture_text = """
Friday, Aug. 15 2025
Liverpool vs. AFC Bournemouth (20:00 UK)

Saturday, Aug. 16, 2025
Aston Villa vs. Newcastle United (15:00 UK)
Brighton & Hove Albion vs. Fulham (15:00 UK)
Sunderland vs. West Ham United (15:00 UK)
Tottenham Hotspur vs. Burnley (15:00 UK)
Wolves vs. Manchester City (17:30 UK)

Sunday, Aug. 17, 2025
Chelsea vs. Crystal Palace (14:00 UK)
Nottingham Forest vs. Brentford (14:00 UK)
Manchester United vs. Arsenal (16:30 UK)

Monday, Aug. 18, 2025
Leeds United vs. Everton (20:00 UK)

CLUB FIXTURES 2025-26
AFC Bournemouth	Leeds United
Arsenal	Liverpool
Aston Villa	Man City
Brentford	Man United
Brighton	Newcastle United
Burnley	Nottm Forest
Chelsea	Sunderland
Crystal Palace	Tottenham Hotspur
Everton	West Ham United
Fulham	Wolves

Friday, Aug. 22 2025
West Ham United vs. Chelsea (20:00 UK)

Saturday, Aug. 23, 2025
Manchester City vs. Tottenham Hotspur (12:30 UK)
AFC Bournemouth vs. Wolves (15:00 UK)
Brentford vs. Aston Villa (15:00 UK)
Burnley vs. Sunderland (15:00 UK)
Arsenal vs. Leeds United (17:30 UK)

Sunday, Aug. 24, 2025
Crystal Palace vs. Nottingham Forest (14:00 UK)
Everton vs. Brighton (14:00 UK)
Fulham vs. Manchester United (16:30 UK)

Monday, Aug. 25, 2025
Newcastle United vs. Liverpool (20:00 UK)

Saturday, Aug. 30, 2025
Chelsea vs. Fulham (12:30 UK)
Manchester United vs. Burnley (15:00 UK)
Sunderland vs. Brentford (15:00 UK)
Tottenham Hotspur vs. AFC Bournemouth (15:00 UK)
Wolves vs. Everton (15:00 UK)
Leeds United vs. Newcastle United (17:30 UK)

Sunday, Aug. 31, 2025
Brighton vs. Manchester City (14:00 UK)
Nottingham Forest vs. West Ham United (14:00 UK)
Liverpool vs. Arsenal (16:30 UK)
Aston Villa vs. Crystal Palace (19:00 UK)

Saturday, Sept. 13, 2025
Arsenal vs. Nottingham Forest (12:30 UK)
AFC Bournemouth vs. Brighton (15:00 UK)
Crystal Palace vs. Sunderland (15:00 UK)
Everton vs. Aston Villa (15:00 UK)
Fulham vs. Leeds United (15:00 UK)
Newcastle United vs. Wolves (15:00 UK)
West Ham United vs. Tottenham Hotspur (17:30 UK)
Brentford vs. Chelsea (20:00 UK)

Sunday, Sept. 14, 2025
Burnley vs. Liverpool (14:00 UK)
Manchester City vs. Manchester United (16:30 UK)

Saturday, Sept. 20, 2025
Liverpool vs. Everton (12:30 UK)
AFC Bournemouth vs. Newcastle United (15:00 UK)
Brighton vs. Tottenham Hotspur (15:00 UK)
Burnley vs. Nottingham Forest (15:00 UK)
West Ham United vs. Crystal Palace (15:00 UK)
Wolves vs. Leeds United (15:00 UK)
Manchester United vs. Chelsea (17:30 UK)
Fulham vs. Brentford (20:00 UK)

Saturday, Sept. 21, 2025
Sunderland vs. Aston Villa (14:00 UK)
Arsenal vs. Manchester City (16:30 UK)

Saturday, Sept. 27, 2025
Brentford vs. Manchester United (12:30 UK)
Aston Villa vs. Fulham (15:00 UK)*
Chelsea vs. Brighton (15:00 UK)
Crystal Palace vs. Liverpool (15:00 UK)
Leeds United vs. AFC Bournemouth (15:00 UK)
Manchester City vs. Burnley (15:00 UK)
Nottingham Forest vs. Sunderland (17:30 UK)*

*May move back due to Europa League fixtures

Sunday, Sept. 27, 2025
Tottenham Hotspur vs. Wolves (14:00 UK)*
Newcastle United vs. Arsenal (16:30 UK)

*May move forward due to Champions League fixture

Monday, Sept. 28, 2025
Everton vs. West Ham United (20:00 UK)

Saturday, Oct. 4, 2025
AFC Bournemouth vs. Fulham
Arsenal vs. West Ham United
Aston Villa vs. Burnley*
Brentford vs. Manchester City
Chelsea vs. Liverpool
Everton vs. Crystal Palace*
Leeds United vs. Tottenham Hotspur
Manchester United vs. Sunderland
Newcastle United vs. Nottingham Forest*
Wolves vs. Brighton

*May move back due to European fixtures

START DATES 2025-26
UEFA Super Cup	Aug. 13
Premier League begins	Aug. 16
Ligue 1 begins	Aug. 16
LaLiga begins	Aug. 17
Bundesliga begins	Aug. 22
Serie A begins	Aug. 23
UCL group stage draw	Aug. 28
Transfer window closes	Sept. 1
UCL begins	Sept. 16
Saturday, Oct. 18, 2025
Brighton vs. Newcastle United
Burnley vs. Leeds United
Crystal Palace vs. AFC Bournemouth
Fulham vs. Arsenal
Liverpool vs. Manchester United
Manchester City vs. Everton
Nottingham Forest vs. Chelsea
Sunderland vs. Wolves
Tottenham Hotspur vs. Aston Villa
West Ham United vs. Brentford

Saturday, Oct. 25, 2025
AFC Bournemouth vs. Nottingham Forest*
Arsenal vs. Crystal Palace*
Aston Villa vs. Manchester City*
Brentford vs. Liverpool
Chelsea vs. Sunderland
Everton vs. Tottenham Hotspur
Leeds United vs. West Ham United
Manchester United vs. Brighton
Newcastle United vs. Fulham
Wolves vs. Burnley

*May move back due to European fixtures

Saturday, Nov. 1, 2025
Brighton vs. Leeds United
Burnley vs. Arsenal
Crystal Palace vs. Brentford
Fulham vs. Wolves
Liverpool vs. Aston Villa
Manchester City vs. AFC Bournemouth
Nottingham Forest vs. Manchester United
Sunderland vs. Everton
Tottenham Hotspur vs. Chelsea
West Ham United vs. Newcastle United

Saturday, Nov. 8, 2025
Aston Villa vs. AFC Bournemouth*
Brentford vs. Newcastle United
Chelsea vs. Wolves
Crystal Palace vs. Brighton*
Everton vs. Fulham
Manchester City vs. Liverpool
Nottingham Forest vs. Leeds United*
Sunderland vs. Arsenal
Tottenham Hotspur vs. Manchester United
West Ham United vs. Burnley

*May move back due to European fixtures

Saturday, Nov. 22, 2025
AFC Bournemouth vs. West Ham United
Arsenal vs. Tottenham Hotspur
Brighton vs. Brentford
Burnley vs. Chelsea
Fulham vs. Sunderland
Leeds United vs. Aston Villa
Liverpool vs. Nottingham Forest
Manchester United vs. Everton
Newcastle United vs. Manchester City
Wolves vs. Crystal Palace

Saturday, Nov. 29, 2025
Aston Villa vs. Wolves*
Brentford vs. Burnley
Chelsea vs. Arsenal
Crystal Palace vs. Manchester United*
Everton vs. Newcastle United
Manchester City vs. Leeds United
Nottingham Forest vs. Brighton*
Sunderland vs. AFC Bournemouth
Tottenham Hotspur vs. Fulham
West Ham United vs. Liverpool

*May move back due to European fixtures

Wednesday, Dec. 3, 2025
AFC Bournemouth vs. Everton
Arsenal vs. Brentford
Brighton vs. Aston Villa
Burnley vs. Crystal Palace
Fulham vs. Manchester City
Leeds United vs. Chelsea
Liverpool vs. Sunderland
Manchester United vs. West Ham United
Newcastle United vs. Tottenham Hotspur
Wolves vs. Nottingham Forest

Saturday, Dec. 6, 2025
AFC Bournemouth vs. Chelsea
Aston Villa vs. Arsenal
Brighton vs. West Ham United
Everton vs. Nottingham Forest
Fulham vs. Crystal Palace
Leeds United vs. Liverpool
Manchester City vs. Sunderland
Newcastle United vs. Burnley
Tottenham Hotspur vs. Brentford
Wolves vs. Manchester United

Saturday, Dec. 13, 2025
Arsenal vs. Wolves
Brentford vs. Leeds United
Burnley vs. Fulham
Chelsea vs. Everton
Crystal Palace vs. Manchester City*
Liverpool vs. Brighton
Manchester United vs. AFC Bournemouth
Nottingham Forest vs. Tottenham Hotspur*
Sunderland vs. Newcastle United
West Ham United vs. Aston Villa*

*May move back due to European fixtures

Saturday, Dec. 20, 2025
AFC Bournemouth vs. Burnley
Aston Villa vs. Manchester United
Brighton vs. Sunderland
Everton vs. Arsenal
Fulham vs. Nottingham Forest
Leeds United vs. Crystal Palace*
Manchester City vs. West Ham United
Newcastle United vs. Chelsea
Tottenham Hotspur vs. Liverpool
Wolves vs. Brentford

*May move back due to Conference League fixture

Saturday, Dec. 27, 2025
Arsenal vs. Brighton
Brentford vs. AFC Bournemouth
Burnley vs. Everton
Chelsea vs. Aston Villa
Crystal Palace vs. Tottenham Hotspur
Liverpool vs. Wolves
Manchester United vs. Newcastle United
Nottingham Forest vs. Manchester City
Sunderland vs. Leeds United
West Ham United vs. Fulham

Tuesday, Dec. 30, 2025
Arsenal vs. Aston Villa
Brentford vs. Tottenham Hotspur
Burnley vs. Newcastle United
Chelsea vs. AFC Bournemouth
Crystal Palace vs. Fulham
Liverpool vs. Leeds United
Manchester United vs. Wolves
Nottingham Forest vs. Everton
Sunderland vs. Manchester City
West Ham United vs. Brighton

Saturday, Jan. 3, 2026
AFC Bournemouth vs. Arsenal
Aston Villa vs. Nottingham Forest
Brighton vs. Burnley
Everton vs. Brentford
Fulham vs. Liverpool
Leeds United vs. Manchester United
Manchester City vs. Chelsea
Newcastle United vs. Crystal Palace
Tottenham Hotspur vs. Sunderland
Wolves vs. West Ham United

Wednesday, Jan. 7, 2026
AFC Bournemouth vs. Tottenham Hotspur
Arsenal vs. Liverpool
Brentford vs. Sunderland
Burnley vs. Manchester United
Crystal Palace vs. Aston Villa
Everton vs. Wolves
Fulham vs. Chelsea
Manchester City vs. Brighton
Newcastle United vs. Leeds United
West Ham United vs. Nottingham Forest

Saturday, Jan. 17, 2026
Aston Villa vs. Everton
Brighton vs. AFC Bournemouth
Chelsea vs. Brentford
Leeds United vs. Fulham
Liverpool vs. Burnley
Manchester United vs. Manchester City
Nottingham Forest vs. Arsenal
Sunderland vs. Crystal Palace
Tottenham Hotspur vs. West Ham United
Wolves vs. Newcastle United

Saturday, Jan. 24, 2026
AFC Bournemouth vs. Liverpool
Arsenal vs. Manchester United
Brentford vs. Nottingham Forest*
Burnley vs. Tottenham Hotspur
Crystal Palace vs. Chelsea
Everton vs. Leeds United
Fulham vs. Brighton
Manchester City vs. Wolves
Newcastle United vs. Aston Villa*
West Ham United vs. Sunderland

*May move back due to Europa League fixtures

Saturday, Jan. 31, 2026
Aston Villa vs. Brentford*
Brighton vs. Everton
Chelsea vs. West Ham United
Leeds United vs. Arsenal
Liverpool vs. Newcastle United
Manchester United vs. Fulham
Nottingham Forest vs. Crystal Palace*
Sunderland vs. Burnley
Tottenham Hotspur vs. Manchester City
Wolves vs. AFC Bournemouth

*May move back due to Europa League fixtures

Saturday, Feb. 7, 2026
AFC Bournemouth vs. Aston Villa
Arsenal vs. Sunderland
Brighton vs. Crystal Palace
Burnley vs. West Ham United
Fulham vs. Everton
Leeds United vs. Nottingham Forest
Liverpool vs. Manchester City
Manchester United vs. Tottenham Hotspur
Newcastle United vs. Brentford
Wolves vs. Chelsea

Wednesday, Feb. 11, 2026
Aston Villa vs. Brighton
Brentford vs. Arsenal
Chelsea vs. Leeds United
Crystal Palace vs. Burnley
Everton vs. AFC Bournemouth
Manchester City vs. Fulham
Nottingham Forest vs. Wolves
Sunderland vs. Liverpool
Tottenham Hotspur vs. Newcastle United
West Ham United vs. Manchester United

Saturday, Feb. 21, 2026
Aston Villa vs. Leeds United
Brentford vs. Brighton
Chelsea vs. Burnley
Crystal Palace vs. Wolves
Everton vs. Manchester United
Manchester City vs. Newcastle United
Nottingham Forest vs. Liverpool
Sunderland vs. Fulham
Tottenham Hotspur vs. Arsenal
West Ham United vs. AFC Bournemouth

Saturday, Feb. 28, 2026
AFC Bournemouth vs. Sunderland
Arsenal vs. Chelsea
Brighton vs. Nottingham Forest
Burnley vs. Brentford
Fulham vs. Tottenham Hotspur
Leeds United vs. Manchester City
Liverpool vs. West Ham United
Manchester United vs. Crystal Palace
Newcastle United vs. Everton
Wolves vs. Aston Villa

Wednesday, March 4, 2026
AFC Bournemouth vs. Brentford
Aston Villa vs. Chelsea
Brighton vs. Arsenal
Everton vs. Burnley
Fulham vs. West Ham United
Leeds United vs. Sunderland
Manchester City vs. Nottingham Forest
Newcastle United vs. Manchester United
Tottenham Hotspur vs. Crystal Palace
Wolves vs. Liverpool

Saturday, March 14, 2026
Arsenal vs. Everton
Brentford vs. Wolves
Burnley vs. AFC Bournemouth
Chelsea vs. Newcastle United
Crystal Palace vs. Leeds United
Liverpool vs. Tottenham Hotspur
Manchester United vs. Aston Villa
Nottingham Forest vs. Fulham
Sunderland vs. Brighton
West Ham United vs. Manchester City

Saturday, March 21, 2026
AFC Bournemouth vs. Manchester United
Aston Villa vs. West Ham United
Brighton vs. Liverpool
Everton vs. Chelsea
Fulham vs. Burnley
Leeds United vs. Brentford
Manchester City vs. Crystal Palace
Newcastle United vs. Sunderland
Tottenham Hotspur vs. Nottingham Forest
Wolves vs. Arsenal

Saturday, April 11, 2026
Arsenal vs. AFC Bournemouth
Brentford vs. Everton
Burnley vs. Brighton
Chelsea vs. Manchester City
Crystal Palace vs. Newcastle United
Liverpool vs. Fulham
Manchester United vs. Leeds United
Nottingham Forest vs. Aston Villa
Sunderland vs. Tottenham Hotspur
West Ham United vs. Wolves

Saturday, April 18, 2026
Aston Villa vs. Sunderland
Brentford vs. Fulham
Chelsea vs. Manchester United
Crystal Palace vs. West Ham United
Everton vs. Liverpool
Leeds United vs. Wolves
Manchester City vs. Arsenal
Newcastle United vs. AFC Bournemouth
Nottingham Forest vs. Burnley
Tottenham Hotspur vs. Brighton

Saturday, April 25, 2026
AFC Bournemouth vs. Leeds United
Arsenal vs. Newcastle United
Brighton vs. Chelsea
Burnley vs. Manchester City
Fulham vs. Aston Villa
Liverpool vs. Crystal Palace
Manchester United vs. Brentford
Sunderland vs. Nottingham Forest
West Ham United vs. Everton
Wolves vs. Tottenham Hotspur


Saturday, May 2, 2026
AFC Bournemouth vs. Crystal Palace
Arsenal vs. Fulham
Aston Villa vs. Tottenham Hotspur
Brentford vs. West Ham United
Chelsea vs. Nottingham Forest
Everton vs. Manchester City
Leeds United vs. Burnley
Manchester United vs. Liverpool
Newcastle United vs. Brighton
Wolves vs. Sunderland

Saturday, May 9, 2026
Brighton vs. Wolves
Burnley vs. Aston Villa
Crystal Palace vs. Everton
Fulham vs. AFC Bournemouth
Liverpool vs. Chelsea
Manchester City vs. Brentford
Nottingham Forest vs. Newcastle United
Sunderland vs. Manchester United
Tottenham Hotspur vs. Leeds United
West Ham United vs. Arsenal

Sunday, May 17, 2026
AFC Bournemouth vs. Manchester City
Arsenal vs. Burnley
Aston Villa vs. Liverpool
Brentford vs. Crystal Palace
Chelsea vs. Tottenham Hotspur
Everton vs. Sunderland
Leeds United vs. Brighton
Manchester United vs. Nottingham Forest
Newcastle United vs. West Ham United
Wolves vs. Fulham

Sunday, May 24, 2026
Brighton vs. Manchester United
Burnley vs. Wolves
Crystal Palace vs. Arsenal
Fulham vs. Newcastle United
Liverpool vs. Brentford
Manchester City vs. Aston Villa
Nottingham Forest vs. AFC Bournemouth
Sunderland vs. Chelsea
Tottenham Hotspur vs. Everton
West Ham United vs. Leeds United
"""

matches = []
current_date = None

# Regex to find date lines (e.g., "Friday, Aug. 15 2025")
date_pattern = re.compile(r"^(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2}),?\s+(\d{4})")
# Regex to find match lines (e.g., "Team A vs. Team B")
match_pattern = re.compile(r"^(.*?)\s+vs\.\s+(.*?)(?:\s+\(.*\)|$|\s*\*)")

for line in fixture_text.strip().split('\n'):
    line = line.strip()
    
    # Skip empty lines or lines with notes
    if not line or line.startswith('*') or "FIXTURES" in line or "DATES" in line:
        continue

    date_match = date_pattern.match(line)
    if date_match:
        # We have found a date line, parse it and set it as the current date
        try:
            # Handle variations in month abbreviation
            month_str = date_match.group(1)
            
            date_str = f"{month_str} {date_match.group(2)} {date_match.group(3)}"
            current_date = datetime.strptime(date_str, "%b %d %Y").strftime("%Y-%m-%d")
        except ValueError:
            print(f"Could not parse date: {line}")
            current_date = None
        continue

    match_match = match_pattern.match(line)
    if match_match and current_date:
        # We have found a match line and have a valid current date
        home_team = match_match.group(1).strip()
        away_team = match_match.group(2).strip().replace('*', '') # Remove trailing asterisks
        
        # Handle team name inconsistencies
        team_name_map = {
            "AFC Bournemouth": "Bournemouth",
            "Brighton & Hove Albion": "Brighton",
            "Nottingham Forest": "Nott'm Forest",
            "Wolverhampton Wanderers": "Wolves",
            "Manchester United": "Man United",
            "Manchester City": "Man City",
            "West Ham United": "West Ham",
            "Tottenham Hotspur": "Tottenham",
            "Newcastle United": "Newcastle",
            "Leeds United": "Leeds"
        }
        home_team = team_name_map.get(home_team, home_team)
        away_team = team_name_map.get(away_team, away_team)
        
        matches.append({
            "Date": current_date,
            "HomeTeam": home_team,
            "AwayTeam": away_team,
            "Season": "2025/26"
        })

# Create a DataFrame and save to CSV
if matches:
    df = pd.DataFrame(matches)
    df.to_csv("fixtures_2025-26.csv", index=False)
    print("Successfully created fixtures_2025-26.csv")
else:
    print("No matches were parsed.")
