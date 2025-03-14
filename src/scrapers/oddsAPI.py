# /// script
# dependencies = [
#   "pandas",
#   "requests",
#   "numpy",
#   "rich",
#   "python-dotenv",
#   "pytz",
#   "openpyxl"
# ]
# ///

import os
import sys
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta
import pytz
from pytz import timezone
from statistics import median
from rich.console import Console
from rich.table import Table
from rich.box import MINIMAL
from rich import print as rprint
from rich.logging import RichHandler
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("rich")
console = Console()

# Determine the project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
data_dir = os.path.join(project_root, 'data')

# Ensure data directory exists
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    logger.info(f"Created data directory: {data_dir}")

# Load environment variables
load_dotenv()

def get_odds_data(sport="basketball_ncaab", region="us", markets="h2h,spreads,totals"):
    """
    Fetches odds data from the API for specified sport and markets.
    Only returns games that haven't started yet.
    """
    key = os.getenv("ODDS_API_KEY")
    if not key:
        logger.error("[red]✗[/red] ODDSAPI key not found in environment variables.")
        raise ValueError("ODDSAPI key not found in environment variables.")

    base_url = "https://api.the-odds-api.com"
    odds_url = f"{base_url}/v4/sports/{sport}/odds/?apiKey={key}&regions={region}&markets={markets}&oddsFormat=american"

    try:
        logger.info(f"[cyan]Fetching odds data for {sport}[/cyan]")
        response = requests.get(odds_url)
        response.raise_for_status()
        data = response.json()
        
        if data:
            # Log raw timestamps from API
            logger.info("\n[cyan]Sample raw timestamps from API:[/cyan]")
            for game in data[:3]:
                logger.info(f"Game: {game['away_team']} @ {game['home_team']}")
                logger.info(f"Raw timestamp: {game['commence_time']}")
            
            # Filter out live games using commence_time
            current_time = datetime.now(timezone('UTC'))
            data = [game for game in data if datetime.strptime(game['commence_time'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone('UTC')) > current_time]
            logger.info(f"[cyan]Filtered to {len(data)} upcoming games[/cyan]")
            
            # Create a rich table to display the data
            table = Table(
                title=f"OddsAPI Request ({len(data)} upcoming games)",
                show_header=True,
                header_style="bold magenta",
                show_lines=False,
                title_style="bold cyan",
                padding=(0, 1),
                box=MINIMAL
            )
            
            # Add columns
            table.add_column("Date", style="cyan", width=5)
            table.add_column("Time", style="cyan", width=7)
            table.add_column("Matchup", style="green", width=60)
            
            # Sort games by time
            sorted_games = sorted(data, key=lambda x: x['commence_time'])
            
            # Track date changes for visual separation
            current_date = None
            
            # Add rows for each game
            for game in sorted_games:
                # Convert UTC to ET and format time
                game_time = datetime.strptime(game['commence_time'], '%Y-%m-%dT%H:%M:%SZ')
                # Subtract 5 hours for ET
                et_time = game_time - timedelta(hours=5)
                date_str = et_time.strftime('%m-%d')
                time_str = et_time.strftime('%I:%M%p').lstrip('0').lower()
                
                # Add separator row if date changes
                if current_date != date_str:
                    if current_date is not None:
                        table.add_row("", "", "─" * 60, style="dim")
                    current_date = date_str
                
                # Format matchup more compactly
                matchup = f"{game['away_team']} @ {game['home_team']}"
                
                # Color code based on conference/matchup importance
                style = get_matchup_style(game['home_team'], game['away_team'])
                
                table.add_row(date_str, time_str, matchup, style=style)
            
            # Log the table
            console.print(table)
            logger.info(f"[green]✓[/green] Successfully fetched {len(data)} upcoming games from Odds API")
        
        return data
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"[red]✗[/red] HTTP error occurred: {http_err}")
        return None
    except Exception as err:
        logger.error(f"[red]✗[/red] Error occurred: {err}")
        return None

def get_matchup_style(home_team, away_team):
    """
    Returns a style string based on the matchup importance/conference.
    Add conference-specific or rivalry-specific styling here.
    """
    # Example conference/matchup styling - expand this based on your needs
    power_conferences = ['Kentucky', 'North Carolina', 'Duke', 'Kansas', 'UCLA']
    if any(team in power_conferences for team in [home_team, away_team]):
        return "bold yellow"
    return None

def american_odds_to_implied_probability(odds):
    """
    Convert American odds to implied probability
    
    Args:
        odds (float): American odds (positive or negative)
        
    Returns:
        float: Implied probability between 0 and 1
    """
    if not odds or pd.isna(odds):
        return None
    
    try:
        odds = float(odds)
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)
    except (ValueError, TypeError):
        return None

def devig_moneyline_odds(home_odds, away_odds):
    """
    Remove the vig from moneyline odds using the basic method of proportionally 
    adjusting implied probabilities to sum to 1.
    
    Args:
        home_odds (float): American odds for home team
        away_odds (float): American odds for away team
        
    Returns:
        tuple: (devigged home probability, devigged away probability)
    """
    # Convert to implied probabilities 
    home_prob = american_odds_to_implied_probability(home_odds)
    away_prob = american_odds_to_implied_probability(away_odds)
    
    if home_prob is None or away_prob is None:
        return None, None
        
    # Calculate sum of probabilities (with vig)
    total_prob = home_prob + away_prob
    
    # Remove the vig by proportionally adjusting probabilities to sum to 1
    devigged_home = home_prob / total_prob
    devigged_away = away_prob / total_prob
    
    return devigged_home, devigged_away

def get_moneyline_odds(data):
    """
    Processes moneyline odds into a DataFrame with two rows per game (home and away teams),
    using the median price across all available bookmakers and includes devigged probabilities.
    Only includes games where both home and away prices are available for proper devigging.
    """
    # Dictionary to store all moneylines for each team in each game
    moneyline_dict = {}

    for game in data:
        game_time = game.get('commence_time')  # Keep original ISO format
        home_team = game.get('home_team')
        away_team = game.get('away_team')

        if not all([game_time, home_team, away_team]):
            continue

        # Use game time + teams as unique identifier
        game_key = f"{game_time}_{home_team}_vs_{away_team}"

        # Initialize empty lists to hold moneylines from each bookmaker
        moneyline_dict.setdefault((game_key, 'home'), [])
        moneyline_dict.setdefault((game_key, 'away'), [])

        # Collect all moneylines from each sportsbook
        for bookmaker in game.get('bookmakers', []):
            sportsbook = bookmaker.get('title')
            for market in bookmaker.get('markets', []):
                if market.get('key') == 'h2h':
                    outcomes = market.get('outcomes', [])
                    for outcome in outcomes:
                        team_name = outcome.get('name')
                        price = outcome.get('price')

                        if team_name == home_team and price is not None:
                            moneyline_dict[(game_key, 'home')].append(price)
                        elif team_name == away_team and price is not None:
                            moneyline_dict[(game_key, 'away')].append(price)

    # Build final records with median prices and devigged probabilities
    h2h_records = []
    for game_key in set(k[0] for k in moneyline_dict.keys()):
        home_prices = moneyline_dict.get((game_key, 'home'), [])
        away_prices = moneyline_dict.get((game_key, 'away'), [])
        
        # Skip games where we don't have both home and away prices
        if not home_prices or not away_prices:
            logger.info(f"[yellow]⚠[/yellow] Skipping game {game_key} due to missing moneyline prices")
            continue
        home_probs = [american_odds_to_implied_probability(price) for price in home_prices]
        away_probs = [american_odds_to_implied_probability(price) for price in away_prices]

        med_home_prob = median([p for p in home_probs if p is not None]) if any(p is not None for p in home_probs) else None
        med_away_prob = median([p for p in away_probs if p is not None]) if any(p is not None for p in away_probs) else None

        if med_home_prob is None or med_away_prob is None:
            continue

        # Convert back to American odds
        def implied_probability_to_american_odds(prob):
            if prob >= 0.5:
                return -1 * (prob * 100)/(1 - prob)
            else:
                return (100 - prob * 100)/prob

        med_home_price = implied_probability_to_american_odds(med_home_prob)
        med_away_price = implied_probability_to_american_odds(med_away_prob)
        # Calculate devigged probabilities
        devigged_home_prob, devigged_away_prob = devig_moneyline_odds(med_home_price, med_away_price)

        # Parse game info from the key
        game_time, teams = game_key.split('_', 1)
        home_team, away_team = teams.split('_vs_')

        # Add home team record
        h2h_records.append({
            'Game Time': game_time,  # Keep ISO format
            'Home Team': home_team,
            'Away Team': away_team,
            'Team': home_team,
            'Moneyline': med_home_price,
            'Devigged Probability': devigged_home_prob,
            'Sportsbook': 'CONSENSUS'
        })

        # Add away team record
        h2h_records.append({
            'Game Time': game_time,  # Keep ISO format
            'Home Team': home_team,
            'Away Team': away_team,
            'Team': away_team,
            'Moneyline': med_away_price,
            'Devigged Probability': devigged_away_prob,
            'Sportsbook': 'CONSENSUS'
        })

    return pd.DataFrame(h2h_records)

def get_spread_odds(data):
    """
    Processes spread odds into a DataFrame with two rows per game (home and away teams),
    using the median of all available sportsbooks.
    """
    # Dictionary to store all spreads and prices for each team in each game
    spread_dict = {}

    # Add debugging for raw data
    logger.info("[cyan]Processing spread odds from API data[/cyan]")
    
    for game in data:
        game_time = game.get('commence_time')
        home_team = game.get('home_team')
        away_team = game.get('away_team')

        if not all([game_time, home_team, away_team]):
            continue

        # Use game time + teams as unique identifier
        game_key = f"{game_time}_{home_team}_vs_{away_team}"

        # Initialize empty lists to hold spreads and prices from each bookmaker
        spread_dict.setdefault((game_key, 'home'), {'points': [], 'prices': []})
        spread_dict.setdefault((game_key, 'away'), {'points': [], 'prices': []})

        # Collect spreads from each sportsbook
        for bookmaker in game.get('bookmakers', []):
            for market in bookmaker.get('markets', []):
                if market.get('key') == 'spreads':
                    outcomes = market.get('outcomes', [])
                    for outcome in outcomes:
                        team_name = outcome.get('name')
                        point = outcome.get('point')
                        price = outcome.get('price')

                        if team_name == home_team and point is not None and price is not None:
                            spread_dict[(game_key, 'home')]['points'].append(point)
                            spread_dict[(game_key, 'home')]['prices'].append(price)
                        elif team_name == away_team and point is not None and price is not None:
                            spread_dict[(game_key, 'away')]['points'].append(point)
                            spread_dict[(game_key, 'away')]['prices'].append(price)

    # Build final records with median spreads and prices
    spreads_records = []
    for (game_key, side), values in spread_dict.items():
        if not values['points'] or not values['prices']:
            logger.info(f"[yellow]Missing spread data for game:[/yellow] {game_key}, side: {side}")
            continue

        # Calculate medians
        med_point = median(values['points'])
        med_price = median(values['prices'])

        # Parse game info from the key
        game_time, teams = game_key.split('_', 1)
        home_team, away_team = teams.split('_vs_')

        if side == 'home':
            team_name = home_team
        else:
            team_name = away_team

        spreads_records.append({
            'Game Time': game_time,
            'Home Team': home_team,
            'Away Team': away_team,
            'Team': team_name,
            'Spread': med_point,
            'Spread Price': med_price,
            'Sportsbook': 'CONSENSUS'  # Indicates this is a median across books
        })

    return pd.DataFrame(spreads_records)

def get_totals_odds(data):
    """
    Processes totals odds into a DataFrame with one row per game,
    using the median of all available sportsbooks for over/under lines and prices.
    """
    # Dictionary to store all totals data for each game
    totals_dict = {}

    for game in data:
        game_time = game.get('commence_time')
        home_team = game.get('home_team')
        away_team = game.get('away_team')

        if not all([game_time, home_team, away_team]):
            continue

        # Use game time + teams as unique identifier
        game_key = f"{game_time}_{home_team}_vs_{away_team}"

        # Initialize data structure for this game
        if game_key not in totals_dict:
            totals_dict[game_key] = {
                'over_points': [],
                'over_prices': [],
                'under_points': [],
                'under_prices': []
            }

        # Collect totals from each sportsbook
        for bookmaker in game.get('bookmakers', []):
            for market in bookmaker.get('markets', []):
                if market.get('key') == 'totals':
                    outcomes = market.get('outcomes', [])
                    for outcome in outcomes:
                        if outcome.get('name') == 'Over':
                            if outcome.get('point') is not None:
                                totals_dict[game_key]['over_points'].append(outcome['point'])
                            if outcome.get('price') is not None:
                                totals_dict[game_key]['over_prices'].append(outcome['price'])
                        elif outcome.get('name') == 'Under':
                            if outcome.get('point') is not None:
                                totals_dict[game_key]['under_points'].append(outcome['point'])
                            if outcome.get('price') is not None:
                                totals_dict[game_key]['under_prices'].append(outcome['price'])

    # Build final records with median values
    totals_records = []
    for game_key, values in totals_dict.items():
        if not values['over_points'] or not values['under_points']:
            continue

        # Calculate medians
        med_over_point = median(values['over_points'])
        med_over_price = median(values['over_prices']) if values['over_prices'] else None
        med_under_point = median(values['under_points'])
        med_under_price = median(values['under_prices']) if values['under_prices'] else None

        # Calculate projected total as average of median over/under points
        projected_total = (med_over_point + med_under_point) / 2

        # Parse game info from the key
        game_time, teams = game_key.split('_', 1)
        home_team, away_team = teams.split('_vs_')

        totals_records.append({
            'Game Time': game_time,
            'Home Team': home_team,
            'Away Team': away_team,
            'Projected Total': projected_total,
            'Over Point': med_over_point,
            'Over Price': med_over_price,
            'Under Point': med_under_point,
            'Under Price': med_under_price,
            'Sportsbook': 'CONSENSUS'  # Indicates this is a median across books
        })

    return pd.DataFrame(totals_records)

def get_combined_odds():
    """
    Fetches odds data from API and returns a combined DataFrame with all odds info
    """
    data = get_odds_data()
    if not data:
        return pd.DataFrame()

    # Remove the time-based filtering completely
    # Keep all games, regardless of their start time
    filtered_data = data

    # If everything got filtered out, return empty DataFrame
    if not filtered_data:
        return pd.DataFrame()

    # Get DataFrames and process using all data
    moneyline_df = get_moneyline_odds(filtered_data).drop(columns=['Sportsbook'])
    logger.info(f"\n[cyan]Sample Game Times from moneyline_df:[/cyan]\n{moneyline_df['Game Time'].head()}")
    
    spreads_df = get_spread_odds(filtered_data).drop(columns=['Sportsbook']).rename(
        columns={'Spread': 'Opening Spread'})
    logger.info(f"\n[cyan]Sample Game Times from spreads_df:[/cyan]\n{spreads_df['Game Time'].head()}")
    
    totals_df = get_totals_odds(filtered_data).drop(columns=['Sportsbook'])
    logger.info(f"\n[cyan]Sample Game Times from totals_df:[/cyan]\n{totals_df['Game Time'].head()}")

    # Merge data
    combined_df = pd.merge(
        moneyline_df,
        spreads_df,
        on=['Game Time', 'Home Team', 'Away Team', 'Team'],
        how='outer'
    )
    logger.info(f"\n[cyan]Sample Game Times after first merge:[/cyan]\n{combined_df['Game Time'].head()}")

    combined_df = pd.merge(
        combined_df,
        totals_df,
        on=['Game Time', 'Home Team', 'Away Team'],
        how='outer'
    )
    logger.info(f"\n[cyan]Sample Game Times after second merge:[/cyan]\n{combined_df['Game Time'].head()}")

    # Create game identifier
    combined_df.insert(0, 'Game',
        combined_df['Home Team'] + " vs. " + combined_df['Away Team'])

    # Convert Game Time to datetime, handling both formats
    def convert_game_time(time_str):
        try:
            # Try ISO format first
            dt = pd.to_datetime(time_str)
            # Convert to ET
            et = timezone('US/Eastern')
            if dt.tzinfo is None:
                dt = dt.tz_localize('UTC')
            dt = dt.tz_convert(et)
            return dt
        except (ValueError, TypeError) as e:
            logger.error(f"[red]Error converting time {time_str}: {str(e)}[/red]")
            return pd.NaT

    logger.info(f"\n[cyan]Sample Game Times before conversion:[/cyan]\n{combined_df['Game Time'].head()}")
    
    # Convert all times to ET format after merges are complete
    combined_df['Game Time'] = combined_df['Game Time'].apply(convert_game_time)
    logger.info(f"\n[cyan]Sample Game Times after datetime conversion:[/cyan]\n{combined_df['Game Time'].head()}")
    
    # Format the Game Time column to Month Abbr, Day Time (ET)
    combined_df['Game Time'] = combined_df['Game Time'].dt.strftime('%b %d %I:%M%p ET')
    logger.info(f"\n[cyan]Sample Game Times after formatting:[/cyan]\n{combined_df['Game Time'].head()}")
    
    combined_df = combined_df.sort_values('Game Time', ascending=True)

    return combined_df

def merge_with_combined_data(odds_df):
    """
    Merges odds data with the combined data from previous steps
    """
    logger.info("[cyan]Loading combined data from CSV...[/cyan]")
    combined_data_path = os.path.join(data_dir, 'combined_data.csv')
    
    if not os.path.exists(combined_data_path):
        logger.error(f"[red]✗[/red] Combined data file not found at {combined_data_path}")
        return pd.DataFrame()
    
    # Load the combined data
    combined_df = pd.read_csv(combined_data_path)
    logger.info(f"[green]✓[/green] Loaded combined data with shape: {combined_df.shape}")
    
    # Preview combined data
    logger.info("\n[cyan]Preview of combined_data.csv:[/cyan]")
    logger.info(f"Columns: {', '.join(combined_df.columns[:10])}")
    logger.info(f"First few teams: {combined_df['Team'].head().tolist()}")
    
    # Preview odds data
    logger.info("\n[cyan]Preview of odds data:[/cyan]")
    logger.info(f"Columns: {', '.join(odds_df.columns)}")
    logger.info(f"First few teams: {odds_df['Team'].head().tolist()}")
    
    # Create a team name mapping table if needed
    # For now, let's assume the team names are already aligned
    
    # Merge data on Team, Home Team, and Away Team
    logger.info("[cyan]Merging combined data with odds data...[/cyan]")
    result_df = pd.merge(
        combined_df,
        odds_df,
        on=['Team', 'Home Team', 'Away Team'],
        how='left'
    )
    
    logger.info(f"[green]✓[/green] Merged data has shape: {result_df.shape}")
    
    # Check for merge results
    merge_success_count = result_df['Moneyline'].notna().sum()
    logger.info(f"[cyan]Successfully matched odds for {merge_success_count} / {len(result_df)} rows[/cyan]")
    
    return result_df

def run_oddsapi_etl():
    """
    Main ETL function to run the OddsAPI workflow
    """
    logger.info("=== Starting OddsAPI ETL process ===")
    
    # Get odds data
    logger.info("[cyan]Fetching odds data from API...[/cyan]")
    odds_df = get_combined_odds()
    
    if odds_df.empty:
        logger.error("[red]✗[/red] Failed to get odds data from API")
        return pd.DataFrame()
    
    logger.info(f"[green]✓[/green] Successfully fetched odds data with {len(odds_df)} rows")
    
    # Temporary save for inspection
    odds_csv_path = os.path.join(data_dir, 'oddsapi_raw.csv')
    odds_df.to_csv(odds_csv_path, index=False)
    logger.info(f"[green]✓[/green] Saved raw odds data to {odds_csv_path}")
    
    # Merge with combined data
    logger.info("[cyan]Merging odds data with combined data...[/cyan]")
    final_df = merge_with_combined_data(odds_df)
    
    if final_df.empty:
        logger.error("[red]✗[/red] Failed to merge odds data with combined data")
        return pd.DataFrame()
    
    # Save final results
    output_path = os.path.join(data_dir, 'final_combined_data.csv')
    final_df.to_csv(output_path, index=False)
    logger.info(f"[green]✓[/green] Saved final combined data to {output_path}")
    
    logger.info("=== OddsAPI ETL process completed successfully ===")
    return final_df

if __name__ == "__main__":
    logger.info("=== Starting OddsAPI script ===")
    
    try:
        # Run the ETL process
        result_df = run_oddsapi_etl()
        
        if not result_df.empty:
            output_file = os.path.join(data_dir, 'final_combined_data.csv')
            logger.info(f"[green]✓[/green] Final data shape: {result_df.shape}")
            logger.info(f"[green]✓[/green] Final data saved to: {output_file}")

            csv_path = 'CBB_Output.csv'
            result_df.to_csv(csv_path, index=False)
            logger.info(f"[green]✓[/green] CSV output saved to: {csv_path}")
            
        else:
            logger.error("[red]✗[/red] Script execution failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"[red]✗[/red] Error in OddsAPI script: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
        
    logger.info("=== OddsAPI script completed successfully ===")