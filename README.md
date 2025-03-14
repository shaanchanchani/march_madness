# KenPom and EvanMiya Basketball Data Scraper

This project scrapes and processes basketball analytics data from KenPom and EvanMiya websites, saving the results as CSV files for further analysis.

## Project Structure

```
kp-em-scrape/
├── src/                           # Source code directory
│   ├── index.js                   # Main entry point that runs all scrapers
│   ├── scrapers/                  # Directory containing web scrapers
│   │   ├── kenpom-scraper.js      # Scrapes KenPom FanMatch HTML data
│   │   └── evanmiya-scraper.js    # Scrapes EvanMiya CSV data
│   ├── parsers/                   # Directory containing data parsers
│   │   └── kenpom-parser.py       # Parses KenPom HTML into CSV format
│   └── transformers/              # Directory containing data transformers
│       └── evanmiya-transformer.js # Transforms EvanMiya CSV to desired format
├── data/                          # Output directory for processed data
│   ├── kp.csv                     # KenPom processed data
│   └── em.csv                     # EvanMiya processed data
├── kenpom-data/                   # Directory where KenPom HTML files are stored
├── package.json                   # Project dependencies and scripts
├── .env.example                   # Example environment variables file
└── .env                           # Environment variables (credentials)
```

## How it Works

1. **EvanMiya Scraper**: Logs into EvanMiya.com, navigates to the game predictions page, and downloads the CSV data directly to `data/em.csv`.

2. **KenPom Scraper**: Logs into KenPom.com, navigates through the FanMatch pages, and saves HTML files to the `kenpom-data` directory.

3. **KenPom Parser**: A Python script that processes the HTML files in `kenpom-data` and extracts the relevant data into `data/kp.csv`. This script is run using UV, which automatically handles dependencies.

4. **EvanMiya Transformer**: Transforms the raw EvanMiya CSV data into a format matching the KenPom data, with two rows per game (one for each team perspective).

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd kp-em-scrape
```

2. Install Node.js dependencies:
```bash
npm install
```

3. Install UV (for Python dependency management):
```bash
pip install uv
```

4. Create `.env` file with your credentials:
```
EMAIL=your_email@example.com
PASSWORD=your_password
```

## Usage

### Run all scrapers
```bash
npm start
```

### Run individual components
```bash
# Scrape KenPom data
npm run scrape:kenpom

# Scrape EvanMiya data
npm run scrape:evanmiya

# Only parse KenPom HTML (if already downloaded)
npm run parse:kenpom

# Only transform EvanMiya data (if already downloaded)
npm run transform:evanmiya
```

## Data Output

The scrapers will output two main files:
- `data/kp.csv`: KenPom data in CSV format
- `data/em.csv`: EvanMiya data in CSV format

The CSV files have the following columns:
- `Home Team`: Name of the home team
- `Away Team`: Name of the away team
- `Team`: Team from whose perspective this row represents
- `Game Date`: Date of the game in YYYYMMDD format
- `spread_kenpom`/`spread_evanmiya`: Point spread (negative numbers favor this team)
- `win_prob_kenpom`/`win_prob_evanmiya`: Probability of winning (0-1)
- `projected_total_kenpom`/`projected_total_evanmiya`: Projected total points in the game

## Requirements

- Node.js 14+
- Python 3.8+
- UV (Python package manager)
- Playwright (automatically installed via npm)
- Python packages (automatically installed via UV when running the parser):
  - beautifulsoup4
  - python-dotenv
  - psycopg2-binary
- Valid accounts for KenPom.com and EvanMiya.com

## Notes

- This project is designed for educational and personal use only.
- Please respect the terms of service of both websites.
- The scrapers include rate limiting and browser-like behavior to avoid being blocked. 