# Telecom Churn Project

## Overview
This project analyzes telecom subscription data in India to understand market share churn patterns and trends. The goal is to provide insights into factors influencing churn and to build predictive models that can help telecom companies retain customers.

## Project Structure
```
Data/
    final/           # Processed datasets ready for modeling
      telecom_market_temporal_patterns.csv   #Finalized for modeling  
    interim/         # Intermediate data during processing
      telecom_market_temporal_patterns.csv    #interim data
    raw/             # Original raw data files
      metadata.csv
      telecom_market_data.csv
docs/                # Project documentation
   model_decision_log   #optuna model optimazation log
models/              # Saved models and model artifacts
notebooks/           # Jupyter notebooks for exploration and analysis
    Cleaning and EDA.ipynb       # For initial data cleaning and EDA
    feature_engineering.ipynb    # For adding new features 
    modeling.ipynb               # Model training and evaluation
reports/
    figures/         # Generated figures and plots
scripts/             # Data processing and utility scripts
   api.py            # Flask API for model serving
   dashboard.py      # streamlit interactive dashboard
src/                 # Source code for data processing and modeling
   data_preprocessing.py   #Data Cleaning and analyzing pattrens
   feature_engineering.py  # business inteligence and feature engineering
tests/               # Unit and integration tests
```

## Data Sources
- **metadata.csv**: Metadata about the dataset.
- **telecom_market_data.csv**: Main dataset containing subscription information.

## Getting Started
1. **Clone the repository:**
   ```powershell
   git clone https://github.com/Lone-Hassan/Telecom-Churn-Project.git
   ```
2. **Set up your environment:**
   - Create a virtual environment (optional but recommended)
   - Install required packages (see `requirements.txt`)

3. **Explore the data:**
   - Use the notebooks in the `notebooks/` directory to explore and analyze the data.

4. **Run scripts:**
   - Data processing and modeling scripts are located in the `scripts/` and `src/` directories.

## Project Goals
- Analyze telecom subscription data to identify market share churn patterns
- Visualize trends and key metrics
- Build predictive models for churn
- Provide actionable insights for telecom companies

## Contributing
Contributions are welcome! Please open issues or submit pull requests for improvements.

## License
This project is licensed under the MIT License.
