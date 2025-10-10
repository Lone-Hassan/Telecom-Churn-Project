# Telecom Churn Project

## Overview
This project analyzes telecom subscription data in India to understand customer churn patterns and trends. The goal is to provide insights into factors influencing churn and to build predictive models that can help telecom companies retain customers.

## Project Structure
```
Data/
    final/           # Processed datasets ready for modeling
    interim/         # Intermediate data during processing
    raw/             # Original raw data files
        metadata.csv
        telecom_market_data.csv
docs/                # Project documentation
models/              # Saved models and model artifacts
notebooks/           # Jupyter notebooks for exploration and analysis
    Subscriptive.ipynb
reports/
    figures/         # Generated figures and plots
scripts/             # Data processing and utility scripts
src/                 # Source code for data processing and modeling
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
- Analyze telecom subscription data to identify churn patterns
- Visualize trends and key metrics
- Build predictive models for churn
- Provide actionable insights for telecom companies

## Contributing
Contributions are welcome! Please open issues or submit pull requests for improvements.

## License
This project is licensed under the MIT License.
