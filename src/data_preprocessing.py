import pandas as pd


def data_assessment(df):
    """_ Creates overall summary about data quality _

    Args:
        df (_DataFrame_): _input dataframe_

    Returns:
        _DataFrame_: _DataFrame containing data quality report _
    """
    quality_report ={
        'Column': [],
        'Data Type': [],
        'Non-Null Count': [],
        'Null Count': [],
        'Unique Values': [],
        'outliers': [],
        'Sample Values': []
    }
    for col in df.columns:
        data_type = df[col].dtype
        non_null_count = df[col].notnull().sum()
        null_count = df[col].isnull().sum()
        unique_values = df[col].nunique()
        
        sample_values = df[col].dropna().unique()[:5]  # Get first 5 unique non-null values
        
        quality_report["Column"].append(col)
        quality_report['Data Type'].append(data_type)
        quality_report['Non-Null Count'].append(non_null_count)
        quality_report['Null Count'].append(null_count)
        quality_report['Unique Values'].append(unique_values)
        if col in df.select_dtypes(include='number').columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
            quality_report['outliers'].append(len(outliers))
        else:
            quality_report['outliers'].append('N/A')
        quality_report['Sample Values'].append(sample_values)
        
      
    
    return pd.DataFrame(quality_report)

def clean_telecom_data(df):
    """_ 
    Converts 'value' and 'year' to numeric. 
    Creats a date Column. 
    Standardize cases and trim whitespace.
    Manual mapping for known duplicates in 'circle' column.
    Remove obvious errors like 'value' <= 0
    Drop unnecessary columns like 'month', 'year', 'unit' 
    _

    Args:
        df (_DataFrame_): _ Raw telecom_market_data _

    Returns:
        _type_: _ Cleaned DataFrame _
    """
   
    # Convert datatype to numeric
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df['value'] = df['value'].fillna(method='ffill')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
   

    
    # Create a date column
    df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str), errors='coerce')
    # Reorder columns to have 'date' first
    cols = ['date'] + [col for col in df.columns if col != 'date']
    df = df[cols]
    
    # Standardize cases and trim whitespace
    df['service_provider'] = df['service_provider'].str.upper().str.strip()
    df['circle'] = df['circle'].str.title().str.strip()
    df['type_of_connection'] = df['type_of_connection'].str.lower().str.strip()
    # Manual mapping for known duplicates
    circle_map = {
        'All India': 'All India',
        'All india': 'All India',
        'Chattisgarh': 'Chhattisgarh',
        'Uttaranchal': 'Uttarakhand',
        'Tamil Nadu (Including Chennai)': 'Tamil Nadu',
        'Chennai': 'Tamil Nadu',
        'North East 1': 'North East',
        'North East1': 'North East',
        'North East 2': 'North East',
        'North East2': 'North East',
        'Andaman And Nicobar': 'Andaman And Nicobar Islands'
    }
    df['circle'] = df['circle'].replace(circle_map)
    # Remove obvious errors
    df = df[df['value'] > 0] # Remove negative or zero subscribers
    # drop unnecessary columns
    df = df.drop(columns=['month', 'year', 'unit', 'notes'], axis = 1)
    print("Data cleaning completed.")
    
    return df
def analyze_market_share(df):
    """Calculate total susbscriber in a circle on a particular date\n
    Calculate market share for each provider in each circle\n
    Calculate HHI for each circle on each date\n
    

    Args:
        df (_DataFrame_): clean telecom dataframe 

    Returns:
    tuple:
        df (_DataFrame_): DataFrame with market share and num_competitors columns
        hhi_by_circle (_DataFrame_): DataFrame containing HHI (Herfindahl-Hirschman Index) for each circle on each date
    """
    
    # Calculate total susbscriber in a circle on a particular date
    df['total_circle_subscriber'] = df.groupby(['circle', 'date'])['value'].transform('sum')
    # Calculate market share for each provider in each circle
    df['market_share'] = df['value'] / df['total_circle_subscriber']
    
    # calculate number of competitors in each circle on each date
    df['num_competitors'] = df.groupby(['circle','date'])['service_provider'].transform('nunique')
    
    # Calculate HHI for each circle on each date
    hhi_by_circle = df.groupby(['circle','date'])["market_share"].apply(lambda x: (x**2).sum()).reset_index(name='HHI')
    
    return df, hhi_by_circle

# time series patterns identification
def analyze_temporal_patterns(df):
    """_Sort by operator, circle, and date\n
    Calculate month-over-month growth rate\n
    Identify seasonal patterns\n
    Calculate rolling averages\n
    _

    Args:
        df (DataFrame): _Cleaned telecom DataFrame _

    Returns:
        _DataFrame_: _ Sorted DataFrame_
    """
    # Sort by operator, circle, and date
    df = df.sort_values(by=['service_provider', 'circle', 'date'])
    # Calculate month-over-month growth rate    
    df['subscriber_change'] = df.groupby(['service_provider', 'circle'])['value'].pct_change()
    

    # Identify seasonal patterns
    df['month_num'] = df['date'].dt.month
    df["quarter"] = df['date'].dt.quarter
    # Calculate rolling averages
    for window in [3, 6, 12]:
        df[f'subscribers_ma_{window}'] = df.groupby(['service_provider', 'circle'])['value'].transform(lambda x: x.rolling(window, min_periods=1).mean())
        
    return df