import pandas as pd

def load_data(file_path):
    '''
    Load data from a CSV file into a pandas DataFrame.

    Parameters
    ----------
    file_path : str
        The path to the CSV file to load.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the data loaded from the CSV file.
    '''
    return pd.read_csv(file_path)

def missing_data_table(data):
    """
    Generate a table showing the count and percentage of missing (NaN) values in each column.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to analyze for missing values.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the number and percentage of NaN values per column.
    """
    count = data.isna().sum()
    percentage = round(count / len(data) * 100, 2)
    result = {
        'NaN Count': count,
        'NaN Percentage (%)': percentage
    }
    table = pd.DataFrame(result)
    return table

def percentage_rows_missing_data(df):
    """
    Calculates the percentage of rows with at least one missing value.

    Args:
        df: pandas DataFrame

    Returns:
        float: Percentage of rows with missing data.
    """

    rows_with_missing_data = df.isnull().any(axis=1).sum()
    total_rows = len(df)
    percentage = (rows_with_missing_data / total_rows) * 100
    return percentage