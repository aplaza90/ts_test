import pandas as pd
import os


def read_data(filename):
    """
    Reads a CSV file and returns a pandas Series of values indexed by dates.

    The function expects a CSV file located in the 'data' directory.

    The function does a first preprocessing step: droping null values, and typing.

    Args:
        filename (str): The name of the file to read, assumed to be in the '../data' directory.

    Returns:
        pandas.Series: A Series where the index is the date and the values are of type float.

    Raises:
        FileNotFoundError: If the file does not exist in the specified path.
        ValueError: If there are parsing errors due to incorrect data formats.
        pd.errors.EmptyDataError: If the file is empty.

    Example:
        >>> read_data('example.csv')
        date
        2021-01-01    100.0
        2021-01-02    200.0
        Name: value, dtype: float
    """
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, '..', 'data', filename)
    data = pd.read_csv(
        file_path,
        names=['date', 'value'],
        header=0,
        parse_dates=['date'],
        date_parser=lambda x: pd.to_datetime(x, format='%d.%m.%y'),
        dtype={'value': float},
    )
    data.dropna(inplace=True)
    data.set_index('date', inplace=True)
    data.index.freq = 'MS'
    return data['value']


def store_data(data, filename):
    """Stores the data into the appropriate directory"""
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, '..', 'data', filename)
    data.to_csv(file_path)