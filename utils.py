import base64
import os
from requests import post
import json
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
client_id = os.getenv("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
GENIUS_ACCESS_TOKEN = os.getenv("GENIUS_TOKEN")


def clean_names(text):
    text = text.strip()
    return " ".join(text.split())


def get_auth_header(token):
    # Create authorization header
    return {"Authorization": f"Bearer {token}"}


def get_token():
    auth_string = client_id + ":" + client_secret
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")

    url = "https://accounts.spotify.com/api/token"
    headers = {"Authorization": "Basic " + auth_base64,
               "Content-Type": "application/x-www-form-urlencoded"
               }
    data = {"grant_type": "client_credentials"}
    result = post(url, headers=headers, data=data)
    json_result = json.loads(result.content)
    token = json_result["access_token"]
    return token


def load_dataset():
    dirpath = "dataset/output_skladba.xlsx"
    columns_to_read = list(range(19))  # load only needed columns
    # noinspection PyTypeChecker
    dataframe = pd.read_excel(dirpath, usecols=columns_to_read)
    return dataframe


def rename_columns(dataframe):
    cols = {
        'Unnamed: 0': 'songs',
        'ritem': 'rhythm_mean',
        'Unnamed: 2': 'rhythm_median',
        'Unnamed: 3': 'rhythm_std',
        'harmonija': 'harmony_mean',
        'Unnamed: 5': 'harmony_median',
        'Unnamed: 6': 'harmony_std',
        'melodija': 'melody_mean',
        'Unnamed: 8': 'melody_median',
        'Unnamed: 9': 'melody_std',
        'globina_besedila': 'lyrics_depth_mean',
        'Unnamed: 11': 'lyrics_depth_median',
        'Unnamed: 12': 'lyrics_depth_std',
        'razumljivost_besedila': 'lyrics_comprehensibility_mean',
        'Unnamed: 14': 'lyrics_comprehensibility_median',
        'Unnamed: 15': 'lyrics_comprehensibility_std',
        'kompleksnost_besedila': 'lyrics_complexity_mean',
        'Unnamed: 17': 'lyrics_complexity_median',
        'Unnamed: 18': 'lyrics_complexity_std'
    }
    dataframe = dataframe.rename(columns=cols)
    # Drop the first two rows because they are not used for analysis and processing
    dataframe = dataframe.drop([0, 1])

    # Reset the index if you want the indices to be continuous after dropping rows
    dataframe = dataframe.reset_index(drop=True)
    return dataframe


def create_output_dir():
    path = "./output_tables"
    os.mkdir(path)


def get_output_directory_path(directory_name="output_tables", start_dir='.'):
    start_path = Path(start_dir).resolve()
    while start_path:
        # Check if the directory exists at this level
        for path in start_path.iterdir():
            if path.is_dir() and path.name == directory_name:
                return str(path.resolve())
        # Move up to the parent directory
        if start_path == start_path.parent:  # If we reach the root, stop
            break
        start_path = start_path.parent
    return None


def merge_dataframes(first_df, second_df, output_file_name=None, column_name="songs", how_param="inner"):
    merged_df = pd.merge(first_df, second_df, on=column_name, how=how_param)
    if output_file_name is not None:
        merged_df.to_excel(get_output_directory_path() + '/' + output_file_name, index=False)
    return merged_df


# def classify_value(value, value_at_1_3, value_at_2_3):
#     if value < value_at_1_3:
#         return 'LOW'
#     elif value <= value_at_2_3:
#         return 'MID'
#     else:
#         return 'HIGH'

#
# def find_boundaries(column):
#     sorted = column.sort_values(ascending=True)
#
#     index_1_3 = int(len(sorted) * (1 / 3))
#     index_2_3 = int(len(sorted) * (2 / 3))
#
#     # Retrieve the value at the 1/3 position
#     value_at_1_3 = sorted.iloc[index_1_3]
#
#     # Retrieve the value at the 2/3 position
#     value_at_2_3 = sorted.iloc[index_2_3]
#
#     return value_at_1_3, value_at_2_3
#
#
# def create_classifications(df):
#     for col in df.columns:
#         if col.lower() == 'songs':
#             continue
#         one_third_value, two_thirds_value = find_boundaries(df[col])
#         df.loc[:, col + '_class'] = df[col].apply(
#             lambda x: classify_value(x, one_third_value, two_thirds_value))
#     df.to_excel(get_output_directory_path() + '/lyrics_features_classified.xlsx', index=False)
#     return df

def find_boundaries(column):
    quantiles = column.quantile([1/3, 2/3]).values
    value_at_1_3, value_at_2_3 = quantiles[0], quantiles[1]
    return value_at_1_3, value_at_2_3


def classify_value(x, one_third_value, two_thirds_value):
    if x <= one_third_value:
        return 'LOW'
    elif x <= two_thirds_value:
        return 'MID'
    else:
        return 'HIGH'


def create_classifications(df):
    for col in df.columns:
        if col.lower() == 'songs':
            continue
        one_third_value, two_thirds_value = find_boundaries(df[col])
        df[col + '_class'] = df[col].apply(lambda x: classify_value(x, one_third_value, two_thirds_value))

    output_path = get_output_directory_path() + '/lyrics_features_classified.xlsx'
    df.to_excel(output_path, index=False)
    return df
