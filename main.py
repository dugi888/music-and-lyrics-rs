import pandas as pd


def load_dataset():
    dirpath = "dataset/output_skladba.xlsx"
    columns_to_read = list(range(19))  # load only needed columns
    # noinspection PyTypeChecker
    dataframe = pd.read_excel(dirpath, usecols=columns_to_read)
    return dataframe


def rename_columns(dataframe):
    rename_dict = {
        'Unnamed: 0': 'skladba',
        'ritem': 'ritem_mean',
        'Unnamed: 2': 'ritem_median',
        'Unnamed: 3': 'ritem_std',
        'harmonija': 'harmonija_mean',
        'Unnamed: 5': 'harmonija_median',
        'Unnamed: 6': 'harmonija_std',
        'melodija': 'melodija_mean',
        'Unnamed: 8': 'melodija_median',
        'Unnamed: 9': 'melodija_std',
        'globina_besedila': 'globina_besedila_mean',
        'Unnamed: 11': 'globina_besedila_median',
        'Unnamed: 12': 'globina_besedila_std',
        'razumljivost_besedila': 'razumljivost_besedila_mean',
        'Unnamed: 14': 'razumljivost_besedila_median',
        'Unnamed: 15': 'razumljivost_besedila_std',
        'kompleksnost_besedila': 'kompleksnost_besedila_mean',
        'Unnamed: 17': 'kompleksnost_besedila_median',
        'Unnamed: 18': 'kompleksnost_besedila_std'
    }
    dataframe = dataframe.rename(columns=rename_dict)
    # Drop the first two rows because they are not used for analysis and processing
    dataframe = dataframe.drop([0, 1])

    # Reset the index if you want the indices to be continuous after dropping rows
    dataframe = dataframe.reset_index(drop=True)
    return dataframe


if __name__ == '__main__':
    df = load_dataset()
    df = rename_columns(df)




    pd.set_option('display.max_columns', None)
    print(df.head())
    #df.to_excel('dataframe.xlsx', index=False)
