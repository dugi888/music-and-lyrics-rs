import pandas as pd
import os
import download_lyrics as dl
import text_processing as tp


class LyricsProcessor:
    def __init__(self, use_stemming=True, n_components=20):
        self.text_processor = tp.TextProcessing(use_stemming, n_components)
        self.download_lyrics = dl.LyricsDownloader()

    @staticmethod
    def load_dataset():
        dirpath = "dataset/output_skladba.xlsx"
        columns_to_read = list(range(19))  # load only needed columns
        # noinspection PyTypeChecker
        dataframe = pd.read_excel(dirpath, usecols=columns_to_read)
        return dataframe

    @staticmethod
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

    @staticmethod
    def create_output_dir():
        directory = "output_tables"
        # Parent Directory path
        parent_dir = "./"
        # Path
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)

    def start_task_chain(self):
        if not os.path.isdir('output_tables'):
            self.create_output_dir()
        print("Getting native dataframe")
        df = self.load_dataset()
        df = self.rename_columns(df)
        pd.set_option('display.max_columns', None)
        df.to_excel('output_tables/skladba_dataframe_cleaned.xlsx', index=True)

        print("Downloading lyrics")
        self.download_lyrics.run()

        print("Processing lyrics")
        self.text_processor.run()

    def debugger(self):
        self.text_processor.run()


if __name__ == '__main__':
    # print("Starting chain")
    #LyricsProcessor().start_task_chain()

    LyricsProcessor().debugger()
