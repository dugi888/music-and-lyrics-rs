import pandas as pd
import os
from lyrics import download_lyrics as dl, text_processing as tp
import utils
from audio import download_song_data as song_features


class LyricsProcessor:
    def __init__(self, use_stemming=True, n_components=20):
        self.text_processor = tp.TextProcessing(use_stemming, n_components)
        self.download_lyrics = dl.LyricsDownloader()
        self.song_features = song_features.SpotifyClient()

    def start_task_chain(self):
        if not os.path.isdir('output_tables'):
            utils.create_output_dir()
        print("Getting native dataframe")
        df = utils.load_dataset()
        df = utils.rename_columns(df)
        pd.set_option('display.max_columns', None)
        df.to_excel(utils.get_output_directory_path() + '/skladba_dataframe_cleaned.xlsx', index=True)

        self.download_lyrics.run()

        self.text_processor.run()

        self.song_features.run()

    def debugger(self):
        self.song_features.run()


if __name__ == '__main__':
     #print("Starting chain")
     #LyricsProcessor().start_task_chain()
    LyricsProcessor().debugger()
