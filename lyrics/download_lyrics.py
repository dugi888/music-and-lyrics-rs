import pandas as pd
import json
import requests
from lyricsgenius import Genius

import utils


class LyricsDownloader(object):
    def __init__(self):
        print("-- Creating LyricsDownloader --")

    # @staticmethod
    # def create_url_link(artist_name, song_title):
    #     return ('https://api.lyrics.ovh/v1/' + artist_name.replace(' ', '%20') + '/' + song_title
    #             .replace(' ', '%20')
    #             .replace("'", "%27")
    #             .replace("(", "%28")
    #             .replace(")", "%29"))
    #
    # def extract_lyrics(self, artist_name, song_title):
    #     link = self.create_url_link(artist_name, song_title)
    #     try:
    #         print("Requesting: ", link, "\n\n")
    #         print(artist_name, " - ", song_title, '\n\n')
    #         req = requests.get(link)
    #         json_data = json.loads(req.content)
    #         song_lyrics = json_data['lyrics']
    #         if song_lyrics:
    #             return song_lyrics, True
    #         else:
    #             return "Lyrics not found.", False
    #     except Exception as e:
    #         return f"Error: {e}", False

    @staticmethod
    def get_lyrics(artist_name, song_title):
        genius = Genius(utils.GENIUS_ACCESS_TOKEN, timeout=5, retries=5)  # Set timeout for Genius API requests
        song = genius.search_song(song_title, artist_name)
        if song and song.lyrics:
            return song.lyrics, True
        else:
            return None, False

    def run(self):
        print("Downloading lyrics!\n")
        print("Loading dataset!\n")
        lyrics_df = pd.DataFrame(columns=["songs", "lyrics"])
        df = pd.read_excel(utils.get_output_directory_path()+'/skladba_dataframe_cleaned.xlsx')

        error_songs = []
        print("Fetching lyrics!\n")

        for song in df['songs']:
            artist = utils.clean_names(song.split(":")[0])
            song_name = utils.clean_names(song.split(":")[1])

            try:
                lyrics, success = self.get_lyrics(artist, song_name)
            except requests.exceptions.Timeout:
                success = False

            if success:
                new_row = {'songs': song, 'lyrics': lyrics}
                lyrics_df.loc[len(lyrics_df)] = new_row
            else:
                error_songs.append(song)

        if len(error_songs) > 0:
            print(f"{len(error_songs)} lyrics couldn't be downloaded!")
            print("I could not download lyrics for these songs:", error_songs)

        print("Saving lyrics dataframe!\n")
        lyrics_df.to_excel(utils.get_output_directory_path()+'/lyrics_output_genius.xlsx', index=False)
        print("Finished saving lyrics dataframe!\n")



