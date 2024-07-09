import pandas as pd
import json
import requests


class LyricsDownloader(object):
    def __init__(self):
        print("-- Creating LyricsDownloader --")

    @staticmethod
    def create_url_link(artist_name, song_title):
        return ('https://api.lyrics.ovh/v1/' + artist_name.replace(' ', '%20') + '/' + song_title
                .replace(' ', '%20')
                .replace("'", "%27")
                .replace("(", "%28")
                .replace(")", "%29"))

    def extract_lyrics(self, artist_name, song_title):
        link = self.create_url_link(artist_name, song_title)
        try:
            print("Requesting: ", link, "\n\n")
            print(artist_name, " - ", song_title, '\n\n')
            req = requests.get(link)
            json_data = json.loads(req.content)
            song_lyrics = json_data['lyrics']
            if song_lyrics:
                return song_lyrics, True
            else:
                return "Lyrics not found.", False
        except Exception as e:
            return f"Error: {e}", False

    @staticmethod
    def clean_names(text):
        text = text.strip()
        return " ".join(text.split())

    def run(self):
        print("Loading dataset\n")
        lyrics_df = pd.DataFrame(columns=["songs", "lyrics"])
        df = pd.read_excel("output_tables/skladba_dataframe_cleaned.xlsx")
        error_songs = []
        print("Fetching lyrics\n")
        for song in df['songs']:
            artist = self.clean_names(song.split(":")[0])
            song_name = self.clean_names(song.split(":")[1])
            lyrics, success = self.extract_lyrics(artist, song_name)
            if success:
                new_row = {'songs': song_name, 'lyrics': lyrics}
                lyrics_df.loc[len(lyrics_df)] = new_row
            else:
                error_songs.append(song)
        print(len(error_songs))
        lyrics_df.to_excel('output_tables/lyrics_output.xlsx', index=True)
        print(error_songs)


