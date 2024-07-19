import os
import requests
from requests import utils
import pandas as pd
from dotenv import load_dotenv
import utils

load_dotenv()
client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")


class SpotifyClient:
    @staticmethod
    def create_url_link(base_url, artist, song_name):
        # Construct the Spotify search URL
        query = f"{artist} {song_name}"
        url = base_url + requests.utils.quote(query)
        url += "&type=track"
        return url

    def search_songs(self, token, artist, song_name):
        base_url = "https://api.spotify.com/v1/search?q="
        url = self.create_url_link(base_url, artist, song_name)
        headers = utils.get_auth_header(token)

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise an exception for bad responses (4xx or 5xx)
            json_result = response.json()["tracks"]["items"]

            # Look for exact match by artist
            for track in json_result:
                if artist.lower() == track["artists"][0]['name'].lower() and song_name.lower() == track['name'].lower():
                    print(f"{artist} - {song_name} found!")
                    print("ID: ", track['id'], "\t URL: ", track['external_urls']['spotify'])
                    return track['id'], True

            # If exact match not found, return the first track that matches the artist
            for track in json_result:
                if artist.lower() == track["artists"][0]['name'].lower():
                    print(f"{artist} - {song_name} not found. Returning closest match.")
                    print("ID: ", track['id'], "\t URL: ", track['external_urls']['spotify'])
                    return track['id'], True

            # If no match found at all
            print(f"No match found for {artist} - {song_name}")
            return None, False

        except requests.exceptions.RequestException as e:
            print("Error in search_songs: ", e)
            return None, False

    @staticmethod
    def get_track_features(token, track_id):
        url = f"https://api.spotify.com/v1/audio-features/{track_id}"
        headers = utils.get_auth_header(token)

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise an exception for bad responses (4xx or 5xx)
            track_features = response.json()
            return track_features, True

        except requests.exceptions.RequestException as e:
            print("Error in get_track_features: ", e)
            return None, False

    def download_song_ids(self):
        print("Loading dataset!\n")
        token = utils.get_token()
        song_ids_df = pd.DataFrame(columns=["songs", "song_id"])
        df = pd.read_excel(utils.get_output_directory_path()+'/skladba_dataframe_cleaned.xlsx')
        error_songs = []
        print("Fetching songs' ids!\n")
        for song in df['songs']:
            artist = utils.clean_names(song.split(":")[0])
            song_name = utils.clean_names(song.split(":")[1])
            song_id, success = self.search_songs(token, artist, song_name)
            if success:
                new_row = {'songs': song, 'song_id': song_id}
                song_ids_df.loc[len(song_ids_df)] = new_row
            else:
                error_songs.append(song)
        print("Saving dataset\n")
        song_ids_df.to_excel(utils.get_output_directory_path()+'/song_ids_output.xlsx', index=False)
        print(error_songs)

    def download_track_features(self):
        print("Downloading track features and saving them!")
        token = utils.get_token()
        df = pd.read_excel(utils.get_output_directory_path()+'/song_ids_output.xlsx')
        error_songs = []

        columns = [
            "songs", "danceability", "energy", "key", "loudness", "mode",
            "speechiness", "acousticness", "instrumentalness", "liveness",
            "valence", "tempo", "duration_ms", "time_signature"
        ]

        song_features_df = pd.DataFrame(columns=columns)

        print("Fetching songs' features!\n")
        for index, row in df.iterrows():
            song_id = row['song_id']
            artist_and_song = row['songs']
            song_features, success = self.get_track_features(token, song_id)
            if success:
                new_row = {
                    'songs': artist_and_song,
                    'danceability': song_features.get('danceability'),
                    'energy': song_features.get('energy'),
                    'key': song_features.get('key'),
                    'loudness': song_features.get('loudness'),
                    'mode': song_features.get('mode'),
                    'speechiness': song_features.get('speechiness'),
                    'acousticness': song_features.get('acousticness'),
                    'instrumentalness': song_features.get('instrumentalness'),
                    'liveness': song_features.get('liveness'),
                    'valence': song_features.get('valence'),
                    'tempo': song_features.get('tempo'),
                    'duration_ms': song_features.get('duration_ms')
                }
                song_features_df.loc[len(song_features_df)] = new_row
            else:
                error_songs.append(artist_and_song)
        print("Saving dataset\n")
        song_features_df.to_excel(utils.get_output_directory_path()+'/song_features_output.xlsx', index=False)
        print(error_songs)

    def run(self):
        self.download_song_ids()
        self.download_track_features()


if __name__ == '__main__':
    sc = SpotifyClient()
    sc.run()



# Example usage:
# main()


# token = get_token()
# artist = "Bo Donaldson & The Heywoods"
# song_name = "Billy, Don't Be A Hero"
# search_songs(token, artist, song_name)
