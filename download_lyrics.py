import pandas as pd
import json
import requests


def read_excel():
    return pd.read_excel("./dataframe.xlsx")


def create_url_link(artist_name, song_title):
    return ('https://api.lyrics.ovh/v1/' + artist_name.replace(' ', '%20') + '/' + song_title
            .replace(' ', '%20')
            .replace("'", "%27")
            .replace("(", "%28")
            .replace(")", "%29"))


def extract_lyrics(artist_name, song_title):
    link = create_url_link(artist_name, song_title)
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


def clean_names(text):
    text = text.strip()
    return " ".join(text.split())


if __name__ == '__main__':
    print("Loading dataset\n")
    lyrics_df = pd.DataFrame(columns=["Song", "Lyrics"])
    df = read_excel()
    error_songs = []
    print("Fetching lyrics\n")
    for song in df['songs']:
        artist = clean_names(song.split(":")[0])
        song_name = clean_names(song.split(":")[1])
        lyrics, success = extract_lyrics(artist, song_name)
        if success:
            new_row = {'Song': song_name, 'Lyrics': lyrics}
            lyrics_df.loc[len(lyrics_df)] = new_row
        else:
            error_songs.append(song)
    print(len(error_songs))
    lyrics_df.to_excel('lyrics_output.xlsx', index=True)
    print(error_songs)
