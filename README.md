# 🎵 DIY Discovery - Spotify Music Recommender

A personalized music recommendation system that finds hidden gems matching your taste, using your own Spotify listening history.

![Config Editor](https://img.shields.io/badge/GUI-Web%20Based-blue) ![Python](https://img.shields.io/badge/Python-3.10%2B-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

## What This Does

This tool analyzes your Spotify listening history and recommends new tracks you'll probably love based on:
- **Your favorite genres** - Finds artists in genres you actually listen to
- **Audio characteristics** - Matches energy, mood, danceability, etc.
- **Listening patterns** - Detects your different "moods" (workout music vs. chill, etc.)
- **Popularity filters** - Surfaces hidden gems, not just top 40 hits

The recommendations get saved as a Spotify playlist and optionally synced to Tidal.

---

## 📋 Requirements

Before you start, you'll need:

1. **Python 3.10 or newer** - [Download Python](https://www.python.org/downloads/)
2. **A Spotify account** (free or premium)
3. **Your Spotify Extended Streaming History** (we'll show you how to get this)
4. **A music database** (SQLite files - instructions below)

---

## 🚀 Installation

### Step 1: Download the Files

Download all these files and put them in the same folder:
- `recommend.py`
- `server.py`
- `config-editor.html`
- `config.yml`
- `requirements.txt`

### Step 2: Install Python Dependencies

Open a terminal (Command Prompt on Windows, Terminal on Mac/Linux) and navigate to your folder:

**On Windows:**
```
cd C:\path\to\your\folder
pip install -r requirements.txt
```

**On Mac/Linux:**
```
cd /path/to/your/folder
pip3 install -r requirements.txt
```

### Step 2b: Install Tidal Sync (Optional)

If you want to sync your playlists to Tidal, install the `spotify_to_tidal` package:

**On Windows:**
```
pip install spotify_to_tidal
```

**On Mac/Linux:**
```
pip3 install spotify_to_tidal
```

**First-time Tidal setup:**
1. Run `spotify_to_tidal` once from the command line
2. It will open a browser window asking you to log into Tidal
3. After logging in, it saves your credentials for future use

If you don't have Tidal, skip this step - you can still use the Spotify playlist feature.

### Step 3: Set Up Spotify API Credentials

1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Log in with your Spotify account
3. Click **"Create App"**
   - App name: `DIY Discovery` (or anything you want)
   - App description: `Personal music recommender`
   - Redirect URI: `http://127.0.0.1:8888/callback`
   - Check the agreement box and click **Create**
4. On your new app's page, click **"Settings"**
5. Copy your **Client ID** and **Client Secret**

### Step 4: Configure Your Settings

Open `config.yml` in a text editor (Notepad, TextEdit, VS Code, etc.) and fill in your Spotify credentials:

```yaml
spotify:
  client_id: PASTE_YOUR_CLIENT_ID_HERE
  client_secret: PASTE_YOUR_CLIENT_SECRET_HERE
  username: YOUR_SPOTIFY_USERNAME
  redirect_uri: http://127.0.0.1:8888/callback
  open_browser: True
```

Save the file.

---

## 📊 Getting Your Data

### Getting Your Spotify Streaming History

This is the most important step! You need your Extended Streaming History from Spotify.

1. Go to [Spotify Privacy Settings](https://www.spotify.com/account/privacy/)
2. Scroll down to **"Download your data"**
3. Check **"Extended streaming history"** 
4. Click **"Request data"**
5. Wait for the email (can take **up to 30 days**, usually 5-7 days)
6. Download the ZIP file when it arrives
7. Extract it somewhere you'll remember

After extracting, you should have a folder like:
```
my_spotify_data/
└── Spotify Extended Streaming History/
    ├── endsong_0.json
    ├── endsong_1.json
    ├── endsong_2.json
    └── ... (more files)
```

**Put this folder in the same directory as the scripts.**

### Getting the Music Database (SQLite Files)

The recommender needs a database of tracks and audio features to search through. You'll need these SQLite files:

```
artists.db       - Artist information and genres
tracks.db        - Track information  
audio_features.db - Audio analysis data (energy, tempo, etc.)
```

**Where to get these:**
These files should be obtained separately. They contain Spotify catalog data. Place them in the same folder as the scripts.

Your folder should look like this:
```
your-folder/
├── recommend.py
├── server.py
├── config-editor.html
├── config.yml
├── requirements.txt
├── artists.db
├── tracks.db
├── audio_features.db
└── my_spotify_data/
    └── Spotify Extended Streaming History/
        ├── endsong_0.json
        └── ...
```

---

## 🎮 Running the Recommender

### Option 1: Web Interface (Recommended for Beginners)

The web interface lets you visually tweak all settings with knobs and sliders.

1. Open a terminal in your folder
2. Start the server:

   **Windows:**
   ```
   python server.py
   ```
   
   **Mac/Linux:**
   ```
   python3 server.py
   ```

3. Open your web browser and go to: **http://localhost:8080/config-editor.html**

4. Adjust the settings using the knobs and sliders
5. Click the green **▶ Run** button
6. Watch the output stream in real-time!

### Option 2: Command Line

If you prefer the terminal:

**Windows:**
```
python recommend.py
```

**Mac/Linux:**
```
python3 recommend.py
```

---

## 🎛️ Understanding the Settings

### Track Counts
- **Total Tracks** - How many songs in your playlist (default: 35)
- **Serendipity** - Wild card slots for songs outside your usual taste
- **Saved Cap** - How many of your saved tracks to check (to avoid duplicates)

### Candidate Limits
- **Genre Limit** - Max songs to consider from genre matching
- **Related Limit** - Max songs from similar artists
- **Profile Limit** - How many of your top tracks to analyze

### Clustering
- **Moods** - Number of distinct listening "moods" to detect (e.g., workout vs. chill)
- **Recency** - How much to favor recent listening (higher = more recent)
- **Range Penalty** - Penalize songs that are too different from your taste

### Thresholds
- **Min Play** - Ignore plays shorter than this (in milliseconds)
- **Popularity Range** - Filter by Spotify popularity score (0-100)

### Feature Weights
Adjust how much each audio characteristic matters:
- **Energy** - Intensity and activity
- **Valence** - Musical positiveness (happy vs. sad)
- **Danceability** - How suitable for dancing
- **Acousticness** - Acoustic vs. electronic
- **Tempo** - Speed of the track
- And more...

---

## 🔧 Troubleshooting

### "ModuleNotFoundError: No module named 'spotipy'"
Run: `pip install -r requirements.txt`

### "No module named 'yaml'"
Run: `pip install pyyaml`

### Browser opens but nothing happens
This is normal on first run! Spotify needs you to authorize the app. Log in and click "Agree" when prompted.

### "Database file not found"
Make sure `artists.db`, `tracks.db`, and `audio_features.db` are in the same folder as the scripts.

### "No streaming history found"
- Check that `history_path` in config.yml points to your JSON files
- The folder should contain files named `endsong_0.json`, `endsong_1.json`, etc.

### Script runs but finds no recommendations
- Try increasing `popularity_max` (songs might be filtered out)
- Try changing `genre_match_mode` from `exact` to `prefix` or `fuzzy`
- Make sure you have enough listening history (need at least a few weeks of data)

---

## 📁 Output Files

After running, you'll get:
- **Spotify Playlist** - Created in your Spotify account (if enabled)
- **Tidal Sync** - Synced to Tidal (if enabled and spotify_to_tidal is installed)
- **Tracklist File** - A text file with all recommendations and explanations

---

## 💡 Tips

1. **Start with defaults** - The default settings work well for most people
2. **Use Randomize** - The 🎲 button generates fun experimental configs
3. **Lower popularity = more obscure** - Set `popularity_max` to 30-40 for deeper cuts
4. **More moods = more variety** - Increase `n_clusters` if your taste is diverse

---

## ❓ FAQ

**Q: How long does it take?**
A: Usually 2-5 minutes depending on your settings and history size.

**Q: Will this post to my Spotify?**
A: Only if you click "Save to Spotify" - it creates a private playlist.

**Q: Does this work without the streaming history?**
A: Not well. The local history is what makes recommendations accurate.

**Q: Can I use this commercially?**
A: I don't know. I haven't tried it.
---

## 🙏 Credits

Built with:
- [Spotipy](https://spotipy.readthedocs.io/) - Spotify API wrapper
- [scikit-learn](https://scikit-learn.org/) - Machine learning
- [spotify_to_tidal](https://github.com/spotify-to-tidal) - Tidal sync (optional)

---

**Enjoy discovering new music!** 🎧
