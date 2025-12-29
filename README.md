# 🎵 DIY Discovery - Spotify Music Recommender

A personalized music recommendation system that finds hidden gems matching your taste, using your own Spotify listening history.

![Config Editor](https://img.shields.io/badge/GUI-Web%20Based-blue) ![Python](https://img.shields.io/badge/Python-3.10%2B-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

## What This Does

This tool analyzes your Spotify listening history and recommends new tracks you'll probably love based on:
- **Your favorite genres** - Finds artists in genres you actually listen to
- **Audio characteristics** - Matches energy, mood, danceability, tempo, key, and more
- **Listening patterns** - Detects your different "moods" (workout music vs. chill, etc.)
- **Popularity filters** - Surfaces hidden gems, not just top 40 hits
- **DJ tools** - Harmonic mixing with Camelot wheel, smart BPM matching

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
- `recommend.py` - Main recommendation engine
- `server.py` - Web server for the config editor
- `config-editor.html` - Web-based configuration UI
- `config.yml` - Your settings and Spotify credentials
- `requirements.txt` - Python dependencies
- `optimize_db.py` - Database optimization utility (optional)

### Step 2: Install Python Dependencies

Open a terminal (Command Prompt on Windows, Terminal on Mac/Linux) and navigate to your folder:

```bash
cd /path/to/your/folder
pip install -r requirements.txt
```

### Step 2b: Install Tidal Sync (Optional)

If you want to sync your playlists to Tidal:

```bash
pip install spotify_to_tidal
```

**First-time Tidal setup:**
1. Run `spotify_to_tidal` once from the command line
2. It will open a browser window asking you to log into Tidal
3. After logging in, it saves your credentials for future use

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

Open `config.yml` in a text editor and fill in your Spotify credentials:

```yaml
spotify:
  client_id: YOUR_SPOTIFY_CLIENT_ID
  client_secret: YOUR_SPOTIFY_CLIENT_SECRET
  username: YOUR_SPOTIFY_USERNAME
  redirect_uri: http://127.0.0.1:8888/callback
```

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
    └── ...
```

**Put this folder in the same directory as the scripts.**

### Getting the Music Database (SQLite Files)

The recommender needs a database of tracks and audio features:
- `spotify_clean.sqlite3` - Artists, tracks, albums, and genres
- `spotify_clean_audio_features.sqlite3` - Audio analysis data (energy, tempo, key, etc.)

Place these in the same folder as the scripts.

---

## 🎮 Running the Recommender

### Option 1: Web Interface (Recommended)

The web interface lets you visually tweak all settings with knobs and sliders.

1. Start the server:
   ```bash
   python server.py
   ```

2. Open your browser to: **http://localhost:8080/config-editor.html**

3. Adjust settings using the knobs and sliders
4. Click the green **▶ Run** button
5. Watch the output stream in real-time!

### Option 2: Command Line

```bash
python recommend.py
```

**Command line options:**
- `--fast` - Test mode with mock data (no API calls)
- `--skip-spotify` - Don't create Spotify playlist
- `--skip-tidal` - Don't sync to Tidal
- `--coverage` - Show database coverage report

---

## 🎛️ Understanding the Settings

### Track Counts
- **Total Tracks** - How many songs in your playlist (default: 35)
- **Serendipity** - Wild card slots for songs outside your usual taste
- **Saved Cap** - How many of your saved tracks to check (to avoid duplicates)

### Candidate Limits
- **Genre Pool** - Max songs to consider from genre matching
- **Related Pool** - Max songs from similar artists
- **Profile Limit** - How many of your top tracks to analyze

### Clustering & Matching
- **Moods** - Number of distinct listening "moods" to detect
- **Recency** - How much to favor recent listening (higher = more recent)
- **Range Penalty** - Penalize songs that are too different from your taste

### DJ Tools (NEW!)
- **Harmonic Mix** - Filter by compatible Camelot keys
- **Smart BPM** - Match 70bpm with 140bpm (double/half time)
- **Target Key** - Lock to a specific key (e.g., 8A for Am)
- **Mode Filter** - Major or minor keys only
- **Vocal Mode** - Instrumental only or vocals only

### Filters
- **Min/Max Tempo** - Filter by BPM range
- **Acoustic Mode** - Acoustic only or electric only
- **Liveness Mode** - Live recordings or studio only
- **Max Speechiness** - Filter out spoken word/podcasts
- **Decade/Era** - Filter by release year
- **Genre Blocklist** - Exclude specific genres

### Feature Weights
Adjust how much each audio characteristic matters:
- **Energy** - Intensity and activity
- **Valence** - Musical positiveness (happy vs. sad)
- **Danceability** - How suitable for dancing
- **Acousticness** - Acoustic vs. electronic
- **Tempo, Loudness, Speechiness, Instrumentalness, Liveness**

---

## 🔧 Troubleshooting

### "ModuleNotFoundError: No module named 'spotipy'"
Run: `pip install -r requirements.txt`

### Browser opens but nothing happens
This is normal on first run! Spotify needs you to authorize the app.

### "Database file not found"
Make sure the SQLite files are in the same folder as the scripts.

### "No streaming history found"
Check that `history_path` in config.yml points to your JSON files.

### Script runs but finds no recommendations
- Try increasing `popularity_max`
- Try changing `genre_match_mode` to `fuzzy`
- Make sure you have enough listening history

### Script is slow
Run the database optimizer to create indexes:
```bash
python optimize_db.py
```

---

## 📁 Output Files

After running, you'll get:
- **Spotify Playlist** - Created in your Spotify account
- **Tidal Sync** - Synced to Tidal (if enabled)
- **Tracklist File** - `~/DIY_Discovery_<timestamp>.txt` with all recommendations

---

## 💡 Tips

1. **Start with defaults** - The default settings work well for most people
2. **Use Randomize** - The 🎲 button generates fun experimental configs
3. **Lower popularity = more obscure** - Set `popularity_max` to 30-40 for deeper cuts
4. **More moods = more variety** - Increase `n_clusters` if your taste is diverse
5. **DJ mixing** - Enable Harmonic Mix and set a Target Key for seamless mixes

---

## 🙏 Credits

Built with:
- [Spotipy](https://spotipy.readthedocs.io/) - Spotify API wrapper
- [scikit-learn](https://scikit-learn.org/) - Machine learning
- [spotify_to_tidal](https://github.com/timrae/spotify_to_tidal) - Tidal sync (optional)

---

**Enjoy discovering new music!** 🎧
