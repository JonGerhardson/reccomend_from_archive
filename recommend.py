import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import numpy as np
import sqlite3
import os
import subprocess
import random
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from typing import Any
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import warnings
import argparse

warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', category=FutureWarning)  # KMeans warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Clean output matching previous print style
)
logger = logging.getLogger(__name__)

# --- CAMELOT WHEEL CONSTANTS ---
# Mapping (key, mode) -> Camelot Code
# key: 0=C, 1=C#, 2=D, ... 11=B
# mode: 1=Major, 0=Minor
CAMELOT_WHEEL = {
    (0, 1): '8B', (1, 1): '3B', (2, 1): '10B', (3, 1): '5B', (4, 1): '12B',
    (5, 1): '7B', (6, 1): '2B', (7, 1): '9B', (8, 1): '4B', (9, 1): '11B', 
    (10, 1): '6B', (11, 1): '1B',
    (0, 0): '5A', (1, 0): '12A', (2, 0): '7A', (3, 0): '2A', (4, 0): '9A',
    (5, 0): '4A', (6, 0): '11A', (7, 0): '6A', (8, 0): '1A', (9, 0): '8A', 
    (10, 0): '3A', (11, 0): '10A'
}

def get_camelot_code(key: int, mode: int) -> str:
    """Convert Spotify key/mode to Camelot Code."""
    return CAMELOT_WHEEL.get((int(key), int(mode)), '??')

def get_compatible_camelot_keys(target_code: str) -> set[str]:
    """Get list of harmonically compatible Camelot keys."""
    if not target_code or target_code == '??':
        return set()
        
    # extract number and letter
    if target_code[-1] not in ('A', 'B'): 
        return {target_code}
    
    number = int(target_code[:-1])
    letter = target_code[-1]
    
    compatible = {target_code}
    
    # 1. Same Number, different letter (Relative Major/Minor)
    # 8A <-> 8B
    other_letter = 'B' if letter == 'A' else 'A'
    compatible.add(f"{number}{other_letter}")
    
    # 2. +/- 1 Hour (Perfect 5th / 4th) - same letter
    # 8A -> 7A, 9A
    prev_num = number - 1 if number > 1 else 12
    next_num = number + 1 if number < 12 else 1
    compatible.add(f"{prev_num}{letter}")
    compatible.add(f"{next_num}{letter}")
    
    # Optional: Diagonal mixes (energy boost/drop) e.g. 8A -> 9B
    # keeping it simple for now
    
    return compatible

# Parse command line arguments (only when run as script, not when imported)
def _parse_args() -> argparse.Namespace:
    """Parse command line arguments. Safe to call when imported."""
    parser = argparse.ArgumentParser(description='Spotify music recommender')
    parser.add_argument('--fast', action='store_true', help='Fast mode with mock data for testing')
    parser.add_argument('--skip-spotify', action='store_true', help='Skip saving playlist to Spotify')
    parser.add_argument('--skip-tidal', action='store_true', help='Skip syncing playlist to Tidal')
    parser.add_argument('--no-cache', action='store_true', help='Force regeneration of user profile (ignore cache)')
    # Use parse_known_args to ignore unknown args (allows pytest to work)
    args, _ = parser.parse_known_args()
    return args

ARGS = _parse_args()


# --- CONFIGURATION ---
# Load secrets and settings from environment variables or config file
def _load_config() -> tuple[str, str, str, dict[str, Any]]:
    """Load Spotify credentials and recommender settings from environment or config.yml."""
    import yaml
    
    client_id = os.getenv('SPOTIPY_CLIENT_ID')
    client_secret = os.getenv('SPOTIPY_CLIENT_SECRET')
    redirect_uri = os.getenv('SPOTIPY_REDIRECT_URI', 'http://127.0.0.1:8888/callback')
    recommender_cfg: dict[str, Any] = {}
    
    # Load from config.yml
    config_path = os.path.join(os.path.dirname(__file__), 'config.yml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            client_id = client_id or config.get('spotify', {}).get('client_id')
            client_secret = client_secret or config.get('spotify', {}).get('client_secret')
            redirect_uri = config.get('spotify', {}).get('redirect_uri', redirect_uri)
            recommender_cfg = config.get('recommender', {})
    
    if not client_id or not client_secret:
        raise ValueError(
            "Spotify credentials not found. Set SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET "
            "environment variables, or create config.yml with spotify.client_id and spotify.client_secret"
        )
    
    return client_id, client_secret, redirect_uri, recommender_cfg

SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI, _RECOMMENDER_CFG = _load_config()

# Fix DB paths to be absolute (robust against CWD changes)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_SCRIPT_DIR, 'spotify_clean.sqlite3')
AUDIO_FEATURES_DB_PATH = os.path.join(_SCRIPT_DIR, 'spotify_clean_audio_features.sqlite3')



class DatabaseConnection:
    """Context manager for database connections with optional caching.
    
    Reduces repeated connection opens/closes by caching connections when used
    with the same path. Can be used as regular context manager or with shared mode.
    
    Usage:
        # Single use (opens/closes each time):
        with DatabaseConnection(DB_PATH) as con:
            con.execute(...)
        
        # Shared mode (uses cached connection):
        db = DatabaseConnection.get_shared(DB_PATH)
        db.execute(...)
    """
    _shared_connections: dict[str, 'sqlite3.Connection'] = {}
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = None
    
    def __enter__(self) -> 'sqlite3.Connection':
        self.connection = sqlite3.connect(self.db_path)
        return self.connection
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            self.connection.close()
            self.connection = None
    
    @classmethod
    def get_shared(cls, db_path: str) -> 'sqlite3.Connection':
        """Get or create a shared connection for the given path."""
        if db_path not in cls._shared_connections:
            cls._shared_connections[db_path] = sqlite3.connect(db_path)
        return cls._shared_connections[db_path]
    
    @classmethod
    def close_all_shared(cls):
        """Close all shared connections (call at end of script)."""
        for con in cls._shared_connections.values():
            con.close()
        cls._shared_connections.clear()

# All available audio features for better matching
FEATURE_COLS = [
    'danceability', 'energy', 'loudness', 'speechiness', 
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

# Type aliases for cleaner signatures
TrackCandidate = dict[str, Any]  # {'id', 'name', 'artist', 'popularity'}
ClusterStats = dict[str, np.ndarray]  # {'mean', 'std', 'min', 'max'}
# Added dict[str, float] for skip_map at the end
ProfileResult = tuple[np.ndarray, np.ndarray, list[ClusterStats], list[str], list[str], MinMaxScaler, np.ndarray, dict[str, float]]

# Default feature weights - can be overridden in config.yml
DEFAULT_FEATURE_WEIGHTS = {
    'energy': 1.5,
    'valence': 1.5,
    'danceability': 1.2,
    'acousticness': 1.2,
    'tempo': 0.8,
    'loudness': 0.8,
    'speechiness': 0.6,
    'instrumentalness': 0.6,
    'liveness': 0.5,
}

# Loaded from config.yml with fallback defaults
@dataclass
class RecommenderConfig:
    """Configuration for the recommender system, loaded from config.yml."""
    # Clustering
    n_clusters: int = 3
    recency_decay: float = 2.0
    range_penalty_strength: float = 0.1
    genre_match_mode: str = 'prefix'
    
    # Track limits
    serendipity_slots: int = 2
    total_tracks: int = 35
    saved_tracks_cap: int = 500
    genre_candidates_limit: int = 200
    related_candidates_limit: int = 80
    profile_tracks_limit: int = 150
    
    # Local history settings
    use_local_history: bool = False
    history_path: str = 'my_spotify_data/Spotify Extended Streaming History'
    use_api_related_artists: bool = False
    
    # Thresholds (magic numbers)
    min_play_ms: int = 30000
    popularity_min: int = 5
    popularity_max: int = 50
    similar_popularity_min: int = 5
    similar_popularity_max: int = 60
    serendipity_popularity_min: int = 10
    serendipity_popularity_max: int = 60
    
    # Collaborative filtering
    top_artists_limit: int = 20
    
    # Display
    console_preview_count: int = 10
    
    # Feature weights (dict loaded separately)
    feature_weights: dict[str, float] = field(default_factory=lambda: DEFAULT_FEATURE_WEIGHTS.copy())
    
    # Filters
    date_start: str | None = None
    date_end: str | None = None
    blocked_genres: list[str] = field(default_factory=list)
    
    # Advanced Tweaks
    skip_penalty: float = 0.5    # Reduce weight for skipped tracks
    repeat_boost: float = 1.2    # Boost for tracks played > 5 times
    max_speechiness: float | None = None
    acoustic_option: str | None = None  # 'acoustic', 'electric', or None
    liveness_option: str | None = None  # 'live', 'studio', or None
    
    # Rediscovery
    rediscovery_mode: bool = False
    rediscovery_days: int = 365  # Default: hasn't been played in a year
    
    # Era / Decade
    release_year_min: int | None = None
    release_year_max: int | None = None

    # New Filters
    min_tempo: int | None = None
    max_tempo: int | None = None
    vocal_mode: str | None = None  # 'instrumental_only', 'vocal_only', or None
    mode_filter: str | None = None # 'major', 'minor', or None

    # DJ Tools
    smart_bpm: bool = True     # Allow tracking doubling/halving (e.g. 80bpm matches 160bpm)
    dj_mode: bool = False      # Prioritize harmonic mixing
    target_camelot: str | None = None # Force specific Camelot key (e.g. '8A')
    
    # Skips
    penalize_skips: bool = True # Toggle for candidate skip penalty

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> 'RecommenderConfig':
        """Create config from dictionary, using defaults for missing keys."""
        return cls(
            n_clusters=cfg.get('n_clusters', 3),
            recency_decay=cfg.get('recency_decay', 2.0),
            range_penalty_strength=cfg.get('range_penalty_strength', 0.1),
            genre_match_mode=cfg.get('genre_match_mode', 'prefix'),
            serendipity_slots=cfg.get('serendipity_slots', 2),
            total_tracks=cfg.get('total_tracks', 35),
            saved_tracks_cap=cfg.get('saved_tracks_cap', 500),
            genre_candidates_limit=cfg.get('genre_candidates_limit', 200),
            related_candidates_limit=cfg.get('related_candidates_limit', 80),
            profile_tracks_limit=cfg.get('profile_tracks_limit', 150),
            use_local_history=cfg.get('use_local_history', False),
            history_path=cfg.get('history_path', 'my_spotify_data/Spotify Extended Streaming History'),
            use_api_related_artists=cfg.get('use_api_related_artists', False),
            min_play_ms=cfg.get('min_play_ms', 30000),
            popularity_min=cfg.get('popularity_min', 5),
            popularity_max=cfg.get('popularity_max', 50),
            similar_popularity_min=cfg.get('similar_popularity_min', 5),
            similar_popularity_max=cfg.get('similar_popularity_max', 60),
            serendipity_popularity_min=cfg.get('serendipity_popularity_min', 10),
            serendipity_popularity_max=cfg.get('serendipity_popularity_max', 60),
            top_artists_limit=cfg.get('top_artists_limit', 20),
            console_preview_count=cfg.get('console_preview_count', 10),
            feature_weights={**DEFAULT_FEATURE_WEIGHTS, **cfg.get('feature_weights', {})},
            date_start=cfg.get('date_start'),
            date_end=cfg.get('date_end'),
            blocked_genres=cfg.get('blocked_genres', []),
            skip_penalty=cfg.get('skip_penalty', 0.5),
            repeat_boost=cfg.get('repeat_boost', 1.2),
            max_speechiness=cfg.get('max_speechiness'),
            acoustic_option=cfg.get('acoustic_option'),
            liveness_option=cfg.get('liveness_option'),
            rediscovery_mode=cfg.get('rediscovery_mode', False),
            rediscovery_days=cfg.get('rediscovery_days', 365),
            release_year_min=cfg.get('release_year_min'),
            release_year_max=cfg.get('release_year_max'),
            min_tempo=cfg.get('min_tempo'),
            max_tempo=cfg.get('max_tempo'),
            vocal_mode=cfg.get('vocal_mode'),
            mode_filter=cfg.get('mode_filter'),
            smart_bpm=cfg.get('smart_bpm', True),
            dj_mode=cfg.get('dj_mode', False),
            target_camelot=cfg.get('target_camelot'),
            penalize_skips=cfg.get('penalize_skips', True),
        )

# Initialize config from loaded YAML
config = RecommenderConfig.from_dict(_RECOMMENDER_CFG)

# Convenience aliases for backward compatibility (can be removed once all code uses config.*)
N_CLUSTERS = config.n_clusters
RANGE_PENALTY_STRENGTH = config.range_penalty_strength
RECENCY_DECAY = config.recency_decay
GENRE_MATCH_MODE = config.genre_match_mode
SERENDIPITY_SLOTS = config.serendipity_slots
TOTAL_TRACKS = config.total_tracks
SAVED_TRACKS_CAP = config.saved_tracks_cap
GENRE_CANDIDATES_LIMIT = config.genre_candidates_limit
RELATED_CANDIDATES_LIMIT = config.related_candidates_limit
USE_LOCAL_HISTORY = config.use_local_history
HISTORY_PATH = config.history_path
USE_API_RELATED_ARTISTS = config.use_api_related_artists
FEATURE_WEIGHTS = config.feature_weights

# --- AUTHENTICATION (lazy-loaded) ---
_sp: spotipy.Spotify | None = None

def get_spotify_client() -> spotipy.Spotify:
    """Get or create the Spotify client (lazy initialization).
    
    Avoids triggering OAuth flow at import time.
    """
    global _sp
    if _sp is None:
        _sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=SPOTIPY_CLIENT_ID,
            client_secret=SPOTIPY_CLIENT_SECRET,
            redirect_uri=SPOTIPY_REDIRECT_URI,
            scope="user-top-read playlist-modify-public user-library-read"
        ))
    return _sp

# --- MOCK DATA FOR FAST MODE ---
def generate_mock_data() -> tuple[pd.DataFrame, list[TrackCandidate], pd.DataFrame, np.ndarray]:
    """Generate mock data for --fast mode testing."""
    np.random.seed(42)
    
    # Mock user profile with 3 distinct clusters
    # Simulate recency: first 10 = short_term, next 10 = medium_term, last 10 = long_term
    time_range_weights = {
        'short_term': RECENCY_DECAY ** 2,
        'medium_term': RECENCY_DECAY ** 1,
        'long_term': RECENCY_DECAY ** 0,
    }
    
    mock_features = pd.DataFrame({
        'track_id': [f'mock_{i}' for i in range(30)],
        'danceability': np.concatenate([np.random.uniform(0.7, 0.9, 10), np.random.uniform(0.3, 0.5, 10), np.random.uniform(0.5, 0.7, 10)]),
        'energy': np.concatenate([np.random.uniform(0.8, 1.0, 10), np.random.uniform(0.2, 0.4, 10), np.random.uniform(0.5, 0.7, 10)]),
        'loudness': np.random.uniform(-10, -5, 30),
        'speechiness': np.random.uniform(0.03, 0.1, 30),
        'acousticness': np.concatenate([np.random.uniform(0.0, 0.2, 10), np.random.uniform(0.6, 0.9, 10), np.random.uniform(0.3, 0.5, 10)]),
        'instrumentalness': np.random.uniform(0.0, 0.3, 30),
        'liveness': np.random.uniform(0.1, 0.3, 30),
        'valence': np.concatenate([np.random.uniform(0.6, 0.9, 10), np.random.uniform(0.2, 0.4, 10), np.random.uniform(0.4, 0.6, 10)]),
        'tempo': np.random.uniform(100, 140, 30),
        'mode': np.random.randint(0, 2, 30),  # 0 or 1
        'key': np.random.randint(0, 12, 30),  # 0-11
    })
    
    # Mock recency weights: short (0-9), medium (10-19), long (20-29)
    mock_sample_weights = np.array(
        [time_range_weights['short_term']] * 10 +
        [time_range_weights['medium_term']] * 10 +
        [time_range_weights['long_term']] * 10
    )
    
    # Mock candidates (using dict format for consistency)
    mock_candidates = [
        {'id': f'cand_{i}', 'name': f'Mock Track {i}', 'artist': f'Mock Artist {i % 5}', 'popularity': int(np.random.randint(5, 50))}
        for i in range(20)
    ]
    
    mock_candidate_features = pd.DataFrame({
        'track_id': [c['id'] for c in mock_candidates],
        'danceability': np.random.uniform(0.3, 0.9, 20),
        'energy': np.random.uniform(0.2, 1.0, 20),
        'loudness': np.random.uniform(-12, -4, 20),
        'speechiness': np.random.uniform(0.03, 0.15, 20),
        'acousticness': np.random.uniform(0.0, 0.8, 20),
        'instrumentalness': np.random.uniform(0.0, 0.5, 20),
        'liveness': np.random.uniform(0.1, 0.4, 20),
        'valence': np.random.uniform(0.2, 0.9, 20),
        'tempo': np.random.uniform(80, 160, 20),
        'mode': np.random.randint(0, 2, 20),
        'key': np.random.randint(0, 12, 20),
    })
    
    return mock_features, mock_candidates, mock_candidate_features, mock_sample_weights

def load_streaming_history() -> pd.DataFrame:
    """Load extended streaming history from local JSON files.
    
    Returns DataFrame with columns: track_id, artist_name, track_name, ts, ms_played,
    platform, country
    """
    import glob
    import json
    
    history_dir = os.path.join(os.path.dirname(__file__), HISTORY_PATH)
    files = sorted(glob.glob(os.path.join(history_dir, 'Streaming_History_Audio_*.json')))
    
    if not files:
        raise FileNotFoundError(f"No streaming history files found in {history_dir}")
    
    logger.info(f"Analyze: Loading local streaming history from {len(files)} files...")
    
    all_records = []
    for f in files:
        with open(f, 'r') as fp:
            data = json.load(fp)
            for r in data:
                # Skip entries without track URI (podcasts, etc.)
                if not r.get('spotify_track_uri'):
                    continue
                # Skip very short plays (likely skips)
                if r.get('ms_played', 0) < config.min_play_ms:
                    continue
                    
                track_id = r['spotify_track_uri'].split(':')[-1]
                all_records.append({
                    'track_id': track_id,
                    'track_name': r.get('master_metadata_track_name', ''),
                    'artist_name': r.get('master_metadata_album_artist_name', ''),
                    'ts': r['ts'],
                    'ms_played': r['ms_played'],
                    'platform': r.get('platform', 'unknown'),
                    'country': r.get('conn_country', 'unknown'),
                    'shuffle': r.get('shuffle', False),
                    'offline': r.get('offline', False),
                    # Improved skip detection: 'skipped' is boolean, 'reason_end' gives context
                    'skipped': r.get('skipped', False) or r.get('reason_end') in ['fwdbtn', 'backbtn'], 
                    'reason_end': r.get('reason_end'),
                })
    
    df = pd.DataFrame(all_records)
    df['ts'] = pd.to_datetime(df['ts'])
    
    # Verbose logging: date range and unique counts
    logger.info(f"  -> Loaded {len(df):,} streams from {df['ts'].min().strftime('%Y-%m-%d')} to {df['ts'].max().strftime('%Y-%m-%d')}")
    logger.info(f"  -> {df['track_id'].nunique():,} unique tracks, {df['artist_name'].nunique():,} unique artists")
    
    # Platform/country breakdown
    platforms = df['platform'].value_counts().head(3)
    countries = df['country'].value_counts().head(3)
    logger.info(f"  -> Platforms: {dict(platforms)}")
    logger.info(f"  -> Countries: {dict(countries)}")
    
    # Shuffle vs intentional listening
    shuffle_pct = df['shuffle'].mean() * 100
    logger.info(f"  -> Shuffle mode: {shuffle_pct:.0f}% of plays")
    
    return df

def get_played_track_ids_local(history_df: pd.DataFrame) -> set[str]:
    """Get all track IDs from local streaming history."""
    return set(history_df['track_id'].unique())

def get_saved_track_ids() -> set[str]:
    """Get IDs of tracks to exclude from recommendations.
    
    Uses local history if USE_LOCAL_HISTORY is True, otherwise API.
    """
    if USE_LOCAL_HISTORY:
        history_df = load_streaming_history()
        played_ids = get_played_track_ids_local(history_df)
        logger.info(f"  -> Found {len(played_ids)} played tracks to exclude")
        return played_ids
    
    # Original API-based approach
    logger.info("Analyze: Checking your saved tracks to exclude duplicates...")
    saved_ids: set[str] = set()
    offset = 0
    while True:
        results = get_spotify_client().current_user_saved_tracks(limit=50, offset=offset)
        if not results['items']:
            break
        for item in results['items']:
            saved_ids.add(item['track']['id'])
        offset += 50
        if offset >= SAVED_TRACKS_CAP:  # Cap to avoid slow startup
            break
    logger.info(f"  -> Found {len(saved_ids)} saved tracks to exclude")
    return saved_ids

def get_audio_features_from_db(track_ids: list[str]) -> pd.DataFrame:
    """Fetches all audio features from the local database.
    
    Uses shared connection for efficiency when called multiple times.
    """
    if not track_ids:
        return pd.DataFrame()
    
    con = DatabaseConnection.get_shared(AUDIO_FEATURES_DB_PATH)
    placeholders = ','.join(['?' for _ in track_ids])
    query = f"""
        SELECT track_id, danceability, energy, loudness, speechiness,
               acousticness, instrumentalness, liveness, valence, tempo,
               mode, key
        FROM track_audio_features
        WHERE track_id IN ({placeholders})
        AND null_response = 0
    """
    df = pd.read_sql_query(query, con, params=track_ids)
    return df

def save_audio_features(features_list: list[dict[str, Any]]) -> None:
    """Save fetched audio features to local database."""
    if not features_list:
        return
        
    logger.info(f"  -> Saving {len(features_list)} new audio features to database...")
    
    with DatabaseConnection(AUDIO_FEATURES_DB_PATH) as con:
        cursor = con.cursor()
        
        # Prepare data for insertion
        data_to_insert = []
        for f in features_list:
            if not f: continue
            data_to_insert.append((
                f['id'], f['danceability'], f['energy'], f['key'],
                f['loudness'], f['mode'], f['speechiness'], f['acousticness'],
                f['instrumentalness'], f['liveness'], f['valence'], f['tempo'],
                f['duration_ms'], f['time_signature'], 0 # null_response = 0
            ))
            
        try:
            cursor.executemany("""
                INSERT OR IGNORE INTO track_audio_features 
                (track_id, danceability, energy, key, loudness, mode, speechiness, 
                 acousticness, instrumentalness, liveness, valence, tempo, 
                 duration_ms, time_signature, null_response)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, data_to_insert)
            con.commit()
        except Exception as e:
            logger.error(f"Error saving audio features: {e}")


def train_mood_clusters(df: pd.DataFrame, sample_weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, list[ClusterStats], MinMaxScaler, np.ndarray]:
    """
    Core profiling logic: clusters audio features to find 'moods'.
    Reusable for both User History and Playlist Seeding.
    """
    if df.empty:
        raise ValueError("Cannot train clusters on empty dataframe")
        
    # Scale features
    scaler = MinMaxScaler()
    feature_matrix = df[FEATURE_COLS].values.astype(float)
    normalized_features = scaler.fit_transform(feature_matrix)
    
    # Apply global feature weights
    weights = np.array([config.feature_weights.get(col, 1.0) for col in FEATURE_COLS])
    weighted_features = normalized_features * weights
    
    # Default weights if None
    if sample_weights is None:
        sample_weights = np.ones(len(df))
        
    # KMeans Clustering
    n_clusters = min(config.n_clusters, len(df))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(weighted_features, sample_weight=sample_weights)
    
    cluster_centers = kmeans.cluster_centers_
    
    # Calculate cluster weights (how much of the history/playlist falls into this mood)
    labels = kmeans.labels_
    label_weights = pd.DataFrame({'label': labels, 'weight': sample_weights})
    total_weight = label_weights['weight'].sum()
    cluster_weights = label_weights.groupby('label')['weight'].sum() / total_weight
    cluster_weights = cluster_weights.reindex(range(n_clusters), fill_value=0).values
    
    # Calculate stats per cluster for range fitting
    df_normalized = pd.DataFrame(normalized_features, columns=FEATURE_COLS)
    df_normalized['cluster'] = labels
    
    cluster_stats = []
    for i in range(n_clusters):
        cluster_data = df_normalized[df_normalized['cluster'] == i]
        stats = {
            'mean': np.zeros(len(FEATURE_COLS)),
            'std': np.zeros(len(FEATURE_COLS)),
            'min': np.zeros(len(FEATURE_COLS)),
            'max': np.zeros(len(FEATURE_COLS)),
        }
        if not cluster_data.empty:
            stats['mean'] = (cluster_data[FEATURE_COLS].mean().values * weights)
            stats['std'] = (cluster_data[FEATURE_COLS].std().values * weights)
            stats['min'] = (cluster_data[FEATURE_COLS].min().values * weights)
            stats['max'] = (cluster_data[FEATURE_COLS].max().values * weights)
        else:
            # Fallback for empty cluster (shouldn't happen with min(n_clusters, len(df)))
            stats['mean'] = np.array([0.5] * len(FEATURE_COLS)) * weights
            stats['std'] = np.array([0.1] * len(FEATURE_COLS)) * weights
            stats['min'] = np.array([0.0] * len(FEATURE_COLS)) * weights
            stats['max'] = np.array([1.0] * len(FEATURE_COLS)) * weights
        cluster_stats.append(stats)
        
    # Describe clusters
    logger.info(f"  -> Detected {n_clusters} distinct listening moods:")
    for i in range(n_clusters):
        cluster_data = df_normalized[df_normalized['cluster'] == i][FEATURE_COLS]
        if not cluster_data.empty:
            energy_level = cluster_data['energy'].mean()
            valence_level = cluster_data['valence'].mean()
            mood = "upbeat" if valence_level > 0.5 else "moody"
            intensity = "high-energy" if energy_level > 0.5 else "chill"
            energy_range = f"{cluster_data['energy'].min():.0%}-{cluster_data['energy'].max():.0%}"
            logger.info(f"     Mood {i+1}: {intensity} + {mood} ({len(cluster_data)} tracks)")
        else:
            logger.info(f"     Mood {i+1}: (empty cluster)")
            
    return cluster_centers, cluster_weights, cluster_stats, scaler, weights

# Profile cache settings
_PROFILE_CACHE_DIR = os.path.expanduser('~/.cache/spot')
_PROFILE_CACHE_FILE = os.path.join(_PROFILE_CACHE_DIR, 'user_profile.pkl')
_PROFILE_CACHE_MAX_AGE_HOURS = 24

def _load_cached_profile() -> ProfileResult | None:
    """Load cached profile if it exists and is fresh."""
    import pickle
    
    if getattr(ARGS, 'no_cache', False):
        logger.info("  -> Profile cache disabled via --no-cache")
        return None
        
    if not os.path.exists(_PROFILE_CACHE_FILE):
        return None
        
    # Check age
    mtime = os.path.getmtime(_PROFILE_CACHE_FILE)
    age_hours = (datetime.now().timestamp() - mtime) / 3600
    
    if age_hours > _PROFILE_CACHE_MAX_AGE_HOURS:
        logger.info(f"  -> Profile cache expired ({age_hours:.1f}h old)")
        return None
        
    try:
        with open(_PROFILE_CACHE_FILE, 'rb') as f:
            cached = pickle.load(f)
        logger.info(f"  -> Loaded profile from cache ({age_hours:.1f}h old)")
        return cached
    except Exception as e:
        logger.warning(f"  -> Cache load failed: {e}")
        return None

def _save_profile_cache(profile: ProfileResult) -> None:
    """Save profile to cache."""
    import pickle
    
    try:
        os.makedirs(_PROFILE_CACHE_DIR, exist_ok=True)
        with open(_PROFILE_CACHE_FILE, 'wb') as f:
            pickle.dump(profile, f)
        logger.info(f"  -> Saved profile to cache")
    except Exception as e:
        logger.warning(f"  -> Cache save failed: {e}")

def get_user_profile_local() -> ProfileResult:
    """Build user profile from local streaming history instead of API.
    
    Uses actual timestamps for recency weighting (more granular than API's
    short/medium/long term buckets). Caches result for 24h to speed up subsequent runs.
    """
    # Try cache first
    cached = _load_cached_profile()
    if cached is not None:
        return cached
        
    logger.info("Analyze: Building profile from local streaming history...")
    
    history_df = load_streaming_history()
    
    # Date Filtering (Timeframe Control)
    if config.date_start:
        logger.info(f"  -> Filtering history: starting from {config.date_start}")
        history_df = history_df[history_df['ts'] >= pd.to_datetime(config.date_start).tz_localize(history_df['ts'].dt.tz)]
    
    if config.date_end:
        logger.info(f"  -> Filtering history: up to {config.date_end}")
        history_df = history_df[history_df['ts'] <= pd.to_datetime(config.date_end).tz_localize(history_df['ts'].dt.tz)]
        
    if history_df.empty:
        raise ValueError(f"No listening history found in the selected timeframe ({config.date_start} to {config.date_end})")
    
    # Calculate recency weight based on actual timestamp
    # More recent = higher weight, with exponential decay
    now = pd.Timestamp.now(tz='UTC')
    history_df['days_ago'] = (now - history_df['ts']).dt.days
    
    # Exponential decay: weight = RECENCY_DECAY ^ (-days/365)
    # 1 year ago = base weight, today = RECENCY_DECAY^1, 2 years ago = RECENCY_DECAY^-1
    history_df['recency_weight'] = RECENCY_DECAY ** (1 - history_df['days_ago'] / 365)
    
    # Aggregate by track: sum of (recency_weight * ms_played)
    track_scores = history_df.groupby('track_id').agg({
        'recency_weight': 'sum',
        'ms_played': 'sum',
        'artist_name': 'first',
        'track_name': 'first',
        'skipped': 'mean'
    }).rename(columns={'recency_weight': 'weight'})
    
    # Apply Smart Weights
    # 1. Skip Penalty
    if config.skip_penalty > 0:
        track_scores['weight'] *= (1.0 - (track_scores['skipped'] * config.skip_penalty))
        
    # 2. Repeat Boost
    play_counts = history_df.groupby('track_id').size()
    track_scores['play_count'] = play_counts
    if config.repeat_boost > 1.0:
        boost_mask = track_scores['play_count'] >= 5
        track_scores.loc[boost_mask, 'weight'] *= config.repeat_boost
    
    # Get top tracks by weighted play time
    track_scores['weighted_play'] = track_scores['weight'] * track_scores['ms_played']
    top_tracks = track_scores.nlargest(config.profile_tracks_limit, 'weighted_play')
    
    logger.info(f"  -> Top tracks: {len(top_tracks)} (from {len(track_scores)} unique)")
    
    # Log top 10 tracks used for profiling
    logger.info("  -> Your top 10 tracks (by weighted listen time):")
    for i, (tid, row) in enumerate(top_tracks.head(10).iterrows()):
        hours = row['ms_played'] / 3600000
        logger.info(f"     {i+1}. {row['track_name'][:40]} - {row['artist_name'][:25]} ({hours:.1f}h)")
    
    # Log top artists by total listen time
    artist_hours = history_df.groupby('artist_name')['ms_played'].sum() / 3600000
    top_artists_by_hours = artist_hours.nlargest(10)
    logger.info("  -> Your top 10 artists (by total listen time):")
    for i, (artist, hours) in enumerate(top_artists_by_hours.items()):
        logger.info(f"     {i+1}. {artist[:40]} ({hours:.1f}h)")
    
    # Get audio features
    all_track_ids = top_tracks.index.tolist()
    df = get_audio_features_from_db(all_track_ids)
    
    if df.empty:
        raise Exception("No audio features found for your top tracks in local database")
    
    logger.info(f"  -> Found audio features for {len(df)}/{len(all_track_ids)} tracks")
    
    # Build sample weights array matching df order
    sample_weights = np.array([
        top_tracks.loc[tid, 'weight'] if tid in top_tracks.index else 1.0 
        for tid in df['track_id']
    ])
    
    # --- CLUSTERING ---
    # Extract clustering logic into reusable function
    cluster_centers, cluster_weights, cluster_stats, scaler, weights = train_mood_clusters(df, sample_weights)
    
    # Get top artists from history and look up genres from local DB
    top_artists = history_df.groupby('artist_name').agg({
        'ms_played': 'sum',
        'recency_weight': 'sum'
    }).nlargest(20, 'ms_played')
    
    # Look up genres from local database
    artist_names = top_artists.index.tolist()
    genres = []
    with sqlite3.connect(DB_PATH) as con:
        for artist_name in artist_names[:10]:
            query = """
                SELECT ag.genre FROM artist_genres ag
                JOIN artists a ON ag.artist_rowid = a.rowid
                WHERE a.name = ? LIMIT 5
            """
            result = con.execute(query, [artist_name]).fetchall()
            genres.extend([r[0] for r in result])
    
    top_genres = pd.Series(genres).value_counts().head(20).index.tolist() if genres else []
    
    # Filter Blocked Genres
    if config.blocked_genres:
        logger.info(f"  -> Applying genre blocklist: {config.blocked_genres}")
        filtered_genres = []
        for g in top_genres:
            is_blocked = any(blocked.lower() in g.lower() for blocked in config.blocked_genres)
            if not is_blocked:
                filtered_genres.append(g)
        top_genres = filtered_genres[:5]
        top_genres = top_genres[:5]
        
    logger.info(f"  -> Top genres (after filters): {top_genres}")
    
    # For genre-based similar artists, we don't need API artist IDs
    # Return artist NAMES instead of IDs for local lookup
    top_artist_names = artist_names[:config.top_artists_limit]
    
    # Calculate global skip rates for all tracks in history
    # (Not just top tracks)
    track_skip_stats = history_df.groupby('track_id').agg({
        'skipped': 'mean',
        'ms_played': 'count'
    })
    # Filter to tracks with at least 2 plays to avoid noise
    significant_skips = track_skip_stats[track_skip_stats['ms_played'] >= 2]
    # Map track_id -> skip_rate
    skip_map = significant_skips['skipped'].to_dict()
    
    result = (cluster_centers, cluster_weights, cluster_stats, top_genres, top_artist_names, scaler, weights, skip_map)
    _save_profile_cache(result)
    return result

def get_playlist_profile(playlist_id: str) -> ProfileResult:
    """Generates a profile from a specific Spotify playlist."""
    logger.info(f"Analyze: Building profile from Spotify playlist: {playlist_id}...")
    
    sp = get_spotify_client()
    
    # 1. Fetch Playlist Tracks
    results = sp.playlist_tracks(playlist_id)
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
        
    logger.info(f"  -> Found {len(tracks)} tracks in playlist")
    
    # 2. Extract IDs and Metadata
    track_ids = []
    artist_names = []
    for t in tracks:
        if t['track'] and t['track']['id']:
            track_ids.append(t['track']['id'])
            if t['track']['artists']:
                artist_names.append(t['track']['artists'][0]['name'])
            else:
                artist_names.append("Unknown Artist")
            
    # 3. Get Audio Features (local db or api)
    df_features = get_audio_features_from_db(track_ids)
    
    missing_ids = [tid for tid in track_ids if tid not in df_features['track_id'].values]
    if missing_ids:
        logger.info(f"  -> Fetching audio features for {len(missing_ids)} missing tracks from API...")
        # Chunking for API calls
        for i in range(0, len(missing_ids), 100):
            batch = missing_ids[i:i+100]
            try:
                features_list = sp.audio_features(batch)
                features_list = [f for f in features_list if f] # Filter Nones
                if features_list:
                    save_audio_features(features_list) # Assuming save_audio_features is available
            except Exception as e:
                logger.error(f"Error fetching features for batch: {e}")
        
        # Reload full dataframe after fetching
        df_features = get_audio_features_from_db(track_ids)

    if df_features.empty:
        raise ValueError("No audio features found for this playlist.")
    
    logger.info(f"  -> Found audio features for {len(df_features)}/{len(track_ids)} tracks")

    # 4. Train Clusters
    # Playlist tracks are usually equally weighted representing the vibe
    cluster_centers, cluster_weights, cluster_stats, scaler, weights = train_mood_clusters(df_features, sample_weights=None)
    
    # 5. Top Artists / Genres (Approximation)
    # For playlist profiling, we don't have user's top genres directly.
    # We can try to infer from the playlist artists, but for simplicity,
    # we'll return an empty list for genres, which means genre-based candidates
    # will be skipped or rely on other sources.
    top_genres = [] 
    
    top_artist_names = pd.Series(artist_names).value_counts().head(config.top_artists_limit).index.tolist()
    
    # Playlist profiling doesn't have a skip map
    skip_map = {}
    
    return cluster_centers, cluster_weights, cluster_stats, top_genres, top_artist_names, scaler, weights, skip_map

def print_coverage_report():
    """Diagnostic tool: Checks percentage of history tracks that have cached audio features."""
    logger.info("Generating Database Coverage Report...")
    
    try:
        history_df = load_streaming_history()
    except FileNotFoundError as e:
        logger.error(f"Error loading history for coverage report: {e}")
        return

    all_track_ids = history_df['track_id'].unique().tolist()
    total_tracks = len(all_track_ids)
    
    if total_tracks == 0:
        logger.info("No listening history found to generate report.")
        return

    # Check DB
    con = DatabaseConnection.get_shared(AUDIO_FEATURES_DB_PATH)
    
    # Avoid SQLite variable limit by fetching all cached IDs and comparing sets in Python
    # This is efficient enough for <100k tracks
    cursor = con.execute("SELECT track_id FROM track_audio_features")
    cached_ids = set(row[0] for row in cursor.fetchall())
    
    # Calculate intersection
    in_history_and_cache = len(set(all_track_ids).intersection(cached_ids))
    
    percent = (in_history_and_cache / total_tracks) * 100
    
    logger.info("-" * 40)
    logger.info(f"Total Unique Tracks in History: {total_tracks}")
    logger.info(f"Cached Audio Features:        {in_history_and_cache}")
    logger.info(f"Missing Features:             {total_tracks - in_history_and_cache}")
    logger.info(f"Coverage:                     {percent:.1f}%")
    logger.info("-" * 40)
    if percent < 80:
        logger.info("Tip: Run standard recommendation once to fetch missing features.")

def get_rediscovery_candidates(limit: int = 50) -> list[TrackCandidate]:
    """Find tracks loved in the past but not played recently (Rediscovery Mode)."""
    if not USE_LOCAL_HISTORY:
        logger.warning("Rediscovery Mode requires Local History (skipped)")
        return []
    
    logger.info("Analyze: Searching for 'Rediscovery' gems (old favorites)...")
    history_df = load_streaming_history()
    
    cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=config.rediscovery_days)
    
    # Tracks played recently (to exclude)
    recent_mask = history_df['ts'] > cutoff
    recent_track_ids = set(history_df[recent_mask]['track_id'].unique())
    
    # Old tracks (played before cutoff)
    old_mask = history_df['ts'] <= cutoff
    old_history = history_df[old_mask]
    
    if old_history.empty:
        logger.info("  -> No listening history found beyond the rediscovery period")
        return []
        
    # Group by track and count plays/time
    old_stats = old_history.groupby('track_id').agg({
        'ms_played': 'sum',
        'track_name': 'first',
        'artist_name': 'first'
    })
    
    # Filter: Must be "Favorites" (>10 min listening time typical for ~3 songs)
    # and NOT in recent tracks
    favorites = old_stats[
        (old_stats['ms_played'] > 600000) & 
        (~old_stats.index.isin(recent_track_ids))
    ]
    
    if favorites.empty:
        logger.info("  -> No old favorites found that haven't been played recently")
        return []
        
    # Pick top N by listening time (most loved)
    top_rediscovery = favorites.nlargest(limit, 'ms_played')
    
    results = []
    
    for tid, row in top_rediscovery.iterrows():
        results.append({
            'id': tid,
            'name': row['track_name'],
            'artist': row['artist_name'],
            'popularity': 50 # Default
        })
        
    logger.info(f"  -> Found {len(results)} rediscovery candidates")
    return results

def get_user_profile() -> ProfileResult:
    """Fetches user's top tracks and clusters them into distinct listening moods.
    
    Uses local history if USE_LOCAL_HISTORY is True, otherwise Spotify API.
    """
    if USE_LOCAL_HISTORY:
        return get_user_profile_local()
    
    # Original API-based approach
    logger.info("Analyze: Fetching your top tracks from Spotify...")
    
    # Recency weight multipliers (higher = more influence on clusters)
    time_range_weights = {
        'short_term': RECENCY_DECAY ** 2,   # e.g., 4.0 with RECENCY_DECAY=2.0
        'medium_term': RECENCY_DECAY ** 1,  # e.g., 2.0
        'long_term': RECENCY_DECAY ** 0,    # Always 1.0 (base weight)
    }
    
    # Get tracks from multiple time ranges, tracking their recency weight
    # Prefer short_term association when deduplicating
    track_id_to_weight = {}  # track_id -> recency weight
    all_tracks = []
    
    for time_range in ['short_term', 'medium_term', 'long_term']:
        top_tracks = get_spotify_client().current_user_top_tracks(limit=50, time_range=time_range)
        weight = time_range_weights[time_range]
        for t in top_tracks['items']:
            if t['id'] not in track_id_to_weight:
                # First time seeing this track - use the most recent time_range's weight
                track_id_to_weight[t['id']] = weight
                all_tracks.append(t)
            # If already seen (from a more recent time range), keep the higher weight
    
    all_track_ids = list(track_id_to_weight.keys())
    logger.info(f"  -> Collected {len(all_track_ids)} unique tracks across time ranges")
    
    # Show recency weighting info
    if RECENCY_DECAY != 1.0:
        short_count = sum(1 for w in track_id_to_weight.values() if w == time_range_weights['short_term'])
        med_count = sum(1 for w in track_id_to_weight.values() if w == time_range_weights['medium_term'])
        long_count = sum(1 for w in track_id_to_weight.values() if w == time_range_weights['long_term'])
        logger.info(f"  -> Recency weighting: short={short_count} (x{time_range_weights['short_term']:.1f}), "
              f"medium={med_count} (x{time_range_weights['medium_term']:.1f}), "
              f"long={long_count} (x{time_range_weights['long_term']:.1f})")
    
    # Get Audio Features from LOCAL database
    logger.info("Analyze: Looking up audio features in local database...")
    df = get_audio_features_from_db(all_track_ids)
    
    if df.empty:
        raise Exception("No audio features found for your top tracks in local database")
    
    logger.info(f"  -> Found audio features for {len(df)}/{len(all_track_ids)} tracks")
    
    # Build sample weights array matching df order
    sample_weights = np.array([track_id_to_weight.get(tid, 1.0) for tid in df['track_id']])
    
    # --- CLUSTERING ---
    cluster_centers, cluster_weights, cluster_stats, scaler, weights = train_mood_clusters(df, sample_weights)
    
    # Get Top Artists and Genres (use all collected tracks)
    # Filter out tracks with no artists to avoid IndexError
    artist_ids = list(set([t['artists'][0]['id'] for t in all_tracks if t.get('artists')]))[:50]
    artists = get_spotify_client().artists(artist_ids) if artist_ids else {'artists': []}
    genres = []
    for artist in artists['artists']:
        genres.extend(artist['genres'])
    
    # Return top genres and artist IDs for collaborative filtering
    top_genres = pd.Series(genres).value_counts().head(5).index.tolist()
    top_artist_ids = artist_ids[:config.top_artists_limit]  # Top artists for related artist lookup
    
    # API mode doesn't support skip map yet (requires local history)
    skip_map = {}
    
    return cluster_centers, cluster_weights, cluster_stats, top_genres, top_artist_ids, scaler, weights, skip_map

def query_candidates(genres: list[str], saved_ids: set[str], limit: int = 100) -> list[TrackCandidate]:
    """Queries local DB using staged approach for speed.
    
    Genre matching controlled by GENRE_MATCH_MODE:
    - 'exact': Only exact matches
    - 'prefix': Exact + prefix expansion (recommended)
    - 'fuzzy': Broad LIKE %% matching
    """
    logger.info(f"Discovery: Searching for tracks in {genres}...")
    logger.info(f"  -> Genre match mode: {GENRE_MATCH_MODE}")
    
    with sqlite3.connect(DB_PATH) as con:
        # STAGE 1: Find artist rowids matching genres
        logger.info("  -> Stage 1: Finding artists by genre...")
        
        artist_rowids = set()
        matched_genres = {'exact': [], 'prefix': [], 'fuzzy': []}
        
        # Always try exact match first (fast, uses index)
        placeholders = ','.join(['?' for _ in genres])
        exact_query = f"SELECT DISTINCT artist_rowid, genre FROM artist_genres WHERE genre IN ({placeholders})"
        cursor = con.execute(exact_query, genres)
        for row in cursor.fetchall():
            artist_rowids.add(row[0])
            if row[1] not in matched_genres['exact']:
                matched_genres['exact'].append(row[1])
        
        # Prefix expansion if mode allows and need more results
        if GENRE_MATCH_MODE in ('prefix', 'fuzzy') and len(artist_rowids) < 300:
            for g in genres:
                if g not in matched_genres['exact']:  # Only expand unmatched genres
                    prefix_query = "SELECT DISTINCT artist_rowid, genre FROM artist_genres WHERE genre LIKE ? LIMIT 100"
                    cursor = con.execute(prefix_query, [f"{g}%"])  # Prefix only, no leading %
                    for row in cursor.fetchall():
                        artist_rowids.add(row[0])
                        if row[1] not in matched_genres['prefix']:
                            matched_genres['prefix'].append(row[1])
        
        # Fuzzy fallback if mode is 'fuzzy' and still need more
        if GENRE_MATCH_MODE == 'fuzzy' and len(artist_rowids) < 200:
            for g in genres[:2]:  # Limit fuzzy to first 2 genres
                fuzzy_query = "SELECT DISTINCT artist_rowid, genre FROM artist_genres WHERE genre LIKE ? LIMIT 100"
                cursor = con.execute(fuzzy_query, [f"%{g}%"])  # Full fuzzy
                for row in cursor.fetchall():
                    artist_rowids.add(row[0])
                    if row[1] not in matched_genres['fuzzy']:
                        matched_genres['fuzzy'].append(row[1])
        
        # Log what matched
        if matched_genres['exact']:
            logger.info(f"  -> Exact matches: {matched_genres['exact'][:5]}{'...' if len(matched_genres['exact']) > 5 else ''}")
        if matched_genres['prefix']:
            logger.info(f"  -> Prefix matches: {matched_genres['prefix'][:5]}{'...' if len(matched_genres['prefix']) > 5 else ''}")
        if matched_genres['fuzzy']:
            logger.info(f"  -> Fuzzy matches: {matched_genres['fuzzy'][:5]}{'...' if len(matched_genres['fuzzy']) > 5 else ''}")
        
        if not artist_rowids:
            logger.info("  -> No matching artists found")
            return []
        
        logger.info(f"  -> Found {len(artist_rowids)} matching artists")
        
        # STAGE 2: Get tracks from these artists (fast with index)
        logger.info("  -> Stage 2: Finding tracks from these artists...")
        
        # Build exclusion set (just track IDs, not in the main query)
        exclude_set = set(saved_ids) if saved_ids else set()
        
        # Random sample of artist_rowids to keep query fast
        artist_list = list(artist_rowids)
        artist_sample = random.sample(artist_list, min(150, len(artist_list)))
        placeholders = ','.join(['?' for _ in artist_sample])
        
        # Simpler query without ORDER BY RANDOM (done in Python)
        sql_select = "SELECT t.id, t.name, art.name as artist_name, t.popularity, alb.release_date"
        sql_from = "FROM tracks t JOIN track_artists ta ON t.rowid = ta.track_rowid JOIN artists art ON ta.artist_rowid = art.rowid JOIN albums alb ON t.album_rowid = alb.rowid"
        sql_where = f"WHERE ta.artist_rowid IN ({placeholders}) AND t.popularity BETWEEN {config.popularity_min} AND {config.popularity_max}"
        sql_limit = f"LIMIT {limit * 15}"
        
        sql_params = list(artist_sample)
        
        # Era Logic
        if config.release_year_min:
            sql_where += " AND alb.release_date >= ?"
            sql_params.append(f"{config.release_year_min}-01-01")
        if config.release_year_max:
            sql_where += " AND alb.release_date <= ?"
            sql_params.append(f"{config.release_year_max}-12-31")
            
        track_query = f"{sql_select} {sql_from} {sql_where} {sql_limit}"
        
        df_tracks = pd.read_sql_query(track_query, con, params=sql_params)
    
    # Filter out saved tracks in Python (faster than SQL NOT IN with large list)
    if exclude_set:
        df_tracks = df_tracks[~df_tracks['id'].isin(exclude_set)]
    
    # Filter to tracks that have audio features
    track_ids = df_tracks['id'].tolist()
    df_features = get_audio_features_from_db(track_ids)
    
    # Merge
    df_merged = df_tracks[df_tracks['id'].isin(df_features['track_id'])]
    
    # === NEW: FILTER BEFORE LIMITING ===
    # 1. Align features with candidates
    # Create candidates list first
    df_merged = df_merged.rename(columns={'artist_name': 'artist'})
    candidates_full = df_merged[['id', 'name', 'artist', 'popularity']].to_dict('records')
    
    # 2. Align df_features to match candidates_full order
    feat_map = {row['track_id']: idx for idx, row in df_features.iterrows()}
    valid_candidates = []
    valid_indices = []
    
    for c in candidates_full:
         if c['id'] in feat_map:
             valid_candidates.append(c)
             valid_indices.append(feat_map[c['id']])
             
    df_features_aligned = df_features.iloc[valid_indices].reset_index(drop=True)
    
    # 3. Calculate Camelot Codes
    camelot_codes = []
    if 'key' in df_features_aligned.columns and 'mode' in df_features_aligned.columns:
         for _, row in df_features_aligned.iterrows():
             camelot_codes.append(get_camelot_code(row['key'], row['mode']))
    else:
         camelot_codes = ['??'] * len(df_features_aligned)
         
    # 4. Apply Filters
    logger.info(f"  -> Applying audio filters to {len(valid_candidates)} potential candidates...")
    _, candidates_filtered, _ = filter_candidates_by_config(
        df_features_aligned, valid_candidates, camelot_codes
    )
    
    # 5. Sample and Limit
    if len(candidates_filtered) > limit:
        candidates_final = random.sample(candidates_filtered, limit)
        logger.info(f"  -> Sampled {limit} from {len(candidates_filtered)} valid candidates")
    else:
        candidates_final = candidates_filtered
        logger.info(f"  -> Found {len(candidates_final)} valid candidates (below limit of {limit})")

    return candidates_final

def get_related_artist_tracks_api(top_artist_ids: list[str], saved_ids: set[str], limit: int = 50) -> list[TrackCandidate]:
    """Gets tracks from artists similar to user's favorites via Spotify API.
    
    Fallback chain:
    1. Try related artists API for top N artists
    2. If too few results, try more artists from the list
    3. If still failing, use Spotify's recommendations API
    """
    logger.info("Discovery: Finding tracks from related artists (API)...")
    
    related_artist_ids = set()
    exclude_set = set(saved_ids) if saved_ids else set()
    
    # FALLBACK 1: Try progressively more artists until we get results
    batch_sizes = [10, 20, len(top_artist_ids)]  # Escalating batches
    
    for batch_size in batch_sizes:
        artists_to_try = top_artist_ids[:batch_size]
        
        for artist_id in artists_to_try:
            try:
                related = get_spotify_client().artist_related_artists(artist_id)
                for artist in related['artists'][:5]:
                    related_artist_ids.add(artist['id'])
            except Exception:
                continue
        
        if len(related_artist_ids) >= 10:
            break  # Got enough related artists
        logger.info(f"  -> Batch of {batch_size} artists yielded {len(related_artist_ids)} related, trying more...")
    
    logger.info(f"  -> Found {len(related_artist_ids)} related artists")
    
    # Get top tracks from related artists
    tracks = []
    
    if related_artist_ids:
        for artist_id in list(related_artist_ids)[:30]:
            try:
                top_tracks = get_spotify_client().artist_top_tracks(artist_id, country='US')
                for t in top_tracks['tracks'][:3]:
                    if t['id'] not in exclude_set and t.get('artists'):
                        tracks.append({
                            'id': t['id'],
                            'name': t['name'],
                            'artist': t['artists'][0]['name'],
                            'popularity': t['popularity']
                        })
            except Exception as e:
                logger.info(f"    -> Skipped artist {artist_id}: {e}")
                continue
    
    # FALLBACK 2: If still no tracks, use Spotify recommendations API
    if len(tracks) < 10 and top_artist_ids:
        logger.info("  -> Few related tracks, trying Spotify recommendations API...")
        try:
            # Use up to 5 seed artists (API limit)
            seed_artists = top_artist_ids[:5]
            recs = get_spotify_client().recommendations(seed_artists=seed_artists, limit=50)
            for t in recs['tracks']:
                if t['id'] not in exclude_set and t.get('artists'):
                    tracks.append({
                        'id': t['id'],
                        'name': t['name'],
                        'artist': t['artists'][0]['name'],
                        'popularity': t['popularity']
                    })
            logger.info(f"  -> Spotify recommendations added {len(recs['tracks'])} tracks")
        except Exception as e:
            logger.info(f"  -> Recommendations API failed: {e}")
    
    # Early return if no tracks found
    if not tracks:
        logger.info("  -> Found 0 collaborative tracks")
        return []
    
    # Filter to tracks with audio features in local DB
    track_ids = [t['id'] for t in tracks]
    df_features = get_audio_features_from_db(track_ids)
    
    # Keep only tracks we have features for
    if df_features.empty:
        logger.info("  -> Found 0 collaborative tracks with audio features")
        return []
    
    valid_ids = set(df_features['track_id'].tolist())
    tracks = [t for t in tracks if t['id'] in valid_ids][:limit]
    
    logger.info(f"  -> Found {len(tracks)} collaborative tracks with audio features")
    return tracks

def get_similar_artist_tracks_local(top_artist_names: list[str], saved_ids: set[str], limit: int = 50) -> list[dict[str, Any]]:
    """Find tracks from similar artists using genre overlap (local DB only).
    
    Uses artist_genres table to find artists who share genres with user's favorites.
    """
    logger.info("Discovery: Finding similar artists by genre overlap (local DB)...")
    
    exclude_set = set(saved_ids) if saved_ids else set()
    
    with sqlite3.connect(DB_PATH) as con:
        # Step 1: Get rowids and genres for user's top artists
        artist_rowids = []
        user_genres = set()
        
        for name in top_artist_names[:10]:
            result = con.execute(
                "SELECT rowid FROM artists WHERE name = ? LIMIT 1", [name]
            ).fetchone()
            if result:
                artist_rowids.append(result[0])
                
                genre_result = con.execute(
                    "SELECT genre FROM artist_genres WHERE artist_rowid = ?", [result[0]]
                ).fetchall()
                user_genres.update(g[0] for g in genre_result)
        
        if not user_genres:
            logger.info("  -> No genres found for your top artists")
            return []
        
        logger.info(f"  -> Your genres: {list(user_genres)[:5]}...")
        
        # Step 2: Find OTHER artists with those genres, ranked by overlap count
        genre_placeholders = ','.join(['?' for _ in user_genres])
        rowid_placeholders = ','.join(['?' for _ in artist_rowids]) if artist_rowids else '0'
        
        similar_query = f"""
            SELECT a.rowid, a.name, COUNT(DISTINCT ag.genre) as overlap_count
            FROM artists a
            JOIN artist_genres ag ON a.rowid = ag.artist_rowid
            WHERE ag.genre IN ({genre_placeholders})
            AND a.rowid NOT IN ({rowid_placeholders})
            GROUP BY a.rowid
            ORDER BY overlap_count DESC
            LIMIT 50
        """
        
        params = list(user_genres) + artist_rowids
        similar_artists = con.execute(similar_query, params).fetchall()
        
        if not similar_artists:
            logger.info("  -> No similar artists found")
            return []
        
        logger.info(f"  -> Found {len(similar_artists)} similar artists by genre")
        
        # Step 3: Get tracks from these artists
        similar_rowids = [r[0] for r in similar_artists]
        track_placeholders = ','.join(['?' for _ in similar_rowids])
        
        track_query = f"""
            SELECT t.id, t.name, a.name as artist_name, t.popularity
            FROM tracks t
            JOIN track_artists ta ON t.rowid = ta.track_rowid
            JOIN artists a ON ta.artist_rowid = a.rowid
            WHERE ta.artist_rowid IN ({track_placeholders})
            AND t.popularity BETWEEN {config.similar_popularity_min} AND {config.similar_popularity_max}
            LIMIT {limit * 10}
        """
        
        df_tracks = pd.read_sql_query(track_query, con, params=similar_rowids)
    
    if df_tracks.empty:
        logger.info("  -> No tracks found from similar artists")
        return []
    
    # Filter out already-played tracks
    if exclude_set:
        df_tracks = df_tracks[~df_tracks['id'].isin(exclude_set)]
    
    # Filter to tracks with audio features
    track_ids = df_tracks['id'].tolist()
    df_features = get_audio_features_from_db(track_ids)
    
    if df_features.empty:
        logger.info("  -> No similar artist tracks with audio features")
        return []
    
    valid_ids = set(df_features['track_id'].tolist())
    df_tracks = df_tracks[df_tracks['id'].isin(valid_ids)]
    
    # === NEW: FILTER BEFORE LIMITING ===
    # 1. Create candidates list
    df_tracks = df_tracks.rename(columns={'artist_name': 'artist'})
    candidates_full = df_tracks[['id', 'name', 'artist', 'popularity']].to_dict('records')
    
    # 2. Align df_features
    feat_map = {row['track_id']: idx for idx, row in df_features.iterrows()}
    valid_candidates = []
    valid_indices = []
    
    for c in candidates_full:
         if c['id'] in feat_map:
             valid_candidates.append(c)
             valid_indices.append(feat_map[c['id']])
             
    df_features_aligned = df_features.iloc[valid_indices].reset_index(drop=True)
    
    # 3. Camelot
    camelot_codes = []
    if 'key' in df_features_aligned.columns and 'mode' in df_features_aligned.columns:
         for _, row in df_features_aligned.iterrows():
             camelot_codes.append(get_camelot_code(row['key'], row['mode']))
    else:
         camelot_codes = ['??'] * len(df_features_aligned)
         
    # 4. Filter
    logger.info(f"  -> Applying audio filters to {len(valid_candidates)} similar tracks...")
    _, candidates_filtered, _ = filter_candidates_by_config(
        df_features_aligned, valid_candidates, camelot_codes
    )
    
    # 5. Sample
    if len(candidates_filtered) > limit:
        candidates_final = random.sample(candidates_filtered, limit)
        logger.info(f"  -> Sampled {limit} from {len(candidates_filtered)} similar tracks")
    else:
        candidates_final = candidates_filtered
        logger.info(f"  -> Found {len(candidates_final)} similar tracks")
        
    return candidates_final

def get_similar_tracks(top_artists: list[str], saved_ids: set[str], limit: int = 50) -> list[TrackCandidate]:
    """Get tracks from similar artists - dispatches to local or API based on config.
    
    Args:
        top_artists: Either list of artist IDs (API mode) or artist names (local mode)
    """
    if USE_API_RELATED_ARTISTS:
        return get_related_artist_tracks_api(top_artists, saved_ids, limit)
    else:
        return get_similar_artist_tracks_local(top_artists, saved_ids, limit)

def get_serendipity_tracks(exclude_genres: list[str], saved_ids: set[str], limit: int = 5) -> list[TrackCandidate]:
    """Gets random 'wild card' tracks from outside user's typical genres.
    
    These bypass similarity scoring to inject variety and break echo chambers.
    """
    if SERENDIPITY_SLOTS <= 0:
        return []
    
    logger.info(f"Serendipity: Finding {limit} wild card tracks outside your bubble...")
    
    exclude_set = set(saved_ids) if saved_ids else set()
    
    # Build exclusion pattern for genres (sanitized to prevent SQL injection)
    # Use parameterized NOT LIKE with safe genre strings
    genre_params = []
    genre_conditions = []
    for g in exclude_genres[:3]:  # Use top 3 genres for exclusion
        # Sanitize: keep only alphanumeric, spaces, and hyphens
        safe_genre = ''.join(c for c in g if c.isalnum() or c in ' -')
        if safe_genre:
            genre_conditions.append("genre NOT LIKE ?")
            genre_params.append(f"%{safe_genre}%")
    
    exclusion_clause = " AND ".join(genre_conditions) if genre_conditions else "1=1"
    
    # Get random artists from different genres
    query = f"""
        SELECT DISTINCT ag.artist_rowid, ag.genre
        FROM artist_genres ag
        WHERE {exclusion_clause}
        LIMIT 500
    """
    
    results = [] # Initialize results list
    try:
        with sqlite3.connect(DB_PATH) as con:
            cursor = con.execute(query, genre_params)
            rows = cursor.fetchall()
            
            if not rows:
                logger.info("  -> No wild card artists found")
                return []
            
            # Random sample of artist rowids
            artist_sample = random.sample(rows, min(50, len(rows)))
            artist_rowids = [r[0] for r in artist_sample]
            sampled_genres = list(set(r[1] for r in artist_sample))[:5]
            
            logger.info(f"  -> Sampling from genres: {sampled_genres}")
            
            # Exclude genres via NOT EXISTS (more reliable than NOT LIKE join)
            # Ensure exclude_genres is not empty for the IN clause
            if not exclude_genres:
                exclusion_placeholders = 'NULL' # No genres to exclude
                exclude_genres_for_query = []
            else:
                exclusion_placeholders = ",".join("?" for _ in exclude_genres)
                exclude_genres_for_query = exclude_genres
            
            sql = f"""
        SELECT t.id, t.name, a.name, t.popularity
        FROM tracks t
        JOIN track_artists ta ON t.rowid = ta.track_rowid
        JOIN artists a ON ta.artist_rowid = a.rowid
        JOIN albums alb ON t.album_rowid = alb.rowid
        WHERE t.popularity BETWEEN ? AND ?
        AND t.duration_ms >= ?
        AND NOT EXISTS (
             SELECT 1 FROM artist_genres ag 
             WHERE ag.artist_rowid = a.rowid
             AND ag.genre IN ({exclusion_placeholders})
        )
    """
    
            params = [config.serendipity_popularity_min, config.serendipity_popularity_max, config.min_play_ms]
            params.extend(exclude_genres_for_query)
            
            # Era Filtering
            if config.release_year_min:
                sql += " AND alb.release_date >= ?"
                params.append(f"{config.release_year_min}-01-01")
            if config.release_year_max:
                sql += " AND alb.release_date <= ?"
                params.append(f"{config.release_year_max}-12-31")
        
            # Fetch more than needed and sample in Python (avoids slow ORDER BY RANDOM)
            sql += " LIMIT ?"
            params.append(limit * 100)  # Fetch 100x to have pool for random sample
            
            # Use the shared connection for the final track query
            con_shared = DatabaseConnection.get_shared(DB_PATH)
            rows = con_shared.execute(sql, params).fetchall()
            
            # Filter out saved tracks and sample
            valid_rows = [r for r in rows if r[0] not in saved_ids]
            if valid_rows:
                sampled = random.sample(valid_rows, min(limit, len(valid_rows)))
                for r in sampled:
                    results.append({
                        'id': r[0],
                        'name': r[1],
                        'artist': r[2],
                        'popularity': r[3],
                        'recommended_from': 'Serendipity'
                    })
            
    except Exception as e:
        logger.error(f"Error in get_serendipity_tracks: {e}")
        return []

    return results
        

        





def score_candidates_vectorized(
    weighted_matrix: np.ndarray,
    cluster_centers: np.ndarray,
    cluster_weights: np.ndarray,
    cluster_stats: list[dict]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score all candidates against all clusters using vectorized operations.
    
    Args:
        weighted_matrix: (N_candidates, N_features) array
        cluster_centers: (N_clusters, N_features) array
        cluster_weights: (N_clusters,) array
        cluster_stats: List of dicts with 'mean', 'std' arrays
        
    Returns:
        Tuple of (final_scores, best_mood_indices, range_fits)
    """
    # 1. Calculate Cosine Similarity (N_cand, N_clust)
    # This is the heavy lifting, done in C via sklearn
    sim_matrix = cosine_similarity(weighted_matrix, cluster_centers)
    
    # 2. Calculate Range Penalties (N_cand, N_clust) using broadcasting
    # Convert stats to arrays: (1, N_clust, N_features)
    # Ensure cluster_stats are correctly structured for this
    means = np.stack([c['mean'] for c in cluster_stats]).reshape(1, len(cluster_stats), -1)
    stds = np.stack([c['std'] for c in cluster_stats]).reshape(1, len(cluster_stats), -1)
    
    # Avoid div by zero
    stds = np.where(stds > 0.01, stds, 0.01)
    
    # Expand candidates: (N_cand, 1, N_features)
    cand_expanded = weighted_matrix[:, np.newaxis, :]
    
    # Calculate Z-scores: (N_cand, N_clust, N_features)
    z_scores = np.abs((cand_expanded - means) / stds)
    
    # Cap Z-scores at 3.0 and take mean across features -> (N_cand, N_clust)
    avg_z = np.clip(z_scores, 0, 3).mean(axis=2)
    range_penalties = avg_z * RANGE_PENALTY_STRENGTH
    
    # 3. Apply Penalties and Weights
    # Adjusted similarity: (N_cand, N_clust)
    adjusted_sim = np.maximum(0, sim_matrix * (1 - range_penalties))
    
    # Weighted similarity: (N_cand, N_clust)
    weighted_sim = adjusted_sim * cluster_weights.reshape(1, -1)
    
    # 4. Final Scores
    # Best single match for each candidate
    best_sim_indices = np.argmax(adjusted_sim, axis=1) # (N_cand,)
    best_sim_values = adjusted_sim[np.arange(len(adjusted_sim)), best_sim_indices]
    
    # Weighted average match
    avg_weighted_score = weighted_sim.sum(axis=1)
    
    # Combine: 70% best match, 30% aggregate match
    final_scores = (best_sim_values * 0.7) + (avg_weighted_score * 0.3)
    final_scores = np.clip(final_scores, 0.0, 1.0)
    
    # Get range fit for the best cluster
    best_penalties = range_penalties[np.arange(len(range_penalties)), best_sim_indices]
    range_fits = 1.0 - best_penalties
    
    return final_scores, best_sim_indices, range_fits


def filter_candidates_by_config(
    df_features: pd.DataFrame, 
    candidates: list[TrackCandidate], 
    camelot_codes: list[str]
) -> tuple[pd.DataFrame, list[TrackCandidate], list[str]]:
    """Apply all configuration-based audio filters to candidates.
    
    Returns filtered dataframe, candidate list, and camelot codes, all aligned.
    """
    if df_features.empty:
        return df_features, [], []
        
    mask = pd.Series([True] * len(df_features), index=df_features.index)
    
    # Speechiness Filter
    if config.max_speechiness is not None:
        mask &= (df_features['speechiness'] <= config.max_speechiness)
        
    # Acoustic Option
    if config.acoustic_option == 'acoustic':
        mask &= (df_features['acousticness'] >= 0.7)
    elif config.acoustic_option == 'electric':
        mask &= (df_features['acousticness'] <= 0.3)
        
    # Liveness Option
    if config.liveness_option == 'live':
        mask &= (df_features['liveness'] >= 0.8)
    elif config.liveness_option == 'studio':
        mask &= (df_features['liveness'] <= 0.2)
        
    # Tempo (BPM) Filter (Smart BPM support)
    if config.min_tempo is not None or config.max_tempo is not None:
        min_t = config.min_tempo or 0
        max_t = config.max_tempo or 999
        
        # Add fuzziness if exact match requested (floats are rarely exact)
        if config.min_tempo is not None and config.max_tempo is not None:
            if config.min_tempo == config.max_tempo:
                min_t -= 1.0
                max_t += 1.0
        
        if config.smart_bpm:
            # Allow: Standard range OR Double range OR Half range
            # e.g. 120-130 -> allows 120-130, 240-260, 60-65
            bpm_mask = (df_features['tempo'] >= min_t) & (df_features['tempo'] <= max_t)
            bpm_mask |= (df_features['tempo'] >= min_t * 2) & (df_features['tempo'] <= max_t * 2)
            bpm_mask |= (df_features['tempo'] >= min_t / 2) & (df_features['tempo'] <= max_t / 2)
            mask &= bpm_mask
        else:
            # Strict range
            if config.min_tempo: mask &= (df_features['tempo'] >= min_t)
            if config.max_tempo: mask &= (df_features['tempo'] <= max_t)

    # Vocal/Instrumental Filter (Speechiness/Instrumentalness combo)
    if config.vocal_mode == 'instrumental_only':
        mask &= (df_features['instrumentalness'] > 0.5)
    elif config.vocal_mode == 'vocal_only':
        mask &= (df_features['instrumentalness'] < 0.5)

    # Key/Mode Filter (Raw)
    if config.mode_filter == 'minor':
        if 'mode' in df_features.columns: mask &= (df_features['mode'] == 0)
    elif config.mode_filter == 'major':
        if 'mode' in df_features.columns: mask &= (df_features['mode'] == 1)

    # DJ Mode / Target Camelot Filter
    if config.target_camelot:
        compatible_keys = get_compatible_camelot_keys(config.target_camelot)
        # Filter mask based on calculated codes
        camelot_mask = pd.Series([c in compatible_keys for c in camelot_codes], index=df_features.index)
        mask &= camelot_mask
        
    if not mask.all():
         logger.info(f"  -> Audio filters removed {len(df_features) - mask.sum()} candidates")
         df_filtered = df_features[mask].copy()
         
         # Filter candidates and camelot_codes maintaining alignment
         # Using list comprehension with enumerate + iloc check on the mask
         # Note: mask is a Series with same index as df_features. 
         # Since candidates/codes list are 0-indexed aligned with df_features rows:
         valid_indices = [i for i in range(len(df_features)) if mask.iloc[i]]
         
         candidates_filtered = [candidates[i] for i in valid_indices]
         camelot_codes_filtered = [camelot_codes[i] for i in valid_indices]
         
         return df_filtered, candidates_filtered, camelot_codes_filtered
    
    return df_features, candidates, camelot_codes


def recommend(
    cluster_centers: np.ndarray,
    cluster_weights: np.ndarray, 
    cluster_stats: list[ClusterStats],
    candidates: list[TrackCandidate],
    scaler: MinMaxScaler,
    weights: np.ndarray,
    skip_map: dict[str, float] = {}
) -> list[TrackCandidate]:
    """Ranks candidates by best match to any listening mood cluster using vectorized scoring.
    
    Args:
        candidates: List of dicts with 'id', 'name', 'artist', 'popularity' keys
        
    Returns:
        List of dicts sorted by score descending
    """
    logger.info(f"Ranking: Analyzing {len(candidates)} candidates against {len(cluster_centers)} mood clusters...")
    
    candidate_ids = [c['id'] for c in candidates]
    df_features = get_audio_features_from_db(candidate_ids)
    
    if df_features.empty:
        logger.warning("  -> No audio features found for candidates")
        return []
    
    # Align df_features with candidates list (DB might return different order or missing rows)
    # Create lookup map
    feat_map = {row['track_id']: idx for idx, row in df_features.iterrows()}
    
    # Filter candidates to those we have features for and keep order consistent with matrix
    valid_candidates = []
    valid_indices = []
    
    for c in candidates:
        if c['id'] in feat_map:
            valid_candidates.append(c)
            valid_indices.append(feat_map[c['id']])
            
            
    # Reorder features dataframe to match valid_candidates order
    df_features = df_features.iloc[valid_indices]
    
    # Pre-calculate Camelot Codes for all candidates
    camelot_codes = []
    if 'key' in df_features.columns and 'mode' in df_features.columns:
        for idx, row in df_features.iterrows():
            camelot_codes.append(get_camelot_code(row['key'], row['mode']))
    else:
        camelot_codes = ['??'] * len(df_features)

    # --- APPLY FILTERS ---
    df_features, valid_candidates, camelot_codes = filter_candidates_by_config(
        df_features, valid_candidates, camelot_codes
    )
    
    if df_features.empty:
        logger.warning("  -> No candidates remain after audio filters.")
        return []

    # Transform features matrix
    feature_matrix = df_features[FEATURE_COLS].values.astype(float)
    normalized_matrix = scaler.transform(feature_matrix)
    weighted_matrix = normalized_matrix * weights
    
    # Run vectorized scoring
    try:
        final_scores, best_clusters, range_fits = score_candidates_vectorized(
            weighted_matrix, cluster_centers, cluster_weights, cluster_stats
        )
        
            
        # Assemble results
        results = []
        for i, cand in enumerate(valid_candidates):
            score = float(final_scores[i])
            
            # Apply Skip Penalty (if configured)
            # If track has > 50% skip rate historically, penalize it
            if config.penalize_skips and cand['id'] in skip_map:
                skip_rate = skip_map[cand['id']]
                if skip_rate > 0.5:
                    penalty = config.skip_penalty * skip_rate # e.g. 0.5 * 0.8 = 0.4 penalty
                    score *= (1.0 - penalty)
                    
            results.append({
                'name': cand['name'],
                'artist': cand['artist'],
                'popularity': cand['popularity'],
                'id': cand['id'],
                'score': score,
                'best_mood': int(best_clusters[i] + 1),
                'range_fit': f"{range_fits[i]*100:.0f}%",
                'camelot': camelot_codes[i],
                'bpm': int(df_features.iloc[i]['tempo']),
                'link': f"https://open.spotify.com/track/{cand['id']}"
            })
            
        return sorted(results, key=lambda x: x['score'], reverse=True)
        
    except Exception as e:
        logger.error(f"Vectorized scoring failed: {e}")
        # Fallback to empty if critical math error, but this shouldn't happen with sanitized inputs
        import traceback
        traceback.print_exc()
        return []


def run_fast_mode() -> None:
    """Fast mode using mock data for quick prototyping."""
    logger.info("🚀 FAST MODE - Using mock data for testing")
    logger.info("="*60)
    
    # Generate mock data with recency weights
    mock_features, mock_candidates, mock_candidate_features, mock_sample_weights = generate_mock_data()
    
    # Simulate user profile building
    logger.info("Analyze: Building mock user profile...")
    
    # Show recency weighting info (matching real implementation)
    if RECENCY_DECAY != 1.0:
        logger.info(f"  -> Recency weighting: short=10 (x{RECENCY_DECAY**2:.1f}), "
              f"medium=10 (x{RECENCY_DECAY**1:.1f}), "
              f"long=10 (x{RECENCY_DECAY**0:.1f})")
    
    # Use shared clustering logic
    cluster_centers, cluster_weights, cluster_stats, scaler, weights = train_mood_clusters(mock_features, mock_sample_weights)
    
    logger.info(f"  -> Mock genres: ['mock-rock', 'mock-electronic']")
    logger.info(f"  -> {len(mock_candidates)} mock candidates")
    
    # Filter candidates in fast mode to verify logic
    df_features = mock_candidate_features
    
    # Pre-calculate Camelot Codes
    camelot_codes = []
    if 'key' in df_features.columns and 'mode' in df_features.columns:
        for idx, row in df_features.iterrows():
            camelot_codes.append(get_camelot_code(row['key'], row['mode']))
    else:
        camelot_codes = ['??'] * len(df_features)

    # --- APPLY FILTERS ---
    df_features, mock_candidates, camelot_codes = filter_candidates_by_config(
        df_features, mock_candidates, camelot_codes
    )
         
    if df_features.empty:
        logger.warning("  -> No candidates remain after audio filters.")
        return

    # Use shared scoring logic (same as recommend())
    logger.info(f"Ranking: Analyzing {len(mock_candidates)} candidates...")
    
    # Batch transform all candidates at once (vectorized)
    feature_matrix = df_features[FEATURE_COLS].values.astype(float)
    normalized_matrix = scaler.transform(feature_matrix)
    weighted_matrix = normalized_matrix * weights
    
    # Run vectorized scoring
    final_scores, best_clusters, range_fits = score_candidates_vectorized(
        weighted_matrix, cluster_centers, cluster_weights, cluster_stats
    )
    
    results = []
    # Using mock_candidates directly as it aligns 1:1 with mock_candidate_features indices in generation
    for i, cand in enumerate(mock_candidates):
        score = float(final_scores[i])
        
        results.append({
            'name': cand['name'],
            'artist': cand['artist'],
            'popularity': cand['popularity'],
            'score': float(final_scores[i]),
            'best_mood': int(best_clusters[i] + 1),
            'range_fit': f"{range_fits[i]*100:.0f}%",
            'camelot': camelot_codes[i],
            'bpm': int(df_features.iloc[i]['tempo']),
            'link': f"https://open.spotify.com/track/{cand['id']}"
        })
    
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    
    # Mock serendipity wild cards
    mock_wild_cards = []
    if SERENDIPITY_SLOTS > 0:
        mock_wild_cards = [
            {'name': f'Wild Discovery {i+1}', 'artist': f'Random Artist {i+1}', 'popularity': 25 + i*5}
            for i in range(SERENDIPITY_SLOTS)
        ]
    
    # Output
    logger.info("\n" + "="*60)
    logger.info("          🎵 MOCK RECOMMENDATIONS (FAST MODE) 🎵")
    logger.info("="*60)
    
    num_ranked = TOTAL_TRACKS - len(mock_wild_cards)
    # Fast mode has limited mock data, so show what we have
    available_ranked = min(num_ranked, len(results))
    
    logger.info(f"\nShowing {available_ranked} mood-matched + {len(mock_wild_cards)} wild cards (TOTAL_TRACKS={TOTAL_TRACKS}):")
    
    for i, track in enumerate(results[:min(config.console_preview_count, available_ranked)]):
        logger.info(f"\n{i+1}. {track['name']}")
        logger.info(f"   by {track['artist']}")
        logger.info(f"   Match: {track['score']:.0%} | Mood #{track['best_mood']} | Key: {track.get('camelot')} | BPM: {track.get('bpm')}")
    
    if available_ranked > config.console_preview_count:
        logger.info(f"\n   ... and {available_ranked - config.console_preview_count} more mood-matched tracks")
    
    if mock_wild_cards:
        logger.info("\n" + "-"*60)
        logger.info(f"🎲 WILD CARDS ({len(mock_wild_cards)} serendipity picks)")
        for i, track in enumerate(mock_wild_cards):
            logger.info(f"   {track['name']} - {track['artist']}")
    
    logger.info("\n✅ Fast mode complete - all code paths tested!")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description='DIY Discovery: Spotify Playlist Generator')
    parser.add_argument('--fast', action='store_true', help='Run in fast mode with mock data for testing')
    parser.add_argument('--skip-spotify', action='store_true', help='Do not create Spotify playlist')
    parser.add_argument('--skip-tidal', action='store_true', help='Do not sync to Tidal')
    parser.add_argument('--seed_playlist', type=str, help='Use specific Spotify Playlist ID as seed instead of history')
    parser.add_argument('--coverage', action='store_true', help='Show database coverage report and exit')
    
    ARGS = parser.parse_args()

    if ARGS.fast:
        run_fast_mode()
    else:
        try:
            # --- MAIN EXECUTION FLOW ---
    
            # 0. Check for Utility Modes
            if ARGS.coverage:
                print_coverage_report()
                sys.exit(0) # Exit after report

            # 1. Load Data
            try:
                # Check if DB needs optimization (vacuukm)
                # optimize_db() 
                pass
            except Exception as e:
                logger.warning(f"DB maintenance warning: {e}")
                
            logger.info(f"Analyze: Building profile...")
            
            # 2. Get User Profile (clustered into moods with range stats)
            # Either from History or from Seed Playlist
            saved_ids: set[str] = set() # Initialize for all paths
            if ARGS.seed_playlist:
                cluster_centers, cluster_weights, cluster_stats, fav_genres, top_artist_ids, scaler, weights, skip_map = get_playlist_profile(ARGS.seed_playlist)
                logger.info(f"  -> Profiling seed playlist: {ARGS.seed_playlist}")
                logger.info(f"  -> Top Artists in playlist: {top_artist_ids[:3]}")
                # For playlist seeding, we don't have user's saved tracks unless explicitly fetched
                # For now, assume no saved tracks to exclude if seeding from playlist
                saved_ids = set() 
            else:
                saved_ids = get_saved_track_ids()
                cluster_centers, cluster_weights, cluster_stats, fav_genres, top_artist_ids, scaler, weights, skip_map = get_user_profile()
                logger.info(f"  -> Your top genres: {fav_genres}")
                
            # 3. Get Candidates from BOTH sources (hybrid approach)
            # Source A: Genre-based from local DB (content-based)
            genre_candidates = query_candidates(fav_genres, saved_ids, limit=GENRE_CANDIDATES_LIMIT)
            
            # Source B: Similar artists (genre-based local or API, based on config)
            related_candidates = get_similar_tracks(top_artist_ids, saved_ids, limit=RELATED_CANDIDATES_LIMIT)
            
            # Source C: Rediscovery (Old favorites, if enabled)
            rediscovery_candidates = []
            if config.rediscovery_mode:
                 rediscovery_candidates = get_rediscovery_candidates(limit=50)

            # Combine and deduplicate
            seen_ids = set()
            candidates = []
            for c in genre_candidates + related_candidates + rediscovery_candidates:
                if c['id'] not in seen_ids:
                    seen_ids.add(c['id'])
                    candidates.append(c)
            
            logger.info(f"  -> Combined: {len(candidates)} unique candidates")
            
            if not candidates:
                logger.info("No candidates found. Try running again for different random results.")
            else:
                # 4. Rank Candidates against mood clusters (with range awareness)
                recommendations = recommend(cluster_centers, cluster_weights, cluster_stats, candidates, scaler, weights, skip_map)
                
                # 5. Get serendipity wild cards (outside user's genre bubble and blocklist)
                exclude_genres = fav_genres + (config.blocked_genres if config.blocked_genres else [])
                wild_cards = get_serendipity_tracks(exclude_genres, saved_ids, limit=SERENDIPITY_SLOTS)
                
                # 6. Output
                logger.info("\n" + "="*60)
                logger.info("          🎵 RECOMMENDED TRACKS FOR YOU 🎵")
                logger.info("="*60)
                
                # Calculate how many ranked tracks to show
                num_ranked = TOTAL_TRACKS - len(wild_cards)
                top_ranked = recommendations[:num_ranked]
                
                # Console output (abbreviated for 35 tracks)
                logger.info(f"\nShowing {len(top_ranked)} mood-matched + {len(wild_cards)} wild cards:")
                for i, track in enumerate(top_ranked[:config.console_preview_count]):  # Show first N in console
                    logger.info(f"\n{i+1}. {track['name']}")
                    logger.info(f"   by {track['artist']}")
                    logger.info(f"   Match: {track['score']:.0%} | Mood #{track['best_mood']} | Key: {track.get('camelot')} | BPM: {track.get('bpm')}")
                
                if len(top_ranked) > 10:
                    logger.info(f"\n   ... and {len(top_ranked) - 10} more mood-matched tracks")
                
                if wild_cards:
                    logger.info("\n" + "-"*60)
                    logger.info(f"🎲 WILD CARDS ({len(wild_cards)} serendipity picks)")
                    for i, track in enumerate(wild_cards):
                        logger.info(f"   {track['name']} - {track['artist']}")
                
                # Combine for playlist
                all_tracks = top_ranked + [{'id': w['id'], 'name': w['name'], 'artist': w['artist'], 'popularity': w['popularity']} for w in wild_cards]
                
                # 7. Generate human-readable tracklist file
                timestamp = datetime.now().strftime('%Y-%m-%d_%H%M')
                tracklist_path = os.path.expanduser(f"~/DIY_Discovery_{timestamp}.txt")
                
                with open(tracklist_path, 'w') as f:
                    f.write(f"DIY Discovery - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Generated from your top genres: {', '.join(fav_genres)}\n\n")
                    
                    f.write(f"MOOD-MATCHED TRACKS ({len(top_ranked)})\n")
                    f.write("-" * 40 + "\n")
                    for i, track in enumerate(top_ranked):
                        f.write(f"\n{i+1}. {track['name']} - {track['artist']}\n")
                        f.write(f"   Match: {track['score']:.0%} | Key: {track.get('camelot')} | BPM: {track.get('bpm')}\n")
                        f.write(f"   Why: Similarity scoring against your listening clusters\n")
                        f.write(f"   {track['link']}\n")
                    
                    if wild_cards:
                        f.write(f"\n\nWILD CARDS ({len(wild_cards)}) - Serendipity picks\n")
                        f.write("-" * 40 + "\n")
                        for i, track in enumerate(wild_cards):
                            idx = len(top_ranked) + i + 1
                            f.write(f"\n{idx}. {track['name']} - {track['artist']}\n")
                            f.write(f"   Pop: {track['popularity']} | 🎲 Random discovery\n")
                            f.write(f"   Why: Serendipity pick from outside your usual genres\n")
                            link = track.get('link') or f"https://open.spotify.com/track/{track['id']}"
                            f.write(f"   {link}\n")
                
                logger.info(f"\n📝 Tracklist saved to: {tracklist_path}")
                
                # 8. Create Spotify playlist (unless --skip-spotify)
                logger.info("\n" + "="*60)
                playlist_id = None
                
                if ARGS.skip_spotify:
                    logger.info("⏭️ Skipping Spotify playlist (--skip-spotify flag)")
                else:
                    # Check if running interactively or from web UI
                    import sys
                    if sys.stdin.isatty():
                        choice = input("Save to Spotify playlist? (y/n): ").strip().lower()
                    else:
                        choice = 'y'  # Non-interactive mode, auto-save
                    
                    if choice == 'y':
                        user_info = get_spotify_client().current_user()
                        user_id = user_info['id']
                        
                        playlist_name = f"DIY Discovery - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                        
                        playlist = get_spotify_client().user_playlist_create(
                            user=user_id,
                            name=playlist_name,
                            public=True,
                            description=f"Hidden gems matching your taste. Genres: {', '.join(fav_genres)}"
                        )
                        
                        track_uris = [f"spotify:track:{t['id']}" for t in all_tracks]
                        get_spotify_client().playlist_add_items(playlist['id'], track_uris)
                        playlist_id = playlist['id']
                        
                        logger.info(f"\n✅ Created Spotify playlist: {playlist_name}")
                        logger.info(f"   {playlist['external_urls']['spotify']}")
                    else:
                        logger.info("\nPlaylist not created. Tracklist file still saved!")
                
                # 9. Sync to Tidal (unless --skip-tidal or no playlist created)
                if playlist_id and not ARGS.skip_tidal:
                    if sys.stdin.isatty():
                        tidal_choice = input("\nSync to Tidal too? (requires spotify_to_tidal installed) (y/n): ").strip().lower()
                    else:
                        tidal_choice = 'y'  # Non-interactive mode, auto-sync
                    
                    if tidal_choice == 'y':
                        logger.info("\n🔄 Syncing to Tidal...")
                        try:
                            result = subprocess.run(
                                ['spotify_to_tidal', '--uri', playlist_id],
                                capture_output=True,
                                text=True,
                                timeout=120
                            )
                            if result.returncode == 0:
                                logger.info("✅ Synced to Tidal successfully!")
                            else:
                                logger.info(f"⚠️ Tidal sync issue: {result.stderr[:200] if result.stderr else 'Unknown error'}")
                        except FileNotFoundError:
                            logger.info("⚠️ spotify_to_tidal not installed. Run: pip install spotify_to_tidal")
                        except subprocess.TimeoutExpired:
                            logger.info("⚠️ Tidal sync timed out (>120s)")
                        except Exception as e:
                            logger.info(f"⚠️ Tidal sync failed: {e}")
                elif ARGS.skip_tidal:
                    logger.info("⏭️ Skipping Tidal sync (--skip-tidal flag)")
            
        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up shared database connections
            DatabaseConnection.close_all_shared()
