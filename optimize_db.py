import sqlite3
import time
import os

DB_PATH = 'spotify_clean.sqlite3'

def optimize_database():
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at: {DB_PATH}")
        return

    print(f"🔌 Connecting to {DB_PATH}...")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        indexes_to_create = [
            ('idx_albums_release_date', 'albums', 'release_date'),
            ('idx_artist_genres_genre', 'artist_genres', 'genre'),
            ('idx_artist_genres_rowid', 'artist_genres', 'artist_rowid'),
            ('idx_track_artists_artist', 'track_artists', 'artist_rowid'),
            ('idx_track_artists_track', 'track_artists', 'track_rowid'),
            ('tracks_album', 'tracks', 'album_rowid'),
        ]
        
        for idx_name, table, column in indexes_to_create:
            print(f"🔍 Checking for '{idx_name}'...")
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='index' AND name=?", (idx_name,))
            if cursor.fetchone():
                print(f"✅ Index '{idx_name}' already exists.")
            else:
                print(f"⏳ Creating index '{idx_name}' on {table}({column})...")
                start_time = time.time()
                cursor.execute(f"CREATE INDEX {idx_name} ON {table}({column})")
                conn.commit()
                elapsed = time.time() - start_time
                print(f"✅ Index created in {elapsed:.1f} seconds.")
        
        # Run ANALYZE to update query planner statistics
        print("📊 Running ANALYZE to update query statistics...")
        start_time = time.time()
        cursor.execute("ANALYZE")
        conn.commit()
        elapsed = time.time() - start_time
        print(f"✅ ANALYZE completed in {elapsed:.1f} seconds.")

        print("\n✨ Optimization complete!")
        conn.close()

    except Exception as e:
        print(f"❌ Error during optimization: {e}")

if __name__ == "__main__":
    optimize_database()

