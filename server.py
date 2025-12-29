#!/usr/bin/env python3
"""
Simple server for the Spotify Recommender Config Editor.
Serves the HTML interface and provides an API to run recommend.py with custom config.

Usage:
    python server.py
    
Then open http://localhost:8080 in your browser.
"""

import http.server
import socketserver
import json
import subprocess
import tempfile
import os
import threading
import queue
from urllib.parse import parse_qs
import sys

PORT = 8080
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Find the Python executable that has spotipy installed
def find_python_with_spotipy():
    """Find a Python with spotipy installed."""
    candidates = [
        os.path.expanduser('~/anaconda3/bin/python'),
        os.path.expanduser('~/miniconda3/bin/python'),
        'python',
        'python3',
        sys.executable,
    ]
    
    for python in candidates:
        try:
            result = subprocess.run(
                [python, '-c', 'import spotipy'],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                return python
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
    
    # Fallback to current executable
    return sys.executable

PYTHON_EXECUTABLE = find_python_with_spotipy()

# Queue to collect output from subprocess
output_queue = queue.Queue()


class ConfigHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler that serves files and handles API requests."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=SCRIPT_DIR, **kwargs)
    
    def do_POST(self):
        if self.path == '/api/run':
            self.handle_run()
        elif self.path == '/api/run-stream':
            self.handle_run_stream()
        elif self.path == '/api/save-and-run':
            self.handle_save_and_run()
        else:
            self.send_error(404, 'Not Found')
    
    def handle_run_stream(self):
        """Run recommend.py and stream output via SSE."""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # Parse JSON payload
            try:
                payload = json.loads(post_data.decode('utf-8'))
                config_yaml = payload.get('config', '')
                save_spotify = payload.get('save_spotify', True)
                save_tidal = payload.get('save_tidal', True)
                seed_playlist = payload.get('seed_playlist')
            except json.JSONDecodeError:
                # Fallback: treat as plain YAML
                config_yaml = post_data.decode('utf-8')
                save_spotify = True
                save_tidal = True
                seed_playlist = None
            
            # Write config to temp file with spotify credentials
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
                existing_config_path = os.path.join(SCRIPT_DIR, 'config.yml')
                if os.path.exists(existing_config_path):
                    with open(existing_config_path, 'r') as ec:
                        existing = ec.read()
                        lines = existing.split('\n')
                        spotify_lines = []
                        in_spotify = False
                        for line in lines:
                            if line.startswith('spotify:'):
                                in_spotify = True
                            elif in_spotify and line and not line.startswith(' ') and not line.startswith('\t'):
                                in_spotify = False
                            if in_spotify:
                                spotify_lines.append(line)
                        if spotify_lines:
                            f.write('\n'.join(spotify_lines) + '\n\n')
                f.write(config_yaml)
                temp_config = f.name
            
            # Backup and replace config
            original_config = os.path.join(SCRIPT_DIR, 'config.yml')
            if os.path.exists(original_config):
                with open(original_config, 'r') as f:
                    original_content = f.read()
            else:
                original_content = None
            
            with open(temp_config, 'r') as f:
                new_config = f.read()
            with open(original_config, 'w') as f:
                f.write(new_config)
            
            # Build command with optional flags
            cmd = [PYTHON_EXECUTABLE, os.path.join(SCRIPT_DIR, 'recommend.py')]
            if not save_spotify:
                cmd.append('--skip-spotify')
            if not save_tidal:
                cmd.append('--skip-tidal')
            if seed_playlist:
                # Extract playlist ID from URL if needed
                playlist_id = seed_playlist
                if 'spotify.com/playlist/' in seed_playlist:
                    playlist_id = seed_playlist.split('playlist/')[-1].split('?')[0]
                elif 'spotify:playlist:' in seed_playlist:
                    playlist_id = seed_playlist.split(':')[-1]
                cmd.extend(['--seed_playlist', playlist_id])
            
            # Send SSE headers
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            try:
                # Run script with streaming output
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=SCRIPT_DIR
                )
                
                # Stream output line by line
                for line in iter(process.stdout.readline, ''):
                    if line:
                        # Send as SSE event
                        data = json.dumps({'type': 'output', 'data': line.rstrip()})
                        self.wfile.write(f"data: {data}\n\n".encode())
                        self.wfile.flush()
                
                process.wait()
                
                # Send completion event
                data = json.dumps({
                    'type': 'complete',
                    'success': process.returncode == 0,
                    'returncode': process.returncode
                })
                self.wfile.write(f"data: {data}\n\n".encode())
                self.wfile.flush()
                
            finally:
                # Restore original config
                if original_content is not None:
                    with open(original_config, 'w') as f:
                        f.write(original_content)
                os.unlink(temp_config)
                
        except Exception as e:
            try:
                data = json.dumps({'type': 'error', 'message': str(e)})
                self.wfile.write(f"data: {data}\n\n".encode())
                self.wfile.flush()
            except:
                pass
    
    def handle_run(self):
        """Run recommend.py with config from request body."""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            config_yaml = post_data.decode('utf-8')
            
            # Write config to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
                # Prepend spotify credentials from existing config
                existing_config_path = os.path.join(SCRIPT_DIR, 'config.yml')
                if os.path.exists(existing_config_path):
                    with open(existing_config_path, 'r') as ec:
                        existing = ec.read()
                        # Extract spotify section
                        lines = existing.split('\n')
                        spotify_lines = []
                        in_spotify = False
                        for line in lines:
                            if line.startswith('spotify:'):
                                in_spotify = True
                            elif in_spotify and line and not line.startswith(' ') and not line.startswith('\t'):
                                in_spotify = False
                            if in_spotify:
                                spotify_lines.append(line)
                        if spotify_lines:
                            f.write('\n'.join(spotify_lines) + '\n\n')
                
                f.write(config_yaml)
                temp_config = f.name
            
            # Run recommend.py with the temp config
            env = os.environ.copy()
            
            # Temporarily replace config.yml
            original_config = os.path.join(SCRIPT_DIR, 'config.yml')
            backup_config = os.path.join(SCRIPT_DIR, 'config.yml.bak')
            
            # Backup original
            if os.path.exists(original_config):
                with open(original_config, 'r') as f:
                    original_content = f.read()
            else:
                original_content = None
            
            # Write new config (with spotify credentials preserved)
            with open(temp_config, 'r') as f:
                new_config = f.read()
            with open(original_config, 'w') as f:
                f.write(new_config)
            
            try:
                # Run the script
                result = subprocess.run(
                    [PYTHON_EXECUTABLE, os.path.join(SCRIPT_DIR, 'recommend.py')],
                    capture_output=True,
                    text=True,
                    cwd=SCRIPT_DIR,
                    timeout=300  # 5 minute timeout
                )
                
                output = result.stdout
                if result.stderr:
                    output += '\n\nSTDERR:\n' + result.stderr
                
                response = {
                    'success': result.returncode == 0,
                    'output': output,
                    'returncode': result.returncode
                }
            finally:
                # Restore original config
                if original_content is not None:
                    with open(original_config, 'w') as f:
                        f.write(original_content)
            
            # Clean up temp file
            os.unlink(temp_config)
            
            # Send response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except subprocess.TimeoutExpired:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'success': False,
                'output': 'Script timed out after 5 minutes',
                'returncode': -1
            }).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'success': False,
                'output': str(e),
                'returncode': -1
            }).encode())
    
    def handle_save_and_run(self):
        """Save config to config.yml and run recommend.py."""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            config_yaml = post_data.decode('utf-8')
            
            # Read existing config to preserve spotify credentials
            config_path = os.path.join(SCRIPT_DIR, 'config.yml')
            spotify_section = ''
            other_sections = ''
            
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    existing = f.read()
                    lines = existing.split('\n')
                    spotify_lines = []
                    other_lines = []
                    in_spotify = False
                    in_recommender = False
                    
                    for line in lines:
                        if line.startswith('spotify:'):
                            in_spotify = True
                            in_recommender = False
                        elif line.startswith('recommender:'):
                            in_spotify = False
                            in_recommender = True
                        elif line and not line.startswith(' ') and not line.startswith('\t') and not line.startswith('#'):
                            in_spotify = False
                            in_recommender = False
                        
                        if in_spotify:
                            spotify_lines.append(line)
                        elif not in_recommender and not line.startswith('recommender:'):
                            # Keep other sections like sync_favorites_default, etc.
                            if not (line.strip().startswith('#') and 'recommender' in line.lower()):
                                other_lines.append(line)
                    
                    spotify_section = '\n'.join(spotify_lines)
                    # Filter out recommender-related lines from other_lines
                    other_lines = [l for l in other_lines if l.strip()]
                    # Keep only non-empty, non-recommender lines
                    other_sections = '\n'.join(other_lines)
            
            # Write new config
            with open(config_path, 'w') as f:
                if spotify_section:
                    f.write(spotify_section + '\n\n')
                f.write(config_yaml)
                # Append remaining sections that aren't spotify or recommender
                remaining = []
                for line in other_sections.split('\n'):
                    if line.startswith('sync_') or line.startswith('max_') or line.startswith('rate_') or line.startswith('excluded_'):
                        remaining.append(line)
                if remaining:
                    f.write('\n\n' + '\n'.join(remaining))
            
            # Now run recommend.py
            result = subprocess.run(
                [PYTHON_EXECUTABLE, os.path.join(SCRIPT_DIR, 'recommend.py')],
                capture_output=True,
                text=True,
                cwd=SCRIPT_DIR,
                timeout=300
            )
            
            output = result.stdout
            if result.stderr:
                output += '\n\nSTDERR:\n' + result.stderr
            
            response = {
                'success': result.returncode == 0,
                'output': output,
                'returncode': result.returncode,
                'saved': True
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'success': False,
                'output': str(e),
                'returncode': -1
            }).encode())
    
    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def log_message(self, format, *args):
        """Custom log format."""
        print(f"[{self.log_date_time_string()}] {args[0]}")


def main():
    # Allow port reuse immediately after restart
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), ConfigHandler) as httpd:
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  🎵 Spotify Recommender Config Editor                        ║
║                                                              ║
║  Server running at: http://localhost:{PORT}                   ║
║  Config editor at:  http://localhost:{PORT}/config-editor.html║
║                                                              ║
║  Press Ctrl+C to stop                                        ║
╚══════════════════════════════════════════════════════════════╝
""")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            httpd.shutdown()


if __name__ == '__main__':
    main()
