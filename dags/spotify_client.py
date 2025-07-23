import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException
from token_manager import SpotifyTokenManager
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpotifyClient:
    def __init__(self, use_oauth=False):
        try:
            token_manager = SpotifyTokenManager()
            token = token_manager.get_access_token(use_oauth=use_oauth)
            if use_oauth:
                self.sp = spotipy.Spotify(auth=token)
            else:
                credentials_manager = SpotifyClientCredentials(
                    client_id=token_manager.client_id,
                    client_secret=token_manager.client_secret
                )
                self.sp = spotipy.Spotify(auth_manager=credentials_manager)
        except Exception as e:
            logger.error(f"Failed to initialize SpotifyClient: {e}")
            raise

    def get_playlist_tracks(self, playlist_id):
        results = []
        offset = 0
        max_retries = 5
        retry_delay = 5  # Initial delay in seconds

        while True:
            for attempt in range(max_retries):
                try:
                    logger.info(f"Fetching tracks for playlist {playlist_id} at offset {offset}")
                    response = self.sp.playlist_tracks(playlist_id, limit=100, offset=offset)
                    # Handle string response
                    if isinstance(response, str):
                        logger.error(f"Received string response for playlist {playlist_id}: {response}")
                        return []  # Skip invalid playlist
                    # Check for error response
                    if isinstance(response, dict) and 'error' in response:
                        error_status = response['error'].get('status')
                        error_msg = response['error'].get('message', 'Unknown error')
                        if error_status == 404:
                            logger.error(f"Playlist {playlist_id} not found: {error_msg}")
                            return []  # Skip invalid playlist
                        elif error_status == 429:
                            retry_after = int(response.get('headers', {}).get('Retry-After', retry_delay))
                            logger.warning(f"Rate limit hit for playlist {playlist_id}. Retrying after {retry_after} seconds...")
                            time.sleep(retry_after)
                            continue
                        else:
                            logger.error(f"Spotify API error for playlist {playlist_id}: {error_status} - {error_msg}")
                            raise ValueError(f"Spotify API error: {error_status} - {error_msg}")
                    # Validate response type
                    if not isinstance(response, dict):
                        logger.error(f"Expected dict response, got {type(response)}: {response}")
                        return []  # Skip invalid playlist
                    items = response.get('items', [])
                    if not isinstance(items, list):
                        logger.error(f"Expected list of items, got {type(items)}: {items}")
                        return []  # Skip invalid playlist
                    if not items:
                        break
                    results.extend(items)
                    offset += len(items)
                    break  # Exit retry loop on success
                except SpotifyException as e:
                    if e.http_status == 429:
                        retry_after = int(e.headers.get('Retry-After', retry_delay))
                        logger.warning(f"Rate limit hit for playlist {playlist_id}. Retrying after {retry_after} seconds...")
                        time.sleep(retry_after)
                        continue
                    elif e.http_status == 404:
                        logger.error(f"Playlist {playlist_id} not found: {e}")
                        return []  # Skip invalid playlist
                    else:
                        logger.error(f"Spotify API exception for playlist {playlist_id}: {e}")
                        raise
                except Exception as e:
                    logger.error(f"Unexpected error fetching playlist {playlist_id}: {e}")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying attempt {attempt + 2}/{max_retries} after {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        logger.error(f"Max retries reached for playlist {playlist_id}")
                        return []  # Skip instead of raising error
            else:
                logger.error(f"Failed to fetch tracks for playlist {playlist_id} after {max_retries} attempts")
                return []  # Skip instead of raising error
            if not response.get('next'):
                break
        logger.info(f"Fetched {len(results)} tracks for playlist {playlist_id}")
        return results