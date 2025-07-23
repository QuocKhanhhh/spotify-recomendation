import os
import pandas as pd
from spotify_client import SpotifyClient
import logging
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] {%(filename)s:%(lineno)d} - %(message)s'
)
logger = logging.getLogger(__name__)

def crawl_single_playlist(playlist_id, output_file='/opt/airflow/data/crawled_tracks.csv'):
    sp_client = SpotifyClient()
    all_data = []

    try:
        logger.info(f"Crawling playlist {playlist_id}")
        items = sp_client.get_playlist_tracks(playlist_id)
        if not items:
            logger.warning(f"No tracks found for playlist {playlist_id}. Marking as failed.")
            return False

        for item in items:
            if not isinstance(item, dict):
                logger.warning(f"Skipping invalid item in playlist {playlist_id}: {item}")
                continue
            track = item.get('track')
            if not isinstance(track, dict):
                logger.warning(f"Skipping invalid track in playlist {playlist_id}: {track}")
                continue
            if not track:
                logger.warning(f"No track data for item in playlist {playlist_id}: {item}")
                continue

            try:
                artists = track.get('artists', [])
                artist_name = artists[0]['name'] if artists and isinstance(artists, list) and len(artists) > 0 and isinstance(artists[0], dict) else None
                artist_id = artists[0]['id'] if artists and isinstance(artists, list) and len(artists) > 0 and isinstance(artists[0], dict) else None
                # Fetch genres from artist endpoint if available
                artist_genres = ''
                if artist_id:
                    try:
                        artist_info = sp_client.sp.artist(artist_id)
                        artist_genres = ','.join(artist_info.get('genres', [])) if isinstance(artist_info, dict) else ''
                    except Exception as e:
                        logger.warning(f"Error fetching genres for artist {artist_id} in playlist {playlist_id}: {str(e)}")
                album = track.get('album', {})
                release_date = album.get('release_date') if isinstance(album, dict) else None
                external_urls = track.get('external_urls', {})
                spotify_url = external_urls.get('spotify') if isinstance(external_urls, dict) else None
                track_id = track.get('id')
                track_name = track.get('name')
                popularity = track.get('popularity')
                explicit = track.get('explicit')

                # Validate required fields
                if not all([track_id, track_name, artist_name, popularity is not None, explicit is not None]):
                    logger.warning(f"Skipping track with missing required fields in playlist {playlist_id}: {track_name}")
                    continue

                all_data.append({
                    'playlist_id': playlist_id,
                    'track_id': track_id,
                    'track_name': track_name,
                    'artist_name': artist_name,
                    'release_date': release_date,
                    'popularity': popularity,
                    'explicit': explicit,
                    'external_url': spotify_url,
                    'artist_genres': artist_genres
                })
            except Exception as e:
                logger.warning(f"Error processing track in playlist {playlist_id}: {str(e)}\n{traceback.format_exc()}")
                continue

        if all_data:
            logger.info(f"Saving {len(all_data)} tracks for playlist {playlist_id}")
            df = pd.DataFrame(all_data)
            write_header = not os.path.exists(output_file)
            try:
                df.to_csv(output_file, mode='a', index=False, header=write_header, encoding='utf-8')
                logger.info(f"Saved {len(df)} tracks to {output_file}")
            except Exception as e:
                logger.error(f"Error saving tracks to {output_file}: {str(e)}\n{traceback.format_exc()}")
                raise
            return True
        else:
            logger.warning(f"No valid tracks found for playlist {playlist_id}")
            return False

    except Exception as e:
        logger.error(f"Error crawling playlist {playlist_id}: {str(e)}\n{traceback.format_exc()}")
        return False