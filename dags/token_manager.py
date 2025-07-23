import requests
import base64
import time
import yaml
import os
import logging
from spotipy.oauth2 import SpotifyOAuth

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpotifyTokenManager:
    def __init__(self):
        config_path = '/opt/airflow/config/config.yaml'
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found at {config_path}")
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Error reading config.yaml: {e}")
            raise ValueError(f"Failed to read config.yaml: {e}")
        
        if not config or 'spotify' not in config:
            logger.error("Invalid config.yaml: Missing 'spotify' section")
            raise ValueError("Invalid config.yaml: Missing 'spotify' section")
        
        self.client_id = config['spotify'].get('client_id')
        self.client_secret = config['spotify'].get('client_secret')
        self.redirect_uri = config['spotify'].get('redirect_uri', 'https://127.0.0.1:8888/')
        if not self.client_id or not self.client_secret:
            logger.error("Missing client_id or client_secret in config.yaml")
            raise ValueError("Missing client_id or client_secret in config.yaml")
        
        self.scope = 'playlist-read-private playlist-read-collaborative user-library-read'
        self.access_token = None
        self.expires_at = 0
        self.refresh_token = None  # For OAuth flow

    def get_access_token(self, use_oauth=False):
        if self.access_token and time.time() < self.expires_at:
            return self.access_token

        max_retries = 5
        retry_delay = 5  # Initial delay in seconds

        for attempt in range(max_retries):
            try:
                if use_oauth:
                    sp_oauth = SpotifyOAuth(
                        client_id=self.client_id,
                        client_secret=self.client_secret,
                        redirect_uri=self.redirect_uri,
                        scope=self.scope,
                        show_dialog=False
                    )
                    # In production, store and use refresh_token
                    if self.refresh_token:
                        token_info = sp_oauth.refresh_access_token(self.refresh_token)
                    else:
                        # For testing, manually obtain code
                        auth_url = sp_oauth.get_authorize_url()
                        logger.info(f"Access the following URL to obtain the code: {auth_url}")
                        code = input("Enter the code from the URL: ").strip()
                        token_info = sp_oauth.get_access_token(code)
                        self.refresh_token = token_info.get('refresh_token')

                    self.access_token = token_info['access_token']
                    self.expires_at = time.time() + token_info['expires_in'] - 60
                    logger.info("Successfully obtained OAuth token")
                    return self.access_token
                else:
                    auth_header = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
                    headers = {"Authorization": f"Basic {auth_header}"}
                    data = {"grant_type": "client_credentials"}

                    response = requests.post("https://accounts.spotify.com/api/token", headers=headers, data=data)

                    if response.status_code == 200:
                        token_info = response.json()
                        self.access_token = token_info['access_token']
                        self.expires_at = time.time() + token_info['expires_in'] - 60
                        logger.info("Successfully obtained client credentials token")
                        return self.access_token
                    elif response.status_code == 429:
                        retry_after = int(response.headers.get('Retry-After', retry_delay))
                        logger.warning(f"Rate limit hit for token request. Retrying after {retry_after} seconds...")
                        time.sleep(retry_after)
                        continue
                    else:
                        logger.error(f"Failed to get token: {response.status_code} - {response.text}")
                        raise Exception(f"Failed to get token: {response.status_code} - {response.text}")
            except Exception as e:
                logger.error(f"Error obtaining token (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise Exception(f"Max retries reached for token request: {e}")