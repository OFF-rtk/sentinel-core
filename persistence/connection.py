import os
import logging
from functools import lru_cache

import redis
from redis.exceptions import RedisError, AuthenticationError

# Configure module-level logger
logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_redis_client() -> redis.Redis:
    """
    Creates a singleton Redis client with a connection pool.
    
    Reads configuration from environment variables:
    - REDIS_HOST: Hostname (default: localhost)
    - REDIS_PORT: Port (default: 6379)
    - REDIS_PASSWORD: Password (REQUIRED)
    """
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", 6379))
    password = os.getenv("REDIS_PASSWORD")

    if not password:
        logger.critical("REDIS_PASSWORD environment variable is not set.")
        raise ValueError("REDIS_PASSWORD is required for production security.")

    try:
        # Initialize connection pool with constraints matching Docker limits
        pool = redis.ConnectionPool(
            host=host,
            port=port,
            password=password,
            decode_responses=True,  # Returns str instead of bytes
            max_connections=50,     # Cap connections to prevent resource exhaustion
            socket_timeout=5.0      # Fail fast if container is down
        )
        
        client = redis.Redis(connection_pool=pool)
        
        # Health check: Ping immediately to verify connection
        client.ping()
        logger.info(f"Successfully connected to Redis at {host}:{port}")
        
        return client

    except AuthenticationError:
        logger.critical("Redis authentication failed. Check REDIS_PASSWORD.")
        raise
    except RedisError as e:
        logger.critical(f"Could not connect to Redis: {e}")
        raise