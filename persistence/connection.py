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
    - REDIS_URL: Full Redis connection URL (e.g., rediss://default:pass@host:port)
    
    The URL format supports both redis:// (unencrypted) and rediss:// (TLS) protocols.
    """
    redis_url = os.getenv("REDIS_URL")

    if not redis_url:
        logger.critical("REDIS_URL environment variable is not set.")
        raise ValueError("REDIS_URL is required for Redis connection.")

    try:
        # Initialize connection pool from URL
        # rediss:// URLs automatically enable SSL/TLS
        pool = redis.ConnectionPool.from_url(
            redis_url,
            decode_responses=True,  # Returns str instead of bytes
            max_connections=50,     # Cap connections to prevent resource exhaustion
            socket_timeout=5.0      # Fail fast if connection is down
        )
        
        client = redis.Redis(connection_pool=pool)
        
        # Health check: Ping immediately to verify connection
        client.ping()
        logger.info("Successfully connected to Redis")
        
        return client

    except AuthenticationError:
        logger.critical("Redis authentication failed. Check REDIS_URL credentials.")
        raise
    except RedisError as e:
        logger.critical(f"Could not connect to Redis: {e}")
        raise