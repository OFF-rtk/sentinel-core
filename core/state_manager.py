"""
Sentinel State Manager

This module acts as the thread-safe, in-memory "Hot Storage" for the Sentinel Core.
It persists the User History snapshots required to calculate the 8 "Golden Metrics" 
in the Context Processor.

Data Structure Responsibility:
The State Manager maintains a global registry of 'UserSnapshot' objects containing:

1.  Location Context (For 'geo_velocity_mph' & 'ip_reputation'):
    - last_ip (str): The IP address from the previous successful action.
    - last_geo_coords (Tuple[float, float]): (Latitude, Longitude) of the last action.
    - last_country (str): To detect rapid cross-border hops.

2.  Temporal Context (For 'time_since_last_seen' & 'time_anomaly_score'):
    - last_seen_timestamp (float): Unix timestamp of the last activity.
    - usual_login_hours (List[int]): A frequency map of hour-of-day (0-23).

3.  Device Context (For 'is_new_device' & 'browser_integrity_score'):
    - known_device_hashes (Set[str]): A collection of previously valid 'ja3_hash' or 'device_id' strings.
    - known_user_agents (Set[str]): To detect sudden OS/Browser switches.

4.  Transactional Context (For 'transaction_magnitude'):
    - avg_transaction_amount (float): Moving average of valid transaction values.
    - transaction_count (int): To weight the average correctly.
    - home_currency (str): The currency the user historically uses (e.g., 'USD').

5.  Identity Context (For 'policy_violation_flag'):
    - role (str): The cached RBAC role (e.g., 'intern', 'admin') to validate against resource targets.

Usage:
    manager = StateManager()
    history = manager.get_user_history("usr_123")
    # history is passed to ContextProcessor.enrich_context(current_ctx, history)
"""

# Code to be implemented later:
# - Singleton Pattern for StateManager class
# - Thread-safe locking for concurrent updates
# - UserSnapshot dataclass definition