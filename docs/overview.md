# What is Sentinel?

Sentinel is a **continuous authentication system** that builds trust through behavior, not just credentials.

## The Problem
Sentinel operates entirely server-side and does not require invasive client permissions.
Traditional security relies on "gates"—passwords, 2FA codes, or CAPTCHAs. Once a user passes the gate, they are trusted until their session expires. This leaves a massive vulnerability: **session hijacking**. If an attacker steals a cookie or a token, they *become* the user.

Sentinel solves this by constantly verifying *who* is behind the keyboard and mouse, throughout the entire session.

## How It Works: The Trust Lifecycle

Sentinel does not just look for "bad" actions; it looks for *your* actions.

1.  **Cold Start (Unverified)**: When a session begins, Sentinel knows nothing. Trust is zero. The system aggressively monitors for obvious bot-like behavior (superhuman typing speeds, perfect straight-line mouse movements).
2.  **Trust Formation**: As the user interacts naturally, Sentinel builds a behavioral profile. If the behavior matches humanity (and eventually, the user's specific history), trust scores rise.
3.  **Mature Session (Trusted)**: Once trust crosses a high threshold, the user is "verified." Sentinel steps back, reducing friction. However, any sudden anomaly (a script takes over, or a different person sits down) causes trust to decay rapidly, triggering a challenge or block.

## How Sentinel is Different

### vs. reCAPTCHA
reCAPTCHA is a "test"—it interrupts the user and asks them to prove they are human *right now*. Sentinel is **invisible**. It runs in the background, analyzing the stream of events (mouse moves, key presses) that the user is *already doing*.

### vs. Device Fingerprinting
Fingerprinting tracks *machines* (browser version, screen resolution, IP). Sentinel tracks *behavior*. A hacker can steal your laptop (matching the fingerprint), but they cannot steal the way you type or move your mouse.

### vs. Static ML Risk Scores
Many fraud tools give a single score at login (e.g., "Risk: Low"). Sentinel produces a **time-variant risk score**. It evolves every second. A session can start safe and turn dangerous in an instant if behavior changes.
