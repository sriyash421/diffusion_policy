"""Env-agnostic pieces of the Veritas pixel-waypoint verifier port.

Shared by the PushT (``env/pusht/veritas``) and SimplerEnv (``env/simpler/veritas``)
integrations: the Gemini structured-output client, the plan schemas, the waypoint
progress tracker with per-image-width constants, and the plan record store.
"""
