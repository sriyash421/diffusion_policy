"""SAC on PushT, to learn a Q that can rank best-of-N candidates.

The PushT best-of-N verifier is currently a hand-written distance heuristic evaluated in a
sim. It ignores `agent_pos` by construction, so before the arm touches the block every
candidate scores identically and `argmax` is a tie-break -- on 28-35% of decisions overall and
68-73% of approach decisions. This package learns `Q*(s, chunk)` instead, whose pre-contact
signal is produced by the Bellman backup rather than by a hand-written approach term.
"""
