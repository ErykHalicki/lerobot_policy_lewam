"""
Launch lerobot's policy server with LeWAM support.

Patches the SUPPORTED_POLICIES list to include "lewam", then starts
the standard gRPC policy server.

Usage:
    python -m lerobot_policy_lewam.serve_lewam \
        --host=0.0.0.0 \
        --port=8080 \
        --fps=30
"""

import lerobot.async_inference.constants as constants

constants.SUPPORTED_POLICIES.append("lewam")

from lerobot.async_inference.policy_server import serve

if __name__ == "__main__":
    serve()
