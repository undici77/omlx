# SPDX-License-Identifier: Apache-2.0
"""Directional steering (DS4-style rank-1 projection on branch outputs).

See docs/experimental/directional-steering-plan.md. Nothing in this package
touches a model unless ``install_steering`` is called with a pack configured
(plan rule R1).
"""

from .install import get_state, install_steering, steering_active
from .pack import SteeringPack, load_pack, save_pack
from .state import SteeringState

__all__ = [
    "SteeringPack",
    "SteeringState",
    "get_state",
    "install_steering",
    "load_pack",
    "save_pack",
    "steering_active",
]
