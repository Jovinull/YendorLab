"""Run the optional Gymnasium validator against real Brogue."""

from pathlib import Path

from gymnasium.utils.env_checker import check_env

from yendorlab.gym_env import GymExplorationEnv


environment = GymExplorationEnv(Path(".local/BrogueCE/bin/brogue"))
try:
    check_env(environment)
    print("CHECK_ENV_PASS")
finally:
    environment.close()
