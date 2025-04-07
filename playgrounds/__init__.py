from playgrounds.reach_ball import SSLReachBallEnv
from playgrounds.drive_ball import SSLDriveBallEnv
from gymnasium.envs.registration import register

register(
    id="SSLReachBall-v0",
    entry_point="playgrounds:SSLReachBallEnv",
    max_episode_steps=1200,
)

register(
    id="SSLDriveBall-v0",
    entry_point="playgrounds:SSLDriveBallEnv",
    max_episode_steps=1200,
)

