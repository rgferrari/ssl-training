from playgrounds.rotate_to_ball import SSLRotateToBallEnv
from playgrounds.reach_ball import SSLReachBallEnv
from playgrounds.drive_ball import SSLDriveBallEnv
from playgrounds.reach_ball_el import SSLELReachBallEnv
from gymnasium.envs.registration import register

register(
    id="SSLRotateToBall-v0",
    entry_point="playgrounds:SSLRotateToBallEnv",
    max_episode_steps=1200,
)

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

register(
    id="SSLReachBallEL-v0",
    entry_point="playgrounds:SSLELReachBallEnv",
    max_episode_steps=1200,
)
