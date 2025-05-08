from playgrounds.rotate_to_ball import SSLRotateToBallEnv
from playgrounds.reach_ball import SSLReachBallEnv
from playgrounds.drive_ball import SSLDriveBallEnv
from gymnasium.envs.registration import register
from playgrounds.el_ssl import CustomSSLEnv
from playgrounds.el_reach_to_ball import ELSSLReachBallEnv

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
    id="ELSSLReachBall-v0",
    entry_point="playgrounds:ELSSLReachBallEnv",
    max_episode_steps=1200,
)

