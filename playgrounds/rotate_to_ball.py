import numpy as np
import gymnasium as gym

from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv

class SSLRotateToBallEnv(SSLBaseEnv):
    """The SSL robot needs to reach the ball


    Description:

    Observation:
        Type: Box(11,)
        Normalized Bounds to [-1.2, 1.2]
        Num      Observation normalized
        0->3     Ball [X, Y, V_x, V_y]
        4->10    Robot [X, Y, V_x, V_y, sin_theta, cos_theta, v_theta]


    Actions:
        Type: Box(3,)
        Value Range: [-1, 1]
        Num     Action
        0       V_X
        1       V_y
        2       V_theta

    Reward:
     - Move
     - Goal

    Starting State:
        - Robot and Ball positions are random

    Episode Termination:
        30 seconds (1200 steps) or reached target
    """
    def __init__(self, render_mode=None, max_steps=1200):
        super().__init__(
            field_type=2,
            n_robots_blue=1,
            n_robots_yellow=0,
            time_step=0.025,
            render_mode=render_mode,
        )


        n_actions = 3
        self.action_space = gym.spaces.Box(low=-1, 
                                           high=1, 
                                           shape=(n_actions,), 
                                           dtype=np.float32)
        
        n_obs = 11
        self.observation_space = gym.spaces.Box(
            low=-self.NORM_BOUNDS,
            high=self.NORM_BOUNDS,
            shape=(n_obs,),
            dtype=np.float32,
        )

        self.max_steps = max_steps

    def _get_commands(self, action):
        """returns a list of commands of type List[Robot] from type action_space action"""
        return [Robot(yellow=False, id=0, v_x=action[0], v_y=action[1], v_theta=action[2])]

    def _frame_to_observations(self):
        """returns a type observation_space observation from a type List[Robot] state"""
        ball, robot = self.frame.ball, self.frame.robots_blue[0]
        return np.array([
            self.norm_pos(ball.x),
            self.norm_pos(ball.y),
            self.norm_v(ball.v_x),
            self.norm_v(ball.v_y),
            self.norm_pos(robot.x),
            self.norm_pos(robot.y),
            self.norm_v(robot.v_x),
            self.norm_v(robot.v_y),
            np.sin(np.deg2rad(robot.theta)),
            np.cos(np.deg2rad(robot.theta)),
            self.norm_w(robot.v_theta)
        ], dtype=np.float32)
    
    def _calculate_reward_and_done(self):
        """returns reward value and done flag from type List[Robot] state"""
        reward = 0
        done = False
        
        # Reward if touched the ball
        if self._robot_is_facing_ball():
            reward += 10
            done = True

        if self.steps >= self.max_steps:
            done = True
            
        if done:
            # Reward negatively for time spent
            reward -= self.steps * 0.01

        return reward, done
        

    def _get_initial_positions_frame(self) -> Frame:
        """returns frame with robots initial positions"""
        
        pos_frame: Frame = Frame()

        pos_frame.ball = Ball(
            x=np.random.uniform(-self.NORM_BOUNDS, self.NORM_BOUNDS), 
            y=np.random.uniform(-self.NORM_BOUNDS, self.NORM_BOUNDS)
        )

        while True:
            robot_x = np.random.uniform(-self.NORM_BOUNDS, self.NORM_BOUNDS)
            robot_y = np.random.uniform(-self.NORM_BOUNDS, self.NORM_BOUNDS)

            distance_robot_ball = np.linalg.norm(
                np.array([
                    robot_x - pos_frame.ball.x,
                    robot_y - pos_frame.ball.y
                ])
            )

            if distance_robot_ball > 0.2:
                pos_frame.robots_blue = {
                    0: Robot(
                        yellow=False,
                        id=0,
                        x=robot_x,
                        y=robot_y,
                        theta=0,
                    )
                }

                self.distance_robot_ball_start = distance_robot_ball
                break

        return pos_frame
    

    def _robot_is_facing_ball(self):
        """returns True if robot is facing ball"""
        robot, ball = self.frame.robots_blue[0], self.frame.ball
        theta_rad = np.deg2rad(robot.theta)

        robot_to_ball_direction = np.array([ball.x - robot.x, ball.y - robot.y])
        robot_to_ball_direction /= np.linalg.norm(robot_to_ball_direction)
        robot_orientation = np.array([np.cos(theta_rad), np.sin(theta_rad)])
        dot_product = np.dot(robot_orientation, robot_to_ball_direction)

        return dot_product > 0.9
