import numpy as np
import gymnasium as gym

from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv

class SSLDriveBallEnv(SSLBaseEnv):
    """The SSL robot needs to reach the ball


    Description:

    Observation:
        Type: Box(9,)
        Normalized Bounds to [-1.2, 1.2]
        Num      Observation normalized
        0->3     Ball [X, Y, V_x, V_y]
        4->8    Robot [X, Y, V_x, V_y, v_theta]


    Actions:
        Type: Box(4,)
        Value Range: [-1, 1]
        Num     Action
        0       V_X
        1       V_y
        2       id 0 Blue Angular Speed  (%)
        3       id 0 Blue Dribbler  (%) (true if % is positive)

    Reward:
     - Move
     - Take Ball to Goal Point

    Starting State:
        - Robot and Ball positions are random
        - Goal Point is random

    Episode Termination:
        30 seconds (1200 steps) or reached target
    """
    def __init__(self, render_mode=None):
        super().__init__(
            field_type=2,
            n_robots_blue=1,
            n_robots_yellow=0,
            time_step=0.025,
            render_mode=render_mode,
        )


        n_actions = 4
        self.action_space = gym.spaces.Box(low=-1, 
                                           high=1, 
                                           shape=(n_actions,), 
                                           dtype=np.float32)
        
        n_obs = 9
        self.observation_space = gym.spaces.Box(
            low=-self.NORM_BOUNDS,
            high=self.NORM_BOUNDS,
            shape=(n_obs,),
            dtype=np.float32,
        )


    def _get_commands(self, action):
        """returns a list of commands of type List[Robot] from type action_space action"""
        return [Robot(
            yellow=False, 
            id=0, 
            v_x=action[0], 
            v_y=action[1], 
            v_theta=action[2], 
            dribbler=action[3] > 0.5
        )]


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
            self.norm_w(robot.v_theta)
        ], dtype=np.float32)


    def _calculate_reward_and_done(self):
        reward = 0
        done = False
        robot_radius = self.field.rbt_radius
        ball_radius = self.field.ball_radius

        ball = self.frame.ball
        robot = self.frame.robots_blue[0]

        if self.distanec(np.array([robot.x, robot.y]), np.array([ball.x, ball.y])) < robot_radius + ball_radius + 0.1:
            reward += 1

        if self.distance(
            np.array([robot.x, robot.y]), np.array([ball.x, ball.y])
        ) < robot_radius + ball_radius + 0.1:
            if robot.dribbler:
                reward += 1

        if self.distance(np.array([ball.x, ball.y]), np.array(self.goal)) < 0.1:
            reward += 1
            done = True
            
        return reward, done


    def distance(self, a, b):
        return np.linalg.norm(a - b)


    def _get_initial_positions_frame(self) -> Frame:
        """returns frame with robots initial positions"""
        
        pos_frame: Frame = Frame()

        pos_frame.ball = Ball(
            x=np.random.uniform(-1.2, 1.2), 
            y=np.random.uniform(-1.2, 1.2)
        )

        self.goal = [np.random.uniform(-1.2, 1.2), np.random.uniform(-1.2, 1.2)]

        while True:
            robot_x = np.random.uniform(-1.2, 1.2)
            robot_y = np.random.uniform(-1.2, 1.2)

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
                break

        return pos_frame
