import pygame
import numpy as np
import gymnasium as gym
from dataclasses import dataclass

from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.Render.utils import COLORS

# Import your custom classes
from playgrounds.Env.SSLELSim import SSL_EL_Env


@dataclass
class Position:
    x: float
    y: float


class SSLELReachToPoint(SSL_EL_Env):
    """Environment where the robot must reach a random target point into the EL Field

    Description:
        The controlled robot (blue[0]) starts at a random position and needs to
        reach a randomly positioned target on the field as quickly as possible.
        Only one robot is physically present in the environment, while others exist
        only in the observation space.
        
    Observation:
        Type: Box(23, )
        Normalized Bounds to [-1.2, 1.2]
        Num      Observation normalized
        0->3     Ball [X, Y, V_x, V_y]
        4->11    Blue Robot 0 [X, Y, sin(theta), cos(theta), v_x, v_y, v_theta, infrared]
        12->13   Blue Robot 1 [X, Y] (fictitious, only in observations)
        14->15   Blue Robot 2 [X, Y] (fictitious, only in observations)
        16->17   Yellow Robot 0 [X, Y] (fictitious, only in observations)
        18->19   Yellow Robot 1 [X, Y] (fictitious, only in observations)
        20->21   Yellow Robot 2 [X, Y] (fictitious, only in observations)
        22->24   Target [X, Y, theta]
        
    Actions:
        Type: Box(5, )
        Num     Action
        0       Blue Robot 0 Global X Direction Speed (%)
        1       Blue Robot 0 Global Y Direction Speed (%)
        2       Blue Robot 0 Angular Speed (%)
        3       Blue Robot 0 Kick x Speed (%) (always 0, not used)
        4       Blue Robot 0 Dribbler (%) (always False, not used)

    Reward:
        +100.0 for reaching the target
        +Additional reward based on time efficiency (faster = better)
        -0.1 penalty per step
        -20.0 penalty for going out of bounds
        -30.0 penalty for timeout
        -Additional penalty based on final distance to target if timeout or out of bounds
        
    Starting State:
        Robot and target at random positions on the field
        
    Episode Termination:
        Target reached, out of bounds, or maximum steps reached
    """
    def __init__(self, render_mode=None, max_steps=1200):
        # Initialize with only 1 blue robot and 0 yellow robots
        # Other robots will only exist in observations, not in the actual environment
        super().__init__(render_mode=render_mode, n_robots_blue=1, n_robots_yellow=0)
        
        # To maintain compatibility with existing code, we still use these values
        # for the observation vector size
        self.n_robots_blue_obs = 3
        self.n_robots_yellow_obs = 3
        
        self.max_steps = max_steps
        self.target_position = Position(x=0.0, y=0.0)
        
        # Actions: v_x, v_y, v_theta, kick_v_x, dribbler
        n_actions = 5
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(n_actions,), dtype=np.float32
        )
        
        # Observations: ball(4) + controlled robot(8) + other blue robots(2x2) + yellow robots(3x2) + target(3) = 23
        n_obs = 4 + 8 + 2 * (self.n_robots_blue_obs - 1) + 2 * self.n_robots_yellow_obs + 3
        self.observation_space = gym.spaces.Box(
            low=-self.NORM_BOUNDS,
            high=self.NORM_BOUNDS,
            shape=(n_obs,),
            dtype=np.float32,
        )

        self.max_v = 5.0
        self.max_w = 10
        self.kick_speed_x = 5.0

        # Metrics for logging
        self.episodes_metrics = {}
        self.steps_to_target = 0
        self.initial_distance = 0.0
        
        # Reward parameters
        self.reward_target_reached = 100.0
        self.penalty_out_of_bounds = -20.0
        self.max_steps_per_meter = 12.0
        self.penalty_per_step = -0.1
        self.penalty_timeout = -30.0

    def _get_initial_positions_frame(self) -> Frame:
        """
        Initializes the positions of all entities for the episode.
        Only the agent (blue[0]) will be actively present in the environment.
        Other robots will only exist in the observation space, not in the actual environment.
        """
        pos_frame = Frame()
        x_limit = self.field.length / 2 - 0.2
        y_limit = self.field.width / 2 - 0.2

        # Randomly sample the target position within the field limits
        self.target_position = Position(
            x=np.random.uniform(-x_limit, x_limit),
            y=np.random.uniform(-y_limit, y_limit),
        )

        # Place the agent (blue[0]) at a random position
        pos_frame.robots_blue = {}
        robot_x = np.random.uniform(-x_limit, x_limit)
        robot_y = np.random.uniform(-y_limit, y_limit)
        pos_frame.robots_blue[0] = Robot(
            yellow=False,
            id=0,
            x=robot_x,
            y=robot_y,
            theta=np.random.uniform(-180, 180),
        )
        
        # Don't add other robots to the environment, only to observations
        # Other robots will be dummies only in observations
        
        # Reset metrics for the new episode
        self.steps_to_target = 0
        
        # Calculate the initial distance between the robot and the target
        self.initial_distance = np.linalg.norm([robot_x - self.target_position.x, robot_y - self.target_position.y])
        
        # Ball is not used actively but kept for compatibility (set at origin)
        pos_frame.ball = Ball(x=0, y=0)
        
        # Initialize empty dictionary for yellow robots
        pos_frame.robots_yellow = {}

        return pos_frame

    def _frame_to_observations(self):
        """
        Converts the current environment state to the observation vector.
        Observation vector:
        [ball(4), blue[0](8), blue[1:3](2x2), yellow robots(3x2), target(3)] = 23 elements
        Robots other than blue[0] are fictitious and exist only in observations.
        """
        obs = []

        # Ball: [x, y, v_x, v_y] (normalized)
        ball = self.frame.ball
        obs.extend([
            self.norm_pos(ball.x), self.norm_pos(ball.y),
            self.norm_v(ball.v_x), self.norm_v(ball.v_y)
        ])

        # Blue robot 0 (controlled agent): 8 values
        if 0 in self.frame.robots_blue:
            robot = self.frame.robots_blue[0]
            obs.extend([
                self.norm_pos(robot.x), self.norm_pos(robot.y),
                np.sin(np.deg2rad(robot.theta)), np.cos(np.deg2rad(robot.theta)),
                self.norm_v(robot.v_x), self.norm_v(robot.v_y),
                self.norm_w(robot.v_theta),
                1 if robot.infrared else 0
            ])
        else:
            # Dummy values for absent blue robot 0
            obs.extend([0, 0, 0, 1, 0, 0, 0, 0])

        # Other blue robots (teammates): only x, y - fictitious values
        for i in range(1, self.n_robots_blue_obs):
            # Fictitious values for uncontrolled blue robots
            x_pos = np.random.uniform(-self.field.length/2, self.field.length/2)
            y_pos = np.random.uniform(-self.field.width/2, self.field.width/2)
            obs.extend([
                self.norm_pos(x_pos), self.norm_pos(y_pos)
            ])

        # Yellow robots (opponents): only x, y - fictitious values
        for i in range(self.n_robots_yellow_obs):
            # Fictitious values for yellow robots
            x_pos = np.random.uniform(-self.field.length/2, self.field.length/2)
            y_pos = np.random.uniform(-self.field.width/2, self.field.width/2)
            obs.extend([
                self.norm_pos(x_pos), self.norm_pos(y_pos)
            ])

        # Target: [x, y, theta] (normalized)
        target = self.target_position
        # We use a fixed angle for the target (0 degrees)
        target_theta = 0.0
        obs.extend([
            self.norm_pos(target.x), 
            self.norm_pos(target.y),
            self.norm_w(target_theta)  # Using the same normalization as other angles
        ])

        return np.array(obs, dtype=np.float32)

    def _get_commands(self, action):
        """
        Converts the action vector into robot commands for the agent (blue[0]).
        The action vector is expected to be [v_x, v_y, v_theta, kick_v_x, dribbler], all normalized to [-1, 1].
        Returns a list of Robot commands (only the agent is controlled here).
        At this initial stage, kick_v_x and dribbler are always 0, regardless of the action.
        """
        angle = self.frame.robots_blue[0].theta
        v_x, v_y, v_theta = self.convert_actions(action[:3], np.deg2rad(angle))
        
        return [Robot(
            yellow=False, 
            id=0, 
            v_x=v_x, 
            v_y=v_y, 
            v_theta=v_theta,
            kick_v_x=0.0,  # Always 0, regardless of the action
            dribbler=False  # Always False, regardless of the action
        )]
        
    def convert_actions(self, action, angle):
        """Denormalize, clip to absolute max and convert to local"""
        # Denormalize
        v_x = action[0] * self.max_v
        v_y = action[1] * self.max_v
        v_theta = action[2] * self.max_w
        
        # Convert to local
        v_x, v_y = v_x*np.cos(angle) + v_y*np.sin(angle), \
            -v_x*np.sin(angle) + v_y*np.cos(angle)

        # clip by max absolute
        v_norm = np.linalg.norm([v_x, v_y])
        c = v_norm < self.max_v or self.max_v / v_norm
        v_x, v_y = v_x*c, v_y*c
        
        return v_x, v_y, v_theta

    def _calculate_reward_and_done(self):
        """
        Calculates the reward and termination condition for the current step.
        Simplified reward: reach target + time efficiency penalty
        """
        robot = self.frame.robots_blue[0]
        target = self.target_position

        # Calculate distance to target (only for verification, not for reward)
        dist_to_target = np.linalg.norm([robot.x - target.x, robot.y - target.y])

        # Base reward is zero (we don't use distance)
        reward = 0.0

        done = False
        self.steps_to_target += 1
        
        # Check if the robot is out of bounds
        field_length_half = self.field.length / 2
        field_width_half = self.field.width / 2
        out_of_bounds = (
            abs(robot.x) > field_length_half or 
            abs(robot.y) > field_width_half
        )
        
        # Check termination conditions and calculate rewards
        if out_of_bounds:
            # Penalty for going out of bounds
            reward += self.penalty_out_of_bounds
            
            # Additional penalty based on distance to target
            dist_penalty = -dist_to_target * 5.0
            reward += dist_penalty
            
            done = True
        
        # Check if the robot reached the target
        success = dist_to_target < 0.09
        if success:
            # Base reward for reaching the target
            reward += self.reward_target_reached
            
            # Additional reward based on time normalized by initial distance
            # Calculate the maximum expected steps based on initial distance
            expected_max_steps = self.initial_distance * self.max_steps_per_meter
            
            # The faster the robot reaches the target (fewer steps), the higher the reward
            if expected_max_steps > 0:
                time_efficiency = max(0, 1 - (self.steps_to_target / expected_max_steps))
                time_reward = self.reward_target_reached * time_efficiency
                reward += time_reward
            
            done = True
        
        # Check if maximum steps reached
        if self.steps_to_target >= self.max_steps:
            # Penalty for timeout
            reward += self.penalty_timeout
            
            # Additional penalty based on final distance to target
            dist_penalty = -dist_to_target * 5.0
            reward += dist_penalty
            
            done = True
        
        # Penalty for each step (encourages the agent to be faster)
        reward += self.penalty_per_step
        
        # Store metrics for logging
        self.episodes_metrics = {
            'success': success,
            'out_of_bounds': out_of_bounds,
            'timeout': self.steps_to_target >= self.max_steps,
            'initial_distance': self.initial_distance,
            'final_distance': dist_to_target,
            'steps_to_target': self.steps_to_target,
            'steps_per_meter': self.steps_to_target / self.initial_distance if self.initial_distance > 0 else 0,
            'episode_reward': reward
        }
        
        # Create info to return to the environment
        info = {}
        if done:
            info['episode_metrics'] = self.episodes_metrics.copy()
        
        return reward, done

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if terminated or truncated:
            info['episode_metrics'] = self.episodes_metrics.copy()
        return obs, reward, terminated, truncated, info

    def _draw_target(self, target_pos, screen):
        """
        Draws the target area as a square on the field during rendering.
        The square is sized so that the robot (radius 0.09m) can fit entirely inside.
        """
        # Robot radius in meters
        robot_radius = self.field.rbt_radius  # Should be 0.09m
        # Size of the square (side = 2 * robot_radius)
        square_side_m = 2 * robot_radius

        # Convert field coordinates to screen coordinates
        center_x, center_y = target_pos
        scale = self.field_renderer.scale

        # Calculate top-left corner of the square in screen coordinates
        top_left_x = int(center_x * scale + self.field_renderer.center_x - (square_side_m * scale) / 2)
        top_left_y = int(center_y * scale + self.field_renderer.center_y - (square_side_m * scale) / 2)
        square_side_px = int(square_side_m * scale)

        # Draw filled red square
        pygame.draw.rect(screen, COLORS["RED"], (top_left_x, top_left_y, square_side_px, square_side_px))
        # Draw black border
        pygame.draw.rect(screen, COLORS["BLACK"], (top_left_x, top_left_y, square_side_px, square_side_px), 2)
        
    def _render(self):
        super()._render()
        # Draw the target area as a square
        self._draw_target([self.target_position.x, self.target_position.y], self.window_surface)