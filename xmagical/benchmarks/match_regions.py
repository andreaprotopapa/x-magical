import math

from gym.utils import EzPickle
from gym import spaces

import numpy as np

#from magical.base_env import BaseEnv, ez_init
#import magical.entities as en
#import magical.geom as geom

from typing import Any, Dict, Tuple
from xmagical.base_env import BaseEnv as BaseEnvXirl
import xmagical.geom as geom
import xmagical.entities as en

colors = {
          "red": en.ShapeColor.RED,
          "green": en.ShapeColor.GREEN,
          "blue": en.ShapeColor.BLUE,
          "yellow": en.ShapeColor.YELLOW
          }

shapes = {
            "square": en.ShapeType.SQUARE,
            "pentagon": en.ShapeType.PENTAGON,
            "star": en.ShapeType.STAR,
            "circle": en.ShapeType.CIRCLE
        }

# Max possible L2 distance (arena diagonal 2*sqrt(2)).
D_MAX = 2.8284271247461903

class MatchRegionsEnv(BaseEnvXirl):
    """Need to push blocks of a certain colour to the corresponding coloured
    region. Aim is for the robot to generalise _that_ rule instead of
    generalising others (e.g. "always move squares to this position" or
    something).
    Args:
            use_state: Whether to use states rather than pixels for the
                observation space.
            use_dense_reward: Whether to use a dense reward or a sparse one.
            rand_layout_full: Whether to randomize the poses of the debris and goal position.
            rand_shapes: Whether to randomize the shapes (and the number) of the debris.
            rand_colors: Whether to randomize the colors of the debris and the
                goal zone.
    """
    #@ez_init()
    def __init__(
            self,
            use_state: bool = False,
            use_dense_reward: bool = False,
            config: dict = None, 
            rand_colors: bool = False,
            rand_shapes: bool = False,
            # shape_randomisation=ShapeRandLevel.NONE,
            rand_target_colour=False,
            rand_shape_type=False,
            rand_shape_count=False,
            rand_layout_minor=False,
            rand_layout_full=False,
            **kwargs):
        super().__init__(**kwargs)
        self.use_state = use_state
        self.use_dense_reward = use_dense_reward
        self.config = config
        # self.rand_target_colour = rand_target_colour EDIT!
        # self.shape_randomisation = shape_randomisation
        # self.rand_shape_type = rand_shape_type  EDIT!
        # self.rand_shape_count = rand_shape_count  EDIT!
        # self.rand_layout_minor = rand_layout_minor  EDIT!
        # self.rand_layout_full = rand_layout_full  EDIT!
        self.max_episode_steps = self.config["max_episode_steps"]
        self.rand_target_colour = rand_colors
        self.rand_shape_type = rand_shapes
        self.rand_shape_count = False
        self.rand_layout_minor = False
        self.rand_layout_full = rand_layout_full
        self.init_cost = None
        if rand_shapes:
            self.rand_target_colour = False #True
            self.rand_layout_full = False #True

        if self.rand_shape_count:
            assert self.rand_layout_full, \
                "if shape count is randomised then layout must also be " \
                "fully randomised"
            assert self.rand_shape_type, \
                "if shape count is randomised then shape type must also be " \
                "randomised"
            assert self.rand_target_colour, \
                "if shape count is randomised then shape colour must also " \
                "be randomised"
            
        if self.use_state:
            # Redefine the observation space if we are using states as opposed
            # to pixels.
            max_num_debris = 2 + 3
            c = 4 if self.action_dim == 2 else 5
            self.observation_space = spaces.Box(
                np.array([-1] * (c + 4 * max_num_debris), dtype=np.float32),
                np.array([+1] * (c + 4 * max_num_debris), dtype=np.float32),
                dtype=np.float32,
            )
    def on_reset_default(self):
        # make the robot
        robot_pos = np.asarray((-0.5, 0.1))
        robot_angle = -math.pi * 1.2
        # if necessary, robot pose is randomised below
        robot = self._make_robot(robot_pos, robot_angle)

        # set up target colour/region/pose
        if self.rand_target_colour:
            target_colour = self.rng.choice(en.SHAPE_COLORS)
        else:
            target_colour = en.ShapeColor.GREEN
        distractor_colours = [
            c for c in en.SHAPE_COLORS if c != target_colour
        ]
        target_h = 0.7
        target_w = 0.6
        target_x = 0.1
        target_y = 0.7
        if self.rand_layout_minor or self.rand_layout_full:
            if self.rand_layout_minor:
                hw_bound = self.JITTER_TARGET_BOUND
            else:
                hw_bound = None
            target_h, target_w = geom.randomise_hw(self.RAND_GOAL_MIN_SIZE,
                                                   self.RAND_GOAL_MAX_SIZE,
                                                   self.rng,
                                                   current_hw=(target_h,
                                                               target_w),
                                                   linf_bound=hw_bound)
        sensor = en.GoalRegion(target_x, target_y, target_h, target_w,
                               target_colour)
        self.add_entities([sensor])
        self.__sensor_ref = sensor

        # set up spec for remaining blocks
        default_target_types = [
            en.ShapeType.STAR,
            en.ShapeType.SQUARE,
        ]
        default_distractor_types = [
            [],
            [en.ShapeType.PENTAGON],
            [en.ShapeType.CIRCLE, en.ShapeType.PENTAGON],
        ]
        default_target_poses = [
            # (x, y, theta)
            (0.8, -0.7, 2.37),
            (-0.68, 0.72, 1.28),
        ]
        default_distractor_poses = [
            # (x, y, theta)
            [],
            [(-0.05, -0.2, -1.09)],
            [(-0.75, -0.55, 2.78), (0.3, -0.82, -1.15)],
        ]

        if self.rand_shape_count:
            target_count = self.rng.randint(1, 2 + 1)
            distractor_counts = [
                self.rng.randint(0, 2 + 1) for c in distractor_colours
            ]
        else:
            target_count = len(default_target_types)
            distractor_counts = [len(lst) for lst in default_distractor_types]

        if self.rand_shape_type:
            shape_types_np = np.asarray(en.SHAPE_TYPES, dtype='object')
            target_types = [
                self.rng.choice(shape_types_np) for _ in range(target_count)
            ]
            distractor_types = [[
                self.rng.choice(shape_types_np) for _ in range(dist_count)
            ] for dist_count in distractor_counts]
        else:
            target_types = default_target_types
            distractor_types = default_distractor_types

        if self.rand_layout_full:
            # will do post-hoc randomisation at the end
            target_poses = [(0, 0, 0)] * target_count
            distractor_poses = [[(0, 0, 0)] * dcount
                                for dcount in distractor_counts]
        else:
            target_poses = default_target_poses
            distractor_poses = default_distractor_poses

        assert len(target_types) == target_count
        assert len(target_poses) == target_count
        assert len(distractor_types) == len(distractor_counts)
        assert len(distractor_types) == len(distractor_colours)
        assert len(distractor_types) == len(distractor_poses)
        assert all(
            len(types) == dcount
            for types, dcount in zip(distractor_types, distractor_counts))
        assert all(
            len(poses) == dcount
            for poses, dcount in zip(distractor_poses, distractor_counts))

        self.__target_shapes = [
            self._make_shape(shape_type=shape_type,
                             color_name=target_colour,
                             init_pos=(shape_x, shape_y),
                             init_angle=shape_angle)
            for shape_type, (shape_x, shape_y,
                             shape_angle) in zip(target_types, target_poses)
        ]
        self.__distractor_shapes = []
        for dist_colour, dist_types, dist_poses \
                in zip(distractor_colours, distractor_types, distractor_poses):
            for shape_type, (shape_x, shape_y, shape_angle) \
                    in zip(dist_types, dist_poses):
                dist_shape = self._make_shape(shape_type=shape_type,
                                              color_name=dist_colour,
                                              init_pos=(shape_x, shape_y),
                                              init_angle=shape_angle)
                self.__distractor_shapes.append(dist_shape)
        shape_ents = self.__target_shapes + self.__distractor_shapes

        self.__debris_shapes = shape_ents
        self.add_entities(shape_ents)

        if self.use_state:
            # Redefine the observation space if we are using states as opposed
            # to pixels.
            c = 4 if self.action_dim == 2 else 5
            self.observation_space = spaces.Box(
                np.array([-1] * (c + 4 * len(self.__debris_shapes)), dtype=np.float32),
                np.array([+1] * (c + 4 * len(self.__debris_shapes)), dtype=np.float32),
                dtype=np.float32,
            )

        # add this last so it shows up on top, but before layout randomisation,
        # since it needs to be added to the space before randomising
        self.add_entities([robot])

        if self.rand_layout_minor or self.rand_layout_full:
            all_ents = (sensor, robot, *shape_ents)
            if self.rand_layout_minor:
                # limit amount by which position and rotation can be randomised
                pos_limits = self.JITTER_POS_BOUND
                rot_limits = self.JITTER_ROT_BOUND
            else:
                # no limits, can randomise as much as needed
                assert self.rand_layout_full
                pos_limits = rot_limits = None
            # randomise rotations of all entities but goal region
            rand_rot = [False] + [True] * (len(all_ents) - 1)

            geom.pm_randomise_all_poses(self._space,
                                        all_ents,
                                        self.ARENA_BOUNDS_LRBT,
                                        rng=self.rng,
                                        rand_pos=True,
                                        rand_rot=rand_rot,
                                        rel_pos_linf_limits=pos_limits,
                                        rel_rot_limits=rot_limits)

        # set up index for lookups
        self.__ent_index = en.EntityIndex(shape_ents)

    def on_reset(self):
        # make the robot
        # robot_pos = np.asarray((-0.5, 0.1))
        # robot_angle = -math.pi * 1.2
        idx = self.rng.choice([0, 1]) if self.config["randomize_robot_pose"] else 0
        robot_pos = np.asarray(self.config["robot"]["pos"][idx])
        robot_angle = -math.pi * self.config["robot"]["rot"][idx]
        print(f"Config {idx}: Robot position: {robot_pos}, angle: {robot_angle}")

        # if necessary, robot pose is randomised below
        robot = self._make_robot(robot_pos, robot_angle)
        # set up target colour/region/pose
        if self.rand_target_colour:
            target_colour = self.rng.choice(en.SHAPE_COLORS)
        else:
            target_colour = colors[self.config["target_color"]]
        distractor_colours = [
            c for c in en.SHAPE_COLORS if c != target_colour
        ]
        target_h = 0.7
        target_w = 0.6
        target_x = 0.1
        target_y = 0.7
        if self.rand_layout_minor or self.rand_layout_full:
            if self.rand_layout_minor:
                hw_bound = self.JITTER_TARGET_BOUND
            else:
                hw_bound = None
            target_h, target_w = geom.randomise_hw(self.RAND_GOAL_MIN_SIZE,
                                                   self.RAND_GOAL_MAX_SIZE,
                                                   self.rng,
                                                   current_hw=(target_h,
                                                               target_w),
                                                   linf_bound=hw_bound)
        sensor = en.GoalRegion(target_x, target_y, target_h, target_w,
                               target_colour)
        self.add_entities([sensor])
        self.__sensor_ref = sensor

        # set up spec for remaining blocks
        default_target_types = [
            shapes[self.config["target"][i]['shape']] for i in range(len(self.config["target"]))
        ]
        default_distractor_types = [
            [shapes[self.config["distractor"][i]['shape']] for i in range(len(self.config["distractor"])) if colors[self.config["distractor"][i]['color']]=='red'],
            [shapes[self.config["distractor"][i]['shape']] for i in range(len(self.config["distractor"])) if colors[self.config["distractor"][i]['color']]=='blue'],
            [shapes[self.config["distractor"][i]['shape']] for i in range(len(self.config["distractor"])) if colors[self.config["distractor"][i]['color']]=='yellow'],
        ]
        default_target_poses = [
            # (x, y, theta)
            tuple(self.config["target"][i]['pos'][idx]) for i in range(len(self.config["target"]))
        ]
        default_distractor_poses = [
            # (x, y, theta)
            [tuple(self.config["distractor"][i]['pos']) for i in range(len(self.config["distractor"])) if colors[self.config["distractor"][i]['color']]=='red'],
            [tuple(self.config["distractor"][i]['pos']) for i in range(len(self.config["distractor"])) if colors[self.config["distractor"][i]['color']]=='blue'],
            [tuple(self.config["distractor"][i]['pos']) for i in range(len(self.config["distractor"])) if colors[self.config["distractor"][i]['color']]=='yellow'],
        ]

        if self.rand_shape_count:
            target_count = self.rng.randint(1, 2 + 1)
            distractor_counts = [
                self.rng.randint(0, 2 + 1) for c in distractor_colours
            ]
        else:
            target_count = len(default_target_types)
            distractor_counts = [len(lst) for lst in default_distractor_types]

        if self.rand_shape_type:
            shape_types_np = np.asarray(en.SHAPE_TYPES, dtype='object')
            target_types = [
                self.rng.choice(shape_types_np) for _ in range(target_count)
            ]
            distractor_types = [[
                self.rng.choice(shape_types_np) for _ in range(dist_count)
            ] for dist_count in distractor_counts]
        else:
            target_types = default_target_types
            distractor_types = default_distractor_types

        if self.rand_layout_full:
            # will do post-hoc randomisation at the end
            target_poses = [(0, 0, 0)] * target_count
            distractor_poses = [[(0, 0, 0)] * dcount
                                for dcount in distractor_counts]
        else:
            target_poses = default_target_poses
            distractor_poses = default_distractor_poses

        assert len(target_types) == target_count
        assert len(target_poses) == target_count
        assert len(distractor_types) == len(distractor_counts)
        assert len(distractor_types) == len(distractor_colours)
        assert len(distractor_types) == len(distractor_poses)
        assert all(
            len(types) == dcount
            for types, dcount in zip(distractor_types, distractor_counts))
        assert all(
            len(poses) == dcount
            for poses, dcount in zip(distractor_poses, distractor_counts))

        self.__target_shapes = [
            self._make_shape(shape_type=shape_type,
                             color_name=target_colour,
                             init_pos=(shape_x, shape_y),
                             init_angle=shape_angle)
            for shape_type, (shape_x, shape_y,
                             shape_angle) in zip(target_types, target_poses)
        ]
        self.__distractor_shapes = []
        for dist_colour, dist_types, dist_poses \
                in zip(distractor_colours, distractor_types, distractor_poses):
            for shape_type, (shape_x, shape_y, shape_angle) \
                    in zip(dist_types, dist_poses):
                dist_shape = self._make_shape(shape_type=shape_type,
                                              color_name=dist_colour,
                                              init_pos=(shape_x, shape_y),
                                              init_angle=shape_angle)
                self.__distractor_shapes.append(dist_shape)
        shape_ents = self.__target_shapes + self.__distractor_shapes
        self.__debris_shapes = shape_ents
        self.add_entities(shape_ents)

        if self.use_state:
            # Redefine the observation space if we are using states as opposed
            # to pixels.
            c = 4 if self.action_dim == 2 else 5
            self.observation_space = spaces.Box(
                np.array([-1] * (c + 4 * len(self.__debris_shapes)), dtype=np.float32),
                np.array([+1] * (c + 4 * len(self.__debris_shapes)), dtype=np.float32),
                dtype=np.float32,
            )

        # add this last so it shows up on top, but before layout randomisation,
        # since it needs to be added to the space before randomising
        self.add_entities([robot])

        if self.rand_layout_minor or self.rand_layout_full:
            all_ents = (sensor, robot, *shape_ents)
            if self.rand_layout_minor:
                # limit amount by which position and rotation can be randomised
                pos_limits = self.JITTER_POS_BOUND
                rot_limits = self.JITTER_ROT_BOUND
            else:
                # no limits, can randomise as much as needed
                assert self.rand_layout_full
                pos_limits = rot_limits = None
            # randomise rotations of all entities but goal region
            rand_rot = [False] + [True] * (len(all_ents) - 1)

            geom.pm_randomise_all_poses(self._space,
                                        all_ents,
                                        self.ARENA_BOUNDS_LRBT,
                                        rng=self.rng,
                                        rand_pos=True,
                                        rand_rot=rand_rot,
                                        rel_pos_linf_limits=pos_limits,
                                        rel_rot_limits=rot_limits)

        # set up index for lookups
        self.__ent_index = en.EntityIndex(shape_ents)
        self.goal_pos = self.__sensor_ref.goal_body.position
        self.init_cost = {
            target_shape: np.linalg.norm(target_shape.shape_body.position - self.goal_pos)
            for target_shape in self.__target_shapes
        }

    def score_on_end_of_traj(self):
        overlap_ents = self.__sensor_ref.get_overlapping_ents(
            com_overlap=True, ent_index=self.__ent_index)
        target_set = set(self.__target_shapes)
        distractor_set = set(self.__distractor_shapes)
        n_overlap_targets = len(target_set & overlap_ents)
        n_overlap_distractors = len(distractor_set & overlap_ents)
        # what fraction of targets are in the overlap set?
        target_frac_done = n_overlap_targets / len(target_set)
        if len(overlap_ents) == 0:
            contamination_rate = 0
        else:
            # what fraction of the overlap set are distractors?
            contamination_rate = n_overlap_distractors / len(overlap_ents)
        # score guide:
        # - 1 if all target shapes and no distractors are in the target region
        # - 0 if no target shapes in the target region
        # - somewhere in between if some are all target shapes are there, but
        #   there's also contamination (more contamination = worse, fewer
        #   target shapes in target region = worse).
        return target_frac_done * (1 - contamination_rate)
    
###################### NEW STUFF ######################

    def _simplified_reward(self) -> float:
        reward = 0.0

        if not hasattr(self, "_placed_targets"):
            self._placed_targets = set()

        overlap_ents = self.__sensor_ref.get_overlapping_ents(
            ent_index=self.__ent_index, com_overlap=True
        )

        # Penalize distractors in goal region
        distractor_set = set(self.__distractor_shapes)
        n_overlap_distractors = len(distractor_set & overlap_ents)
        reward -= 1.0 * n_overlap_distractors

        # Reward for each target moving toward the goal
        for target_shape in self.__target_shapes:
            target_pos = target_shape.shape_body.position
            dist = np.linalg.norm(target_pos - self.goal_pos)
            init_dist = self.init_cost[target_shape]

            if target_shape in overlap_ents:
                dist = 0.0  # full reward if in goal

            reward += (init_dist - dist)  # linear progress reward

        reward -= 0.1  # step penalty

        # Optional: smooth bounded normalization
        scale = len(self.__target_shapes)  # or average expected distance
        norm_reward = 10.0 * np.tanh(reward / scale)

        return norm_reward

    def _simplified_reward_with_proximity(self) -> float:

        reward = 0.0

        # Initialize placed_targets tracking if not done
        if not hasattr(self, "_placed_targets"):
            self._placed_targets = set()

        # Get which entities are in the goal region based on their center of mass
        overlap_ents = self.__sensor_ref.get_overlapping_ents(
            ent_index=self.__ent_index, com_overlap=True
        )

        # Distractor penalty
        distractor_set = set(self.__distractor_shapes)
        n_overlap_distractors = len(distractor_set & overlap_ents)
        reward -= 1.0 * n_overlap_distractors

        for target_shape in self.__target_shapes:
            target_pos = target_shape.shape_body.position     
            dist_to_goal = np.linalg.norm(target_pos - self.goal_pos)
            init_dist = self.init_cost[target_shape]

            if target_shape in overlap_ents:
                dist_to_goal = 0.0  # No distance penalty if target is in goal
                dist_to_target = 0.0  # No distance to target if in goal
            else:
                robot_pos = self._robot.body.position
                dist_to_target = np.linalg.norm(target_pos - robot_pos)        
            
            reward += self._distance_reward(dist_to_goal) + self._distance_reward(dist_to_target)  # linear progress reward
        
        reward -= 0.1  # step penalty
        print(reward)
        # Optional: smooth bounded normalization
        scale = len(self.__target_shapes)  # or average expected distance
        norm_reward = 10.0 * np.tanh(reward / scale)

        return norm_reward

    def _distance_reward(self, d, alpha=0.006, beta=500, gamma=1e-3):
        return -alpha * d**2 - beta * np.log(d**2 + gamma)

    def _dense_reward(self) -> float:
        """Mean distance of all TARGET entitity positions to goal zone MULTIPLIED for the CONTAMINATION RATE."""
        # Goal position
        goal_x = self.__sensor_ref.goal_body.position[0]
        goal_y = self.__sensor_ref.goal_body.position[1]
        goal_pos = (goal_x, goal_y)

        # Proximity bonus: encourage robot to get closer to the cubes
        proximity_bonus = 0
        min_distance_to_cube = float('inf')

        # Calculate the mean distance of targets to the goal zone
        target_goal_dists = []
        robot_pos = self._robot.body.position

        for target_shape in self.__target_shapes:
            target_pos = target_shape.shape_body.position
            dist_to_goal  = np.linalg.norm(target_pos - goal_pos)
            target_goal_dists.append(dist_to_goal )

            # Check if the cube is outside the goal region
            if len(self.__sensor_ref.get_overlapping_ents(ent_index=en.EntityIndex([target_shape]), com_overlap=True,)) == 0:
                # Calculate distance between robot and the cube
                dist_to_cube = np.linalg.norm((target_pos - robot_pos) / D_MAX)
                min_distance_to_cube = min(min_distance_to_cube, dist_to_cube)

                # Encourage proximity to cubes (inversely related to the distance)
                proximity_bonus += max(0, 1 - min_distance_to_cube)

        mean_dist = np.mean(target_goal_dists)
        normalized_dist = mean_dist / D_MAX

        # Add discrete bonus if target cube are completely in the goal region
        overlap_ents = self.__sensor_ref.get_overlapping_ents(
            com_overlap=True, ent_index=self.__ent_index)
        target_set = set(self.__target_shapes)
        n_overlap_target = len(target_set & overlap_ents)
        if len(overlap_ents) == 0:
            precise_placement_bonus = 0
        else:
            precise_placement_bonus = n_overlap_target / len(overlap_ents)


        # Calculate contamination rate    
        distractor_set = set(self.__distractor_shapes)
        n_overlap_distractors = len(distractor_set & overlap_ents)

        if len(overlap_ents) == 0:
            contamination_rate = 0
        else:
            # what fraction of the overlap set are distractors?
            contamination_rate = n_overlap_distractors / len(overlap_ents)
        
        # Weaken the effect of contamination by taking the square root
        reduced_contamination_rate = np.sqrt(contamination_rate)

        # Adjust the reward formula
        reward = -normalized_dist * (1 + reduced_contamination_rate)

        scaling_factor = 10.0
        reward *= scaling_factor # Apply a scaling factor to ensure reward values are more distinguishable
        reward += proximity_bonus  # Encourage approaching cubes
        reward += precise_placement_bonus * 10.0  # Increase weight of precise placement bonus
        reward += 10.0 * n_overlap_target # Add explicit reward per target placed
        reward -= 5.0 * n_overlap_distractors # explicit penalty per distractor in goal

        # Optionally add a small positive/negative baseline to ensure reward is not too negative
        reward += 4

        return reward
    
    def _sparse_reward(self) -> float:
        """Fraction of debris entities inside goal zone."""
        # `score_on_end_of_traj` is supposed to be called at the end of a
        # trajectory but we use it here since it gives us exactly the reward
        # we're looking for.
        return self.score_on_end_of_traj()
    
    def get_reward(self) -> float:
        if self.use_dense_reward:
            # return self._dense_reward()
            return self._simplified_reward()
            # return self._simplified_reward_with_proximity()
        return self._sparse_reward()
    
    def get_state(self) -> np.ndarray:
        robot_pos = self._robot.body.position
        robot_angle_cos = np.cos(self._robot.body.angle)
        robot_angle_sin = np.sin(self._robot.body.angle)

        goal_x = self.__sensor_ref.goal_body.position[0]
        goal_y = self.__sensor_ref.goal_body.position[1]
        gpos = (goal_x, goal_y)

        # TARGET block features
        target_pos = []
        robot_target_dist = []
        target_goal_dist = []
        for target_shape in self.__target_shapes:
            tpos = target_shape.shape_body.position
            target_pos.extend(tuple(tpos))  # x, y
            robot_target_dist.append(np.linalg.norm(robot_pos - tpos) / D_MAX)
            target_goal_dist.append(np.linalg.norm(tpos - gpos) / D_MAX)
        
        # Optional: DISTRACTOR features (separate section)
        distractor_pos = []
        robot_distractor_dist = []
        distractor_goal_dist = []
        for distractor_shape in self.__distractor_shapes:
            dpos = distractor_shape.shape_body.position
            distractor_pos.extend(tuple(dpos))
            robot_distractor_dist.append(np.linalg.norm(robot_pos - dpos) / D_MAX)
            distractor_goal_dist.append(np.linalg.norm(dpos - gpos) / D_MAX)
        
        state = [
            *tuple(robot_pos),             # 2
            *target_pos,                   # 2 * num_targets
            *distractor_pos,              # 2 * num_distractors
            robot_angle_cos,              # 1
            robot_angle_sin,              # 1
            *robot_target_dist,           # num_targets
            *target_goal_dist,            # num_targets
            *robot_distractor_dist,       # num_distractors
            *distractor_goal_dist,        # num_distractors
        ]
        
        if self.action_dim == 3:
            state.append(self._robot.finger_width)

        return np.array(state, dtype=np.float32)

    def reset(self) -> np.ndarray:
        obs = super().reset()
        if self.use_state:
            return self.get_state()
        return obs

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs, rew, done, info = super().step(action)
        if self.use_state:
            obs = self.get_state()
        return obs, rew, done, info
