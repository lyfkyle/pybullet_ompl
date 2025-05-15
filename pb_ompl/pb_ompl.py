import math
from functools import partial

import matplotlib
import numpy as np
from matplotlib import cm, pyplot as plt
from scipy.spatial.transform import Rotation

np.set_printoptions(precision=3, suppress=True)
import pinocchio
from scipy import interpolate

# if the ompl module is not in the PYTHONPATH assume it is installed in a
# subdirectory of the parent directory called "py-bindings."
from os.path import abspath, dirname, join
import sys

from pb_ompl import utils

sys.path.insert(0, join(dirname(dirname(abspath(__file__))), 'ompl/py-bindings'))
# print(*sys.path, sep='\n')

from ompl import base as ob
from ompl import geometric as og
import pybullet as p
import time
from itertools import product
import copy

INTERPOLATE_NUM = 500
DEFAULT_PLANNING_TIME = 5.0


class PbOMPLRobot:
    '''
    To use with Pb_OMPL. You need to construct a instance of this class and pass to PbOMPL.

    Note:
    This parent class by default assumes that all joints are acutated and should be planned. If this is not your desired
    behaviour, please write your own inheritated class that overrides respective functionalities.
    '''
    def __init__(self, pybullet_client, id, urdf_path=None, link_name_ee=None) -> None:
        # Public attributes
        self.pybullet_client = pybullet_client
        self.id = id

        assert link_name_ee is not None, "Please specify end-effector link name"
        self.link_name_ee = link_name_ee

        # prune fixed joints
        all_joint_num = self.pybullet_client.getNumJoints(id)
        all_joint_idx = list(range(all_joint_num))
        joint_idx = [j for j in all_joint_idx if self._is_not_fixed(j)]
        self.num_dim = len(joint_idx)
        self.joint_idx = joint_idx
        print(self.joint_idx)
        self.joint_bounds = []

        # for sampling inside joint bounds
        self.joint_bounds_np = np.array(self.get_joint_bounds())
        self.joint_bounds_low_np = self.joint_bounds_np[:, 0]
        self.joint_bounds_low_l = self.joint_bounds_low_np.tolist()
        self.joint_bounds_high_np = self.joint_bounds_np[:, 1]
        self.joint_bounds_high_l = self.joint_bounds_high_np.tolist()
        self.joint_ranges_np = self.joint_bounds_high_np - self.joint_bounds_low_np
        self.joint_ranges_l = self.joint_ranges_np.tolist()

        # get link name to index
        self._link_name_to_index = {self.pybullet_client.getBodyInfo(id)[0].decode('UTF-8'): -1, }
        for _id in range(self.pybullet_client.getNumJoints(id)):
            _name = self.pybullet_client.getJointInfo(id, _id)[12].decode('UTF-8')
            self._link_name_to_index[_name] = _id

        self.link_ee_idx = self._link_name_to_index[self.link_name_ee]

        # pinocchio
        self.pinocchio_robot_model = None
        if urdf_path is not None:
            self.pinocchio_robot_model = pinocchio.buildModelFromUrdf(urdf_path)
            self.pinocchio_robot_model_data = self.pinocchio_robot_model.createData()
            self.pinocchio_ee_frameid = self.pinocchio_robot_model.getFrameId(self.link_name_ee)
            self.pinocchio_ee_parent_joint_id = self.pinocchio_robot_model.frames[self.pinocchio_ee_frameid].parent

        ############################################################
        self.reset()

    def _is_not_fixed(self, joint_idx):
        joint_info = self.pybullet_client.getJointInfo(self.id, joint_idx)
        return joint_info[2] != self.pybullet_client.JOINT_FIXED

    def get_joint_bounds(self):
        '''
        Get joint bounds.
        By default, read from pybullet
        '''
        joint_bounds = []
        for i, joint_id in enumerate(self.joint_idx):
            joint_info = self.pybullet_client.getJointInfo(self.id, joint_id)
            low = joint_info[8]  # lower bounds
            high = joint_info[9]  # higher bounds
            if low < high:
                joint_bounds.append([low, high])
        # print("Joint bounds: {}".format(joint_bounds))
        self.joint_bounds = joint_bounds
        return self.joint_bounds

    def get_cur_state(self):
        return copy.deepcopy(self.state)

    def set_state(self, state):
        '''
        Set robot state.
        To faciliate collision checking
        Args:
            state: list[Float], joint values of robot
        '''
        self._set_joint_positions(self.joint_idx, state)
        self.state = state

    def reset(self):
        '''
        Reset robot state
        Args:
            state: list[Float], joint values of robot
        '''
        state = [0] * self.num_dim
        self._set_joint_positions(self.joint_idx, state)
        self.state = state

    def _set_joint_positions(self, joints, positions):
        for joint, value in zip(joints, positions):
            self.pybullet_client.resetJointState(self.id, joint, value, targetVelocity=0)

    def get_random_joint_position(self):
        rv = np.random.random(self.num_dim)
        state = self.joint_bounds_low_np + rv * (self.joint_bounds_high_np - self.joint_bounds_low_np)
        return state

    def get_ee_pose(self, state, quat_wxyz=False, return_transformation=False, **kwargs):
        """
        Get end-effector pose
        """
        self.set_state(self.state_to_list(state))
        position, orientation = self.pybullet_client.getLinkState(
            self.id, self.link_ee_idx,
            # computeLinkVelocity=0, computeForwardKinematics=1
        )[:2]
        if quat_wxyz:  # orientation in pybullet is xyzw
            orientation = [orientation[3], orientation[0], orientation[1], orientation[2]]
        if return_transformation:
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_quat(orientation).as_matrix()
            pose[:3, 3] = np.array(position)
            return pose
        else:
            return [np.array(position), np.array(orientation)]

    def state_to_list(self, state):
        return [state[i] for i in range(self.num_dim)]

    def run_ik(
            self, world_H_EEtarget, q_initial=None, debug=False,
            eps_position=0.005, eps_orientation = np.deg2rad(1.0),
            **kwargs):
        # Gradient-based IK solver using pinochio
        assert self.pinocchio_robot_model is not None, "Please specify urdf_path when constructing PbOMPLRobot"

        if debug:
            print(f'Running Inverse Kinematics for {self.link_name_ee}...')

        world_H_EEtarget = pinocchio.SE3(world_H_EEtarget)

        if q_initial is None:
            q = self.get_random_joint_position()
        else:
            q = q_initial + np.random.normal(0, 0.5, size=q_initial.shape)
            q = np.clip(q, self.joint_bounds_low_np, self.joint_bounds_high_np)

        IT_MAX = 1000
        DT = 1e-2
        damp = 1e-12

        np_eye_6 = np.eye(6)

        i = 0
        while True:
            pinocchio.forwardKinematics(self.pinocchio_robot_model, self.pinocchio_robot_model_data, q)
            pinocchio.updateFramePlacements(self.pinocchio_robot_model, self.pinocchio_robot_model_data)
            pinocchio.framesForwardKinematics(self.pinocchio_robot_model, self.pinocchio_robot_model_data, q)

            # Transform the EE target pose from the robot base frame to the (local/parent) joint frame
            world_H_joint = self.pinocchio_robot_model_data.oMi[self.pinocchio_ee_parent_joint_id]
            world_H_EE = self.pinocchio_robot_model_data.oMf[self.pinocchio_ee_frameid]

            joint_H_EEtarget = world_H_EEtarget.act(world_H_EE.actInv(world_H_joint))

            dMi = joint_H_EEtarget.actInv(world_H_joint)

            # The error is computed in the joint frame
            err = pinocchio.log(dMi).vector
            err_position = err[:3]
            err_orientation = err[3:]
            if np.all(np.abs(err_position) < eps_position) and np.linalg.norm(err_orientation) < eps_orientation:
                success = True
                break
            if i >= IT_MAX:
                success = False
                break
            # The jacobian is computed in the joint frame, because the error is in the joint frame
            pinocchio.computeJointJacobians(self.pinocchio_robot_model, self.pinocchio_robot_model_data, q)
            J = pinocchio.computeJointJacobian(
                self.pinocchio_robot_model, self.pinocchio_robot_model_data, q,
                self.pinocchio_ee_parent_joint_id
            )
            v = - J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np_eye_6, err))
            q = pinocchio.integrate(self.pinocchio_robot_model, q, v * DT)
            if not i % 10 and debug:
                print('%d: error = %s' % (i, err.T))
            i += 1

        if debug:
            print('------------------')
            print('IK RESULTS\n')
            if success:
                print("IK convergence achieved!")
            else:
                print("\nWarning: the iterative algorithm has not reached convergence to the desired precision")

            print(f'\nIK joint position: {q.flatten()}')
            print(f'\nIK error: {err.T}')
            print(f'...Done Inverse Kinematics for {self.link_name_ee}\n')

        if success:
            return q.flatten().tolist()
        else:
            return None

class PbStateSpace(ob.RealVectorStateSpace):
    def __init__(self, num_dim) -> None:
        super().__init__(num_dim)
        self.num_dim = num_dim
        self.state_sampler = None

    def allocStateSampler(self):
        '''
        This will be called by the internal OMPL planner
        '''
        # WARN: This will cause problems if the underlying planner is multi-threaded!!!
        if self.state_sampler:
            return self.state_sampler

        # when ompl planner calls this, we will return our sampler
        return self.allocDefaultStateSampler()

    def set_state_sampler(self, state_sampler):
        '''
        Optional, Set custom state sampler.
        '''
        self.state_sampler = state_sampler


class PbOMPL():
    def __init__(self, pybullet_client, robot, obstacles = [],
                 min_distance_robot_env=0.0,
                 min_distance_robot_env_waypoint_checking=0.01,
                 ) -> None:
        '''
        Args
            robot: A PbOMPLRobot instance.
            obstacles: list of obstacle ids. Optional.
        '''
        self.pybullet_client = pybullet_client
        self.robot = robot
        self.robot_id = robot.id
        self.obstacles = obstacles
        print(self.obstacles)

        self.space = PbStateSpace(robot.num_dim)

        bounds = ob.RealVectorBounds(robot.num_dim)
        joint_bounds = self.robot.get_joint_bounds()
        for i, bound in enumerate(joint_bounds):
            bounds.setLow(i, bound[0])
            bounds.setHigh(i, bound[1])
        self.space.setBounds(bounds)

        self.ss = og.SimpleSetup(self.space)
        self.min_distance_robot_env = min_distance_robot_env
        self.min_distance_robot_env_waypoint_checking = min(min_distance_robot_env, min_distance_robot_env_waypoint_checking)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_state_valid))
        self.si = self.ss.getSpaceInformation()
        # self.si.setStateValidityCheckingResolution(0.005)
        # self.collision_fn = pb_utils.get_collision_fn(self.robot_id, self.robot.joint_idx, self.obstacles, [], True, set(),
        #                                                 custom_limits={}, max_distance=0, allow_collision_links=[])

        self.set_obstacles(obstacles)
        self.set_planner("RRT")  # RRT by default

    def set_obstacles(self, obstacles):
        self.obstacles = obstacles

        # update collision detection
        self.setup_collision_detection(self.robot, self.obstacles)

    def add_obstacles(self, obstacle_id):
        self.obstacles.append(obstacle_id)

    def remove_obstacles(self, obstacle_id):
        self.obstacles.remove(obstacle_id)

    def is_state_valid(self, state, max_distance=None, check_bounds=False):
        # Satisfy bounds
        # Should be unecessary if joint bounds is properly set
        # Check for joint bounds due to the bspline interpolation
        if check_bounds:
            if np.any(np.logical_or(state < self.robot.joint_bounds_np[:, 0], state > self.robot.joint_bounds_np[:, 1])):
                return False

        # set the robot internal state
        self.robot.set_state(self.state_to_list(state))

        # check self-collision
        for link1, link2 in self.check_link_pairs:
            # max_distance=0: don't admit any self-collision
            if utils.pairwise_link_collision(self.pybullet_client, self.robot_id, link1, self.robot_id, link2, max_distance=0.):
                # print(get_body_name(body), get_link_name(body, link1), get_link_name(body, link2))
                return False

        # check collision against environment
        if max_distance is None:
            max_distance = self.min_distance_robot_env
        for body1, body2 in self.check_body_pairs:
            if utils.pairwise_collision(
                    self.pybullet_client,
                    body1, body2,
                    max_distance=max_distance):
                # print('body collision', body1, body2)
                # print(get_body_name(body1), get_body_name(body2))
                return False
        return True

    def setup_collision_detection(self, robot, obstacles, self_collisions=True, allow_collision_links=[]):
        self.check_link_pairs = utils.get_self_link_pairs(self.pybullet_client, robot.id, robot.joint_idx) if self_collisions else []
        moving_links = frozenset(
            [item for item in utils.get_moving_links(self.pybullet_client, robot.id, robot.joint_idx) if not item in allow_collision_links])
        moving_bodies = [(robot.id, moving_links)]
        self.check_body_pairs = list(product(moving_bodies, obstacles))

    def set_planner(self, planner_name):
        '''
        Note: Add your planner here!!
        '''
        if planner_name == "PRM":
            self.planner = og.PRM(self.ss.getSpaceInformation())
        elif planner_name == "PRMstar":
            self.planner = og.PRM(self.ss.getSpaceInformation())
        elif planner_name == "RRT":
            self.planner = og.RRT(self.ss.getSpaceInformation())
        elif planner_name == "RRTConnect":
            self.planner = og.RRTConnect(self.ss.getSpaceInformation())
        elif planner_name == "RRTstar":
            self.planner = og.RRTstar(self.ss.getSpaceInformation())
        elif planner_name == "EST":
            self.planner = og.EST(self.ss.getSpaceInformation())
        elif planner_name == "FMT":
            self.planner = og.FMT(self.ss.getSpaceInformation())
        elif planner_name == "BITstar":
            self.planner = og.BITstar(self.ss.getSpaceInformation())
        elif planner_name == "ABITstar":
            self.planner = og.ABITstar(self.ss.getSpaceInformation())
        elif planner_name == "AITstar":
            self.planner = og.AITstar(self.ss.getSpaceInformation())
        else:
            print("{} not recognized, please add it first".format(planner_name))
            raise NotImplementedError

        self.ss.setPlanner(self.planner)

    def plan_start_goal(self, start, goal, allowed_time=DEFAULT_PLANNING_TIME,
                        simplify_path=True,
                        interpolate_num=INTERPOLATE_NUM,
                        fit_bspline=False,
                        bspline_num_control_points=32,
                        bspline_degree=3,
                        bspline_zero_vel_at_start_and_goal=True,
                        bspline_zero_acc_at_start_and_goal=True,
                        debug=False,
                        **kwargs):
        # plan a path from the given robot start state to goal
        # clear any existing planning data
        st = time.time()
        self.ss.clear()

        if debug:
            print("\n#################################################################################")
            print("START planning")
            print(self.planner.params())

        orig_robot_state = self.robot.get_cur_state()

        # set the start and goal states;
        s = ob.State(self.space)
        g = ob.State(self.space)
        for i in range(len(start)):
            s[i] = start[i]
            g[i] = goal[i]

        self.ss.setStartAndGoalStates(s, g)

        # attempt to solve the problem within allowed planning time
        planner_status = self.ss.solve(allowed_time)
        solved = self.ss.haveExactSolutionPath()  # check if the problem is solved exactly, and not just approximately

        success = False
        sol_path_np = None
        bspline_params = None
        sol_path_after_bspline_fit_np = None
        all_states_valid_after_bspline_fit = False

        if solved:
            success = True

            if debug:
                print(f"\nFound solution: (smoothing and) interpolating into {interpolate_num} segments")
            sol_path_geometric = self.ss.getSolutionPath()

            # Get the path
            if debug:
                sol_path_states = sol_path_geometric.getStates()
                sol_path_list = [self.state_to_list(state) for state in sol_path_states]
                sol_path_list_raw_before_simplify = copy.deepcopy(sol_path_list)
                print(f'length before simplify+smooth: {len(sol_path_list)}')

            # Simplify path - shortcut, smooth, and interpolate
            if simplify_path:
                ps = og.PathSimplifier(self.si)
                # Simplify the path
                # https://ompl.kavrakilab.org/classompl_1_1geometric_1_1PathSimplifier.html
                # https://ompl.kavrakilab.org/PathSimplifier_8cpp_source.html - lines 677
                res_simplify = ps.simplify(sol_path_geometric, maxTime=1e-1)
                if debug:
                    print(f"simplify path return: {res_simplify}")

            if debug:
                sol_path_states = sol_path_geometric.getStates()
                sol_path_list = [self.state_to_list(state) for state in sol_path_states]
                sol_path_list_after_simplify = copy.deepcopy(sol_path_list)
                print(f'length after simplify+smooth: {len(sol_path_list)}')

            # interpolate
            sol_path_geometric.interpolate(interpolate_num)
            if debug:
                sol_path_states = sol_path_geometric.getStates()
                sol_path_list = [self.state_to_list(state) for state in sol_path_states]
                sol_path_list_after_interpolate = copy.deepcopy(sol_path_list)
                print(f'length after interpolation: {len(sol_path_list)}')

            sol_path_list = [self.state_to_list(state) for state in sol_path_geometric.getStates()]
            sol_path_np = np.array(sol_path_list)

            ############################################################
            # fit a bspline and check if all states are valid
            try:
                if fit_bspline:
                    bspline_params = fit_bspline_to_path(
                        sol_path_np,
                        bspline_degree,
                        bspline_num_control_points,
                        bspline_zero_vel_at_start_and_goal,
                        bspline_zero_acc_at_start_and_goal,
                    )
                    tt, cc, k = bspline_params

                    # Evaluate the b-spline for validity checking
                    bspl = interpolate.BSpline(tt, cc.T, k)  # note the transpose
                    u_interpolation = np.linspace(0, 1, interpolate_num)
                    bspline_path_interpolated_pos = bspl(u_interpolation)

                    # Check if after bspline fitting all states are not in collision
                    if debug:
                        print(f'Checking if all states are valid after B-spline fitting...')
                    for state_in_path in bspline_path_interpolated_pos:
                        # Check if all states in the path are not in collision.
                        # They can be in collision due to the bspline fitting and interpolation
                        if not self.is_state_valid(
                                state_in_path,
                                max_distance=self.min_distance_robot_env_waypoint_checking,
                                check_bounds=True):
                            all_states_valid_after_bspline_fit = False
                            break
                    if all_states_valid_after_bspline_fit:
                        if debug:
                            print(f'...all states are valid.')
                    else:
                        if debug:
                            print(f'...NOT all states are valid.')

            except Exception as e:
                print(f'Exception: {e}')
                sol_path_list = []
                bspline_params = None
        else:
            print("No EXACT solution found\n")

        # Plot the bspline for debugging
        if solved and debug:
            fig, axs = plt.subplots(1, 3, figsize=(16, 6), squeeze=False)

            u_interpolation = np.linspace(0, 1, interpolate_num)

            colors = cm.rainbow(np.linspace(0, 1, sol_path_np.shape[1]))
            sol_path_list_raw_before_simplify_np = np.array(sol_path_list_raw_before_simplify)
            sol_path_list_after_simplify_np = np.array(sol_path_list_after_simplify)
            sol_path_list_after_interpolate_np = np.array(sol_path_list_after_interpolate)
            for i, (sol_path_raw_before_simplify_dim_i_pos, sol_path_after_simplify_dim_i_pos, sol_path_after_interpolate_dim_i_pos) \
                    in enumerate(zip(
                        sol_path_list_raw_before_simplify_np.T,
                        sol_path_list_after_simplify_np.T,
                        sol_path_list_after_interpolate_np.T)):

                # axs[0, 0].plot(np.linspace(0, 1, len(sol_path_raw_before_simplify_dim_i_pos)), sol_path_raw_before_simplify_dim_i_pos,
                #                linestyle='dashed', color=colors[i], marker='+', label=f'path-raw-before-simplify-{i}-pos', zorder=10)
                # axs[0, 0].plot(np.linspace(0, 1, len(sol_path_after_simplify_dim_i_pos)), sol_path_after_simplify_dim_i_pos,
                #                linestyle='dotted', color=colors[i], marker='x', label=f'path-after-simplify-{i}-pos', zorder=10)
                axs[0, 0].plot(u_interpolation, sol_path_after_interpolate_dim_i_pos, linestyle='solid', color=colors[i],
                               label=f'path-after-interpolate-{i}-pos', zorder=10)

            # Get the trajectory position, velocity and acceleration from the b-spline in phase space
            if bspline_params is not None:
                bspline_path_interpolated_pos = bspl(u_interpolation)
                bspline_path_interpolated_vel = bspl(u_interpolation, nu=1)
                bspline_path_interpolated_acc = bspl(u_interpolation, nu=2)

                for i, (bspline_path_dim_i_pos, bspline_path_dim_i_vel, bspline_path_dim_i_acc) \
                        in enumerate(zip(
                            bspline_path_interpolated_pos.T, bspline_path_interpolated_vel.T, bspline_path_interpolated_acc.T)):
                    axs[0, 0].plot(u_interpolation, bspline_path_dim_i_pos, lw=3, alpha=0.7, color=colors[i], label=f'BSpline-{i}-pos', zorder=10)
                    axs[0, 1].plot(u_interpolation, bspline_path_dim_i_vel, lw=3, alpha=0.7, color=colors[i], label=f'BSpline-{i}-vel', zorder=10)
                    axs[0, 2].plot(u_interpolation, bspline_path_dim_i_acc, lw=3, alpha=0.7, color=colors[i], label=f'BSpline-{i}-acc', zorder=10)

            axs[0, 0].set_ylim(np.min(self.robot.joint_bounds_low_np), np.max(self.robot.joint_bounds_high_np))
            axs[0, 0].legend(loc='best')
            axs[0, 0].set_title('Position')
            axs[0, 1].legend(loc='best')
            axs[0, 1].set_title('Velocity')
            axs[0, 2].legend(loc='best')
            axs[0, 2].set_title('Acceleration')
            plt.show()

            if sol_path_np.shape[1] == 2:
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))

                q_limits_low = self.robot.joint_bounds_low_np
                q_limits_high = self.robot.joint_bounds_high_np

                # create meshgrid
                N = 100
                q1 = np.linspace(q_limits_low[0], q_limits_high[0], N)
                q2 = np.linspace(q_limits_low[1], q_limits_high[1], N)
                Q1, Q2 = np.meshgrid(q1, q2)

                # check if the state is valid for each point in the meshgrid
                valid = np.zeros(Q1.shape)
                for i in range(Q1.shape[0]):
                    for j in range(Q1.shape[1]):
                        valid[i, j] = self.is_state_valid(np.array([Q1[i, j], Q2[i, j]]))

                # plot the meshgrid and validity

                # plot the meshgrid
                cMap = matplotlib.colors.ListedColormap(['grey', 'white'])
                ax.contourf(Q1, Q2, valid, cmap=cMap)

                ax.set_xlabel('$q_1$ [rad]')
                ax.set_ylabel('$q_2$ [rad]')

                ax.plot(sol_path_list_raw_before_simplify_np.T[0], sol_path_list_raw_before_simplify_np.T[1], marker='+', label='Raw Path')
                ax.plot(sol_path_list_after_simplify_np.T[0], sol_path_list_after_simplify_np.T[1], marker='+', markersize=25, label='Simplified Path')
                ax.plot(sol_path_list_after_interpolate_np.T[0], sol_path_list_after_interpolate_np.T[1], label='Interpolated Path')
                if bspline_params is not None:
                    ax.scatter(cc.T[:, 0], cc.T[:, 1], marker='o', color='orange', label='Control Points')
                    ax.plot(bspline_path_interpolated_pos[:, 0], bspline_path_interpolated_pos[:, 1], c='b', lw=3, alpha=0.7)
                ax.legend(loc='best')

                plt.show()

        results_dict = dict(
            success=success,
            sol_path=sol_path_np,
            bspline_params=bspline_params,
            sol_path_after_bspline_fit=sol_path_after_bspline_fit_np,
            all_states_valid_after_bspline_fit=all_states_valid_after_bspline_fit
        )

        return results_dict

    def plan(self, goal, allowed_time=DEFAULT_PLANNING_TIME, **kwargs):
        '''
        plan a path to gaol from current robot state
        '''
        start = self.robot.get_cur_state()
        return self.plan_start_goal(start, goal, allowed_time=allowed_time, **kwargs)

    def execute(self, path, dynamics=False, sleep_time=0.05, substeps=10):
        '''
        Execute a planned plan. Will visualize in pybullet.
        Args:
            path: list[state], a list of state
            dynamics: allow dynamic simulation. If dynamics is false, this API will use robot.set_state(),
                      meaning that the simulator will simply reset robot's state WITHOUT any dynamics simulation. Since the
                      path is collision free, this is somewhat acceptable.
            sleep_time: float, sleep time between each step
            substeps: int, number of physics simulations per step
        '''
        orig_robot_state = path[0]
        self.robot.set_state(orig_robot_state)
        for q in path:
            if dynamics:
                for i in range(self.robot.num_dim):
                    self.pybullet_client.setJointMotorControl2(
                        self.robot.id, i, self.pybullet_client.POSITION_CONTROL, q[i], force=5 * 240.)
                for _ in range(substeps):
                    self.pybullet_client.stepSimulation()
            else:
                self.robot.set_state(q)

            time.sleep(sleep_time)

    def get_state_not_in_collision(self, ee_pose_target=None, max_tries=500, raise_error=True, debug=False, **kwargs):
        """
        Get a state not in collision, with IK if ee_pose_target is not None
        """
        if debug:
            print(f'\n---> Getting state not in collision...')
        state_valid = None
        state = None
        for j in range(max_tries):
            if ee_pose_target is not None:
                state = self.robot.run_ik(ee_pose_target, **kwargs)
                if state is None:
                    continue
            else:
                state = self.robot.get_random_joint_position()

            if self.is_state_valid(state, check_bounds=True):
                state_valid = state
                break

        if state_valid is not None:
            if debug:
                print(f'...Found a valid state after {j}/{max_tries} tries')
            return state_valid

        if not raise_error:
            return None

        if ee_pose_target is not None and state is None:
            raise RuntimeError(f"Failed to solve IK for:\n{ee_pose_target}")

        raise RuntimeError("Failed to find a state not in collision")

    def get_ee_pose(self, state, **kwargs):
        """
        Get end-effector pose
        """
        return self.robot.get_ee_pose(state, **kwargs)

    # -------------
    # Configurations
    # ------------

    def set_state_sampler(self, state_sampler):
        self.space.set_state_sampler(state_sampler)

    # -------------
    # Util
    # ------------

    def state_to_list(self, state):
        return [state[i] for i in range(self.robot.num_dim)]


###############################################################################################################
# HELPER FUNCTIONS
def add_box(pybullet_client, box_pos, half_box_size,
            orientation=(0, 0, 0, 1),  # orientation quaternion xyzw
            color=(220./255., 220./255., 220./255., 1.0)):
    col_id = pybullet_client.createCollisionShape(pybullet_client.GEOM_BOX, halfExtents=half_box_size)
    visual_id = pybullet_client.createVisualShape(
        pybullet_client.GEOM_BOX, halfExtents=half_box_size, rgbaColor=color
    )
    body_id = pybullet_client.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=visual_id,
        basePosition=box_pos, baseOrientation=orientation
    )
    return body_id


def add_sphere(pybullet_client, sphere_pos, sphere_radius,
               orientation=(0, 0, 0, 1),  # orientation quaternion xyzw
               color=(220./255., 220./255., 220./255., 1.0)
               ):
    col_id = pybullet_client.createCollisionShape(pybullet_client.GEOM_SPHERE, radius=sphere_radius)
    visual_id = pybullet_client.createVisualShape(
        pybullet_client.GEOM_SPHERE, radius=sphere_radius, rgbaColor=color
    )
    body_id = pybullet_client.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=visual_id,
        basePosition=sphere_pos, baseOrientation=orientation
    )
    return body_id


def finite_difference_vector(x, dt=1., method='central'):
    # finite differences with zero padding at the borders
    diff_vector = np.zeros_like(x)
    if method == 'forward':
        diff_vector[..., :-1, :] = np.diff(x, axis=-2) / dt
    elif method == 'backward':
        diff_vector[..., 1:, :] = (x[..., 1:, :] - x[..., :-1, :]) / dt
    elif method == 'central':
        diff_vector[..., 1:-1, :] = (x[..., 2:, :] - x[..., :-2, :]) / (2*dt)
    else:
        raise NotImplementedError
    return diff_vector


def create_bspline_knots(num_control_points, degree):
    """
    Create the knots for a B-spline.
    # https://arxiv.org/pdf/2301.04330.pdf
    # Equally spaced knots.
    # These knots ensure that the first and last control points are the start and goal states
    """
    d = degree
    c = num_control_points
    m = d + c
    knots = np.pad(np.linspace(0., 1., m + 1 - 2 * d), d, 'edge')
    # OLD implementation (for reference)
    # knots = np.zeros(d + 1)
    # knots = np.append(knots, np.linspace(1 / (c - d), (c - d - 1) / (c - d), c - d - 1))
    # knots = np.append(knots, np.ones(d + 1))
    return knots


def fit_bspline_to_path(
        path,
        bspline_degree,
        bspline_num_control_points,
        bspline_zero_vel_at_start_and_goal,
        bspline_zero_acc_at_start_and_goal,
        debug=False,
        **kwargs
):
    """
    Fit a B-spline to the path.
    """
    print(f'\nFitting B-spline') if debug else None

    knots = create_bspline_knots(bspline_num_control_points, bspline_degree)

    # Fit a b-spline to the path
    tck, u = interpolate.splprep(path.T, k=bspline_degree, t=knots, task=-1, quiet=True)
    tt, cc, k = tck
    cc = np.array(cc)

    print(f'u shape: {u.shape}') if debug else None
    print(f'knots shape: {tt.shape}') if debug else None
    print(f'coefficients shape: {cc.shape}') if debug else None

    if bspline_zero_vel_at_start_and_goal:
        # The initial and final velocity should be zero
        # Set the second and second-to-last control points to be the same
        # as the first and last control points.
        # assign the control points to the start and goal states - https://arxiv.org/pdf/2301.04330.pdf
        cc[:, 0] = path[0].copy()
        cc[:, -1] = path[-1].copy()
        cc[:, 1] = cc[:, 0].copy()
        cc[:, -2] = cc[:, -1].copy()
    if bspline_zero_acc_at_start_and_goal:
        # The initial and final acceleration should be zero
        # Set the third and third-to-last control points to be the same
        # as the first and last control points.
        cc[:, 2] = cc[:, 0].copy()
        cc[:, -3] = cc[:, -1].copy()

    # Update the bspline parameters
    tck[1] = cc.copy()
    bspline_params = tck
    return bspline_params

