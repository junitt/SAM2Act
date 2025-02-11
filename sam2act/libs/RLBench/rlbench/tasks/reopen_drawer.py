import math
from typing import List, Tuple
import numpy as np
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
from pyrep.objects.shape import Shape
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.task import Task
from rlbench.backend.conditions import Condition, DetectedCondition, JointCondition, NothingGrasped, ConditionSet

DRAWER_NAMES = ['bottom', 'middle', 'top']

class JointConditionEx(Condition):
    def __init__(self, joint: Joint, position: float):
        """in radians if revoloute, or meters if prismatic"""
        self._joint = joint
        self._original_pos = joint.get_joint_position()
        self._pos = position
        self._done = False

    def condition_met(self):
        met = math.fabs(
            self._joint.get_joint_position() - self._original_pos) > self._pos
        if met:
            self._done = True
        return self._done, False


class ReopenDrawer(Task):

    def init_task(self):
        self.button = Shape(f"push_buttons_target1")
        self.drawer_parts = [Shape(f"drawer_{name}") for name in DRAWER_NAMES]
        self.drawer_joints = [Joint(f"drawer_joint_{name}") for name in DRAWER_NAMES]
        self.button_joint = Joint("target_button_joint1")

        self.detector = ProximitySensor("success")

        self.spawn_boundary = SpawnBoundary([Shape("boundary")])
        self.target_drawer_joint = None

        self.goal_conditions = []

    def init_episode(self, index: int) -> List[str]:
        for i in range(len(DRAWER_NAMES)):
            if i == index:
                self.drawer_joints[i].set_joint_position(0.21, disable_dynamics=True)
            else:
                self.drawer_joints[i].set_joint_position(0.0, disable_dynamics=True)

        target_anchor = Dummy(f"waypoint_anchor_{DRAWER_NAMES[index]}")
        waypoint0 = Dummy("waypoint0")
        waypoint1 = Dummy("waypoint1")
        waypoint6 = Dummy("waypoint6")
        waypoint7 = Dummy("waypoint7")
        waypoint8 = Dummy("waypoint8")

        _, _, z = target_anchor.get_position()
        x, y, _ = waypoint0.get_position()
        waypoint0.set_position([x, y, z])

        x, y, _ = waypoint1.get_position()
        waypoint1.set_position([x, y, z])

        x, y, _ = waypoint6.get_position()
        waypoint6.set_position([x, y, z])

        x, y, _ = waypoint7.get_position()
        waypoint7.set_position([x, y, z])

        x, y, _ = waypoint8.get_position()
        waypoint8.set_position([x, y, z])

        self.spawn_boundary.clear()
        self.spawn_boundary.sample(self.button, min_distance=0.05)

        self.goal_conditions = [
            JointConditionEx(self.drawer_joints[index], 0.19),
            JointConditionEx(self.button_joint, 0.003),
            DetectedCondition(self.drawer_parts[index], self.detector),
        ]
        condition_set = ConditionSet(self.goal_conditions, True) # type: ignore
        self.register_success_conditions([condition_set])

        self.target_drawer_joint = self.drawer_joints[index]

        return ['Close the opened drawer, push the button, and then open the previous drawer again']

#    def step(self) -> None:
#        #if self.target_drawer_joint:
#        #    print(f"drawer_joint: {self.target_drawer_joint.get_joint_position()}")
#
#        if len(self.goal_conditions) < 1:
#            return
#
#        met1, _ = self.goal_conditions[0].condition_met()
#        print(f"first condition met: {met1}")
#
#        met2, _ = self.goal_conditions[1].condition_met()
#        print(f"second condition met: {met2}")
#
#        met3, _ = self.goal_conditions[2].condition_met()
#        print(f"second condition met: {met3}")

    def variation_count(self) -> int:
        return len(DRAWER_NAMES)

    def is_static_workspace(self) -> bool:
        return True
