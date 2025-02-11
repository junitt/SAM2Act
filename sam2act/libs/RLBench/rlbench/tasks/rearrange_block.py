import itertools
import math
from typing import Dict, List

from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
from pyrep.objects.object import Object
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.task import Task
from rlbench.backend.conditions import Condition, ConditionSet, DetectedCondition, NothingGrasped

NUM_TARGETS = 2

class JointTriggerCondition(Condition):
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

class DetectedTriggerCondition(Condition):
    def __init__(self, obj: Object, detector: ProximitySensor,
                 negated: bool = False):
        self._obj = obj
        self._detector = detector
        self._negated = negated
        self._done = False

    def condition_met(self):
        met = self._detector.is_detected(self._obj)
        if self._negated:
            met = not met
        if met:
            self._done = True
        return self._done, False



class RearrangeBlock(Task):
    def init_task(self) -> None:
        self._detectors = [ProximitySensor(f"success{i+1}") for i in range(NUM_TARGETS)]
        self._targets = [Shape(f"target{i+1}") for i in range(NUM_TARGETS)]

        self._patch_block = Shape("block1")
        self._center_block = Shape("block2")
        self._center_detector = ProximitySensor("success0")

        self._button = Shape("push_buttons_target1")
        self._button_joint = Joint("target_button_joint1")

        self.spawn_boundary = SpawnBoundary([Shape("boundary")])

        self.register_graspable_objects([self._patch_block, self._center_block])

        self.goal_conditions = []

    def init_episode(self, index: int) -> List[str]:
        target_patch = self._targets[index]
        dual_index = NUM_TARGETS - index - 1
        dual_patch = self._targets[dual_index]

        self.spawn_boundary.clear()
        self.spawn_boundary.sample(self._button, min_distance=0.05)

        # Set the position of the patch block to be the initial target XY ------
        x, y, _ = target_patch.get_position()
        _, _, z = self._patch_block.get_position()
        self._patch_block.set_position([x, y, z])

        waypoint6 = Dummy("waypoint6")
        _, _, z = waypoint6.get_position()
        waypoint6.set_position([x, y, z])

        waypoint7 = Dummy("waypoint7")
        _, _, z = waypoint7.get_position()
        waypoint7.set_position([x, y, z])

        waypoint8 = Dummy("waypoint8")
        _, _, z = waypoint8.get_position()
        waypoint8.set_position([x, y, z])

        # ----------------------------------------------------------------------

        # Place the waypoints associated with the dual patch correctly ---------
        waypoint2 = Dummy("waypoint2")
        x, y, _ = dual_patch.get_position()
        _, _, z = waypoint2.get_position()
        waypoint2.set_position([x, y, z])

        waypoint3 = Dummy("waypoint3")
        _, _, z = waypoint3.get_position()
        waypoint3.set_position([x, y, z])

        # ----------------------------------------------------------------------

        self.goal_conditions = [
            # Checks that the center block was lifted up
            DetectedTriggerCondition(self._center_block, self._center_detector, negated=True),
            # Checks that the center block was placed in the empty patch
            DetectedCondition(self._center_block, self._detectors[dual_index]),
            # Checks that the button was pressed
            JointTriggerCondition(self._button_joint, 0.003),
            # Checks that the patch block was lifted up
            DetectedTriggerCondition(self._patch_block, self._detectors[index], negated=True),
            # Checks that the patch block was placed in the center
            DetectedCondition(self._patch_block, self._center_detector),
            # Checks that nothing is still grasped
            NothingGrasped(self.robot.gripper),
        ]
        condition_set = ConditionSet(self.goal_conditions, False) # type: ignore
        self.register_success_conditions([condition_set])

        return ['Move the block not on the patch to the empty patch, then press the button, then move the block that has not been moved off the patch']

    def step(self) -> None:
        if len(self.goal_conditions) < 1:
            return

        met1, _ = self.goal_conditions[0].condition_met()
        #print(f"first condition met: {met1}")

        met2, _ = self.goal_conditions[1].condition_met()
        #print(f"second condition met: {met2}")

        met3, _ = self.goal_conditions[2].condition_met()
        #print(f"third condition met: {met3}")

        met4, _ = self.goal_conditions[3].condition_met()
        #print(f"fourth condition met: {met4}")

        met5, _ = self.goal_conditions[4].condition_met()
        #print(f"fifth condition met: {met5}")

        met6, _ = self.goal_conditions[5].condition_met()
        #print(f"sixth condition met: {met6}")

    def variation_count(self) -> int:
        return 2

    def is_static_workspace(self) -> bool:
        return True
