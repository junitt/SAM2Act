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

NUM_TARGETS = 4

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



class PutBlockBack(Task):
    def init_task(self) -> None:
        self._block = Shape("block")
        self._detectors = [ProximitySensor(f"success{i+1}") for i in range(NUM_TARGETS)]
        self._targets = [Shape(f"target{i+1}") for i in range(NUM_TARGETS)]

        self._button = Shape("push_buttons_target1")
        self._button_joint = Joint("target_button_joint1")
        self._center_detector = ProximitySensor("success0")

        self.spawn_boundary = SpawnBoundary([Shape("boundary")])

        self.register_graspable_objects([self._block])

        self.goal_conditions = []

    def init_episode(self, index: int) -> List[str]:
        target_patch = self._targets[index]
        target_detector = self._detectors[index]

        self.spawn_boundary.clear()
        self.spawn_boundary.sample(self._button, min_distance=0.05)

        # Set the position of the block to be the initial target XY ------------
        x, y, _ = target_patch.get_position()
        _, _, z = self._block.get_position()
        self._block.set_position([x, y, z])

        waypoint1 = Dummy("waypoint1")
        _, _, z = waypoint1.get_position()
        waypoint1.set_position([x, y, z])

        waypoint10 = Dummy("waypoint10")
        _, _, z = waypoint10.get_position()
        waypoint10.set_position([x, y, z])

        waypoint11 = Dummy("waypoint11")
        _, _, z = waypoint11.get_position()
        waypoint11.set_position([x, y, z])
        # ----------------------------------------------------------------------

        self.goal_conditions = [
            # Checks that the block was lifted from the table
            DetectedTriggerCondition(self._block, target_detector, negated=True),
            # Checks that the cube was placed at the center of the table
            DetectedTriggerCondition(self._block, self._center_detector),
            # Checks that the button was pressed
            JointTriggerCondition(self._button_joint, 0.003),
            # Checks that the block was returned back to its original position
            DetectedCondition(self._block, target_detector),
            # Checks that nothing is still grasped
            NothingGrasped(self.robot.gripper),
        ]
        condition_set = ConditionSet(self.goal_conditions, False) # type: ignore
        self.register_success_conditions([condition_set])

        return ['move the block on the color patch to the center, then press the button, and finally move the block back to where it was placed']

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

    def variation_count(self) -> int:
        return 4

    def is_static_workspace(self) -> bool:
        return True
