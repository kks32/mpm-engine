"""MuJoCo Franka adapter: load a Panda, drive a scripted end-effector descent, expose its
world pose/velocity to the coupling layer, and render the arm (for the reference-style
view). Isaac Lab plugs into the same contract later (set_robot_kinematics / wrench).

v1 keeps the arm KINEMATIC (scripted qpos), since the dough is the dynamics; the arm's
end-effector drives the MPM gripper box and reads back the reaction wrench.
"""
from __future__ import annotations

import numpy as np


class FrankaArm:
    """Franka Panda in MuJoCo, native on Apple Silicon. Scripted vertical descent."""

    # raised / lowered arm configs (7 arm joints); fingers held closed-ish
    Q_UP = np.array([0.0, -0.3, 0.0, -1.9, 0.0, 1.6, 0.79])
    Q_DOWN = np.array([0.0, 0.35, 0.0, -2.4, 0.0, 2.75, 0.79])

    def __init__(self, height: int = 480, width: int = 640):
        import mujoco
        from robot_descriptions import panda_mj_description

        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(panda_mj_description.MJCF_PATH)
        self.data = mujoco.MjData(self.model)
        # end-effector body = the hand (fall back to last body)
        try:
            self.ee = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        except Exception:
            self.ee = self.model.nbody - 1
        if self.ee < 0:
            self.ee = self.model.nbody - 1
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.cam.distance = 1.8
        self.cam.azimuth = 135
        self.cam.elevation = -20
        self._prev_ee = None

    def set_descent(self, frac: float, dt: float) -> dict:
        """Set the arm to a scripted descent fraction in [0,1]; return EE world pose+vel."""
        q = (1.0 - frac) * self.Q_UP + frac * self.Q_DOWN
        self.data.qpos[:7] = q
        self.mj.mj_forward(self.model, self.data)
        ee_pos = self.data.xpos[self.ee].copy()
        ee_vel = (np.zeros(3) if (self._prev_ee is None or dt <= 0)
                  else (ee_pos - self._prev_ee) / dt)
        self._prev_ee = ee_pos
        self.cam.lookat[:] = [ee_pos[0], ee_pos[1], ee_pos[2] - 0.2]
        return {"pos": ee_pos, "vel": ee_vel}

    def render_rgb(self) -> np.ndarray:
        self.renderer.update_scene(self.data, self.cam)
        return self.renderer.render()
