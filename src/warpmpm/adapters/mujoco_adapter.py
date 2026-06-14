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
        # the default offscreen framebuffer is 640x480; enlarge it so any requested size renders
        self.model.vis.global_.offwidth = max(int(self.model.vis.global_.offwidth), width)
        self.model.vis.global_.offheight = max(int(self.model.vis.global_.offheight), height)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.cam.distance = 1.8
        self.cam.azimuth = 135
        self.cam.elevation = -20
        self._prev_ee = None

    def set_descent(self, frac: float, dt: float, track_camera: bool = True) -> dict:
        """Set the arm to a scripted descent fraction in [0,1]; return EE world pose+vel.
        track_camera follows the EE with the camera (good for the 2-panel view); pass False
        when the caller drives a fixed camera (e.g. the composite single-view)."""
        q = (1.0 - frac) * self.Q_UP + frac * self.Q_DOWN
        self.data.qpos[:7] = q
        self.mj.mj_forward(self.model, self.data)
        ee_pos = self.data.xpos[self.ee].copy()
        ee_vel = (np.zeros(3) if (self._prev_ee is None or dt <= 0)
                  else (ee_pos - self._prev_ee) / dt)
        self._prev_ee = ee_pos
        if track_camera:
            self.cam.lookat[:] = [ee_pos[0], ee_pos[1], ee_pos[2] - 0.2]
        return {"pos": ee_pos, "vel": ee_vel}

    def render_rgb(self) -> np.ndarray:
        self.renderer.update_scene(self.data, self.cam)
        return self.renderer.render()

    def render_with_particles(self, pts_world, rgba, radius=0.004, table=None, boxes=None):
        """Composite render: the Franka + the MPM material as spheres in ONE camera view.
        pts_world (M,3) world-frame particle positions; rgba (M,4) per-particle colour;
        table=(cx,cy,z,half) draws a flat support box; boxes is a list of (center3, half3,
        rgba4) drawn as solid boxes (e.g. a plate mounted on the gripper). Subsample pts to
        fit max_geom."""
        self.renderer.update_scene(self.data, self.cam)
        sc = self.renderer.scene
        eye = np.eye(3).flatten()
        if table is not None:
            cx, cy, z, half = table
            g = sc.geoms[sc.ngeom]
            self.mj.mjv_initGeom(g, self.mj.mjtGeom.mjGEOM_BOX,
                                 np.array([half, half, 0.01]), np.array([cx, cy, z - 0.01]),
                                 eye, np.array([0.55, 0.57, 0.6, 1.0], np.float32))
            sc.ngeom += 1
        for center, half3, col in (boxes or []):
            if sc.ngeom >= sc.maxgeom:
                break
            g = sc.geoms[sc.ngeom]
            self.mj.mjv_initGeom(g, self.mj.mjtGeom.mjGEOM_BOX,
                                 np.asarray(half3, np.float64), np.asarray(center, np.float64),
                                 eye, np.asarray(col, np.float32))
            sc.ngeom += 1
        room = sc.maxgeom - sc.ngeom
        n = len(pts_world)
        stride = max(1, int(np.ceil(n / max(room, 1))))
        for i in range(0, n, stride):
            if sc.ngeom >= sc.maxgeom:
                break
            g = sc.geoms[sc.ngeom]
            self.mj.mjv_initGeom(g, self.mj.mjtGeom.mjGEOM_SPHERE,
                                 np.array([radius, 0.0, 0.0]), pts_world[i].astype(np.float64),
                                 eye, rgba[i].astype(np.float32))
            sc.ngeom += 1
        return self.renderer.render()
