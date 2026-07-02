"""MuJoCo Franka adapter: load a Panda, drive a scripted end-effector descent, expose its
world pose/velocity to the coupling layer, and render the arm (for the reference-style
view). Isaac Lab plugs into the same contract later (set_robot_kinematics / wrench).

v1 keeps the arm KINEMATIC (scripted qpos), since the dough is the dynamics; the arm's
end-effector drives the MPM gripper box and reads back the reaction wrench.
"""
from __future__ import annotations

import os

import numpy as np


def _default_mujoco_gl() -> None:
    """Use EGL for offscreen MuJoCo rendering on headless Linux."""
    if os.name == "posix" and not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


class FrankaArm:
    """Franka Panda in MuJoCo, native on Apple Silicon. Scripted vertical descent."""

    # raised / lowered arm configs (7 arm joints); fingers held closed-ish
    Q_UP = np.array([0.0, -0.3, 0.0, -1.9, 0.0, 1.6, 0.79])
    Q_DOWN = np.array([0.0, 0.35, 0.0, -2.4, 0.0, 2.75, 0.79])

    def __init__(self, height: int = 480, width: int = 640, ft_sensor: bool = False,
                 hide_gripper: bool = False):
        _default_mujoco_gl()
        import mujoco
        from robot_descriptions import panda_mj_description

        self.mj = mujoco
        # optionally compile with a wrist force/torque sensor (a load cell at the hand)
        self._ft = None
        if ft_sensor:
            spec = mujoco.MjSpec.from_file(panda_mj_description.MJCF_PATH)
            hand = next((b for b in spec.bodies if b.name == "hand"), None)
            if hand is not None:
                hand.add_site(name="wrist_ft", pos=[0.0, 0.0, 0.0])
                for nm, ty in (("wrist_force", mujoco.mjtSensor.mjSENS_FORCE),
                               ("wrist_torque", mujoco.mjtSensor.mjSENS_TORQUE)):
                    sn = spec.add_sensor(); sn.name = nm; sn.type = ty
                    sn.objtype = mujoco.mjtObj.mjOBJ_SITE; sn.objname = "wrist_ft"
            self.model = spec.compile()
        else:
            self.model = mujoco.MjModel.from_xml_path(panda_mj_description.MJCF_PATH)
        self.data = mujoco.MjData(self.model)
        # end-effector body = the hand (fall back to last body)
        try:
            self.ee = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        except Exception:
            self.ee = self.model.nbody - 1
        if self.ee < 0:
            self.ee = self.model.nbody - 1
        if ft_sensor and self.model.nsensor:
            fid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "wrist_force")
            self._ft = int(self.model.sensor_adr[fid])
        if hide_gripper:
            # the squeeze tool is a flat PLATE (the box collider), not the Panda's jaws;
            # make the gripper fingers + hand flange invisible so the render matches the
            # physics (arm + mounted plate). Collision is unaffected (rgba is render-only).
            for gid in range(self.model.ngeom):
                bid = int(self.model.geom_bodyid[gid])
                bname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
                if "finger" in bname or bname == "hand":
                    self.model.geom_rgba[gid, 3] = 0.0
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

    def close(self) -> None:
        """Release MuJoCo renderer resources, suppressing backend shutdown noise."""
        renderer = getattr(self, "renderer", None)
        if renderer is not None:
            try:
                renderer.close()
            except Exception:
                pass
            self.renderer = None

    def __del__(self):
        self.close()

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

    def wrist_load_cell(self, frac: float, f_dough_world, settle: int = 300) -> np.ndarray:
        """Read the wrist force-torque sensor (the load cell) for the reaction the dough
        exerts on the arm. The arm is held at descent fraction `frac` by its position
        actuators; we settle it with no load (baseline = the gripper's own weight) and with
        the dough reaction `f_dough_world` applied to the hand, and return the DIFFERENCE --
        the dough's contribution at the wrist. By Newton's third law this equals the force we
        fed in (= the MPM grid-impulse reaction), now read at the wrist like a real robot.
        Requires ft_sensor=True. Uses forward dynamics, so it mutates the arm state."""
        if self._ft is None:
            raise RuntimeError("FrankaArm was built without ft_sensor=True")
        q = (1.0 - frac) * self.Q_UP + frac * self.Q_DOWN

        def _settle(F):
            self.data.qpos[:7] = q; self.data.qvel[:] = 0.0
            if self.model.nu >= 7:
                self.data.ctrl[:7] = q
            for _ in range(settle):
                self.data.xfrc_applied[self.ee, :3] = F
                self.mj.mj_step(self.model, self.data)
            self.data.xfrc_applied[self.ee, :3] = 0.0
            return self.data.sensordata[self._ft:self._ft + 3].copy()

        base = _settle(np.zeros(3))
        loaded = _settle(np.asarray(f_dough_world, dtype=float))
        return loaded - base

    def render_rgb(self) -> np.ndarray:
        self.renderer.update_scene(self.data, self.cam)
        return self.renderer.render()

    def render_with_particles(self, pts_world, rgba, radius=0.004, table=None, boxes=None,
                              cylinders=None):
        """Composite render: the Franka + the MPM material as spheres in ONE camera view.
        pts_world (M,3) world-frame particle positions; rgba (M,4) per-particle colour;
        table=(cx,cy,z,half) draws a flat support box; boxes is a list of (center3, half3,
        rgba4) drawn as solid boxes (e.g. a plate mounted on the gripper); cylinders is a list
        of (center3, radius, half_height, mat9, rgba4) drawn as (optionally translucent)
        cylinders, e.g. the pouring and receiving glasses. Subsample pts to fit max_geom."""
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
        for center, rad, half_h, mat9, col in (cylinders or []):
            if sc.ngeom >= sc.maxgeom:
                break
            g = sc.geoms[sc.ngeom]
            self.mj.mjv_initGeom(g, self.mj.mjtGeom.mjGEOM_CYLINDER,
                                 np.array([rad, rad, half_h], np.float64),
                                 np.asarray(center, np.float64),
                                 np.asarray(mat9, np.float64).flatten(),
                                 np.asarray(col, np.float32))
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
