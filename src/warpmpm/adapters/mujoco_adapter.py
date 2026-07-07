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
                 hide_gripper: bool = False, base_pos=None, max_geom: int = 10000):
        _default_mujoco_gl()
        import mujoco
        from robot_descriptions import panda_mj_description

        self.mj = mujoco
        # optionally compile with a wrist force/torque sensor (a load cell at the hand)
        self._ft = None
        if ft_sensor or base_pos is not None:
            spec = mujoco.MjSpec.from_file(panda_mj_description.MJCF_PATH)
            if base_pos is not None:
                # reposition the arm base (link0) so MuJoCo world == the scene's world
                root = next((b for b in spec.bodies if b.name == "link0"), None)
                if root is None:
                    root = spec.bodies[1]  # first body under the world
                root.pos = [float(base_pos[0]), float(base_pos[1]), float(base_pos[2])]
            hand = next((b for b in spec.bodies if b.name == "hand"), None)
            if ft_sensor and hand is not None:
                hand.add_site(name="wrist_ft", pos=[0.0, 0.0, 0.0])
                for nm, ty in (("wrist_force", mujoco.mjtSensor.mjSENS_FORCE),
                               ("wrist_torque", mujoco.mjtSensor.mjSENS_TORQUE)):
                    sn = spec.add_sensor(); sn.name = nm; sn.type = ty
                    sn.objtype = mujoco.mjtObj.mjOBJ_SITE; sn.objname = "wrist_ft"
            self._customize_spec(spec, mujoco)
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
        self.renderer = mujoco.Renderer(self.model, height=height, width=width,
                                        max_geom=max_geom)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.cam.distance = 1.8
        self.cam.azimuth = 135
        self.cam.elevation = -20
        self._prev_ee = None

    def _customize_spec(self, spec, mujoco) -> None:
        """Hook for subclasses to extend the MjSpec (extra assets/bodies) before
        compile. Only invoked on the MjSpec path (ft_sensor or base_pos set)."""

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
        rgba4) drawn as solid boxes (e.g. a plate mounted on the gripper); cylinders is a
        list of (center3, mat33 or None, radius, half_height, rgba4) drawn as (optionally
        transparent) cylinders -- the glasses of the pouring scene. Subsample pts to fit
        max_geom."""
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
        for box in (boxes or []):
            if sc.ngeom >= sc.maxgeom:
                break
            center, half3, col = box[0], box[1], box[2]
            rot = np.asarray(box[3], np.float64).flatten() if len(box) > 3 else eye
            g = sc.geoms[sc.ngeom]
            self.mj.mjv_initGeom(g, self.mj.mjtGeom.mjGEOM_BOX,
                                 np.asarray(half3, np.float64), np.asarray(center, np.float64),
                                 rot, np.asarray(col, np.float32))
            sc.ngeom += 1
        for cyl_center, cyl_mat, cyl_r, cyl_half_h, cyl_col in (cylinders or []):
            if sc.ngeom >= sc.maxgeom:
                break
            g = sc.geoms[sc.ngeom]
            rot = eye if cyl_mat is None else np.asarray(cyl_mat, np.float64).flatten()
            self.mj.mjv_initGeom(g, self.mj.mjtGeom.mjGEOM_CYLINDER,
                                 np.array([cyl_r, cyl_r, cyl_half_h], np.float64),
                                 np.asarray(cyl_center, np.float64), rot,
                                 np.asarray(cyl_col, np.float32))
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


class PandaPour(FrankaArm):
    """Scripted Franka POUR kinematics + the cup grasp transform, ported 1:1 from the
    Dogma95 Genesis pouring study (robotic_arm_pour_genesis.py) so the robot action and
    scene geometry are cross-comparable between the SPH and MPM simulators. FK is
    bit-identical between this Menagerie panda and Genesis's panda.xml (verified at the
    upright / 80% / full pour configs). The held glass's pose is the fixed handle-grasp
    transform applied to the hand FK; drive the MPM cup collider with cup_pose_at(t).

    Default action: smoothstep joint interpolation upright -> POUR_POSE_FRACTION of the
    full-pour config over TILT_SECONDS, then back over RETURN_SECONDS. The default
    profile's peak pose-fraction rate (1.5x average) is far below the Panda's tightest
    joint-velocity ceiling for this motion (~2.2 fraction/s), so the scripted action is
    physically executable."""

    Q_UPRIGHT = np.array([-1.5916905403137207, -1.2717534303665161, -0.06664533913135529,
                          -2.951836109161377, 1.5030548572540283, 1.537463665008545,
                          2.2464425563812256, 0.026, 0.026])
    Q_FULL_POUR = np.array([-1.6516658067703247, -0.798383355140686, 0.6634261012077332,
                            -2.0772507190704346, 0.5259794592857361, 1.4164435863494873,
                            2.8647570610046387, 0.026, 0.026])
    POUR_POSE_FRACTION = 0.80
    TILT_SECONDS = 3.0
    RETURN_SECONDS = 1.6
    BASE_POS = (-0.15, 0.0, 0.0)
    TCP_LOCAL = np.array([0.0, 0.0, 0.092])       # tool centre point in the hand frame
    GRASP_LOCAL = np.array([-0.149, 0.0, 0.055])  # grasped handle point in the cup frame
    # cup axes expressed in hand axes (fixed grasp): cup z along hand x, cup x along -hand z
    CUP_TO_HAND = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])

    def __init__(self, height: int = 480, width: int = 640, max_geom: int = 30000,
                 glass_mesh=None, glass_rgba=(0.76, 0.92, 1.0, 0.30)):
        # optional render glasses: the watertight open-top mesh (write_glass_obj) is
        # added as an asset with two mocap bodies ("glass_src", "glass_rcv") so MuJoCo
        # draws the REAL glass geometry (thick base, filleted cavity) with glass-like
        # transparency -- the Dogma95 look. Pose them per frame with set_glass_pose.
        self._glass_mesh = None if glass_mesh is None else str(glass_mesh)
        self._glass_rgba = tuple(float(c) for c in glass_rgba)
        super().__init__(height=height, width=width, base_pos=self.BASE_POS,
                         max_geom=max_geom)
        self.q_pour = self.Q_UPRIGHT + self.POUR_POSE_FRACTION * (
            self.Q_FULL_POUR - self.Q_UPRIGHT
        )
        self._glass_mocap = {}
        if self._glass_mesh is not None:
            for nm in ("glass_src", "glass_rcv"):
                bid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, nm)
                self._glass_mocap[nm] = int(self.model.body_mocapid[bid])

    def _customize_spec(self, spec, mujoco) -> None:
        if self._glass_mesh is None:
            return
        mesh = spec.add_mesh()
        mesh.name = "pour_glass"
        mesh.file = self._glass_mesh
        for nm in ("glass_src", "glass_rcv"):
            b = spec.worldbody.add_body()
            b.name = nm
            b.mocap = True
            g = b.add_geom()
            g.type = mujoco.mjtGeom.mjGEOM_MESH
            g.meshname = "pour_glass"
            g.rgba = self._glass_rgba
            g.contype = 0
            g.conaffinity = 0

    def set_glass_pose(self, name: str, pos, quat) -> None:
        """Pose a render glass ("glass_src" / "glass_rcv") in the world frame (wxyz
        quat). Takes effect at the next mj_forward (set_time does one)."""
        mid = self._glass_mocap[name]
        self.data.mocap_pos[mid] = np.asarray(pos, dtype=np.float64)
        self.data.mocap_quat[mid] = np.asarray(quat, dtype=np.float64)

    @property
    def duration(self) -> float:
        return self.TILT_SECONDS + self.RETURN_SECONDS

    @staticmethod
    def _smoothstep(x: float) -> float:
        x = float(np.clip(x, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def motion_fraction(self, t: float) -> float:
        """Pose fraction upright->pour at time t (smoothstep tilt, then return)."""
        if t < 0.0:
            return 0.0
        if t < self.TILT_SECONDS:
            return self._smoothstep(t / self.TILT_SECONDS)
        t -= self.TILT_SECONDS
        if t < self.RETURN_SECONDS:
            return 1.0 - self._smoothstep(t / self.RETURN_SECONDS)
        return 0.0

    def q_at(self, t: float) -> np.ndarray:
        return self.Q_UPRIGHT + self.motion_fraction(t) * (self.q_pour - self.Q_UPRIGHT)

    def set_time(self, t: float) -> None:
        """Pose the arm at trajectory time t (kinematic; fingers held at the grasp)."""
        self.data.qpos[:9] = self.q_at(t)
        self.mj.mj_forward(self.model, self.data)

    def cup_pose_at(self, t: float):
        """World pose (pos, wxyz quat) of the held glass at time t: hand FK -> TCP ->
        fixed handle-grasp transform (Dogma95 _cup_pose_from_grasp_tcp)."""
        from warpmpm.colliders.glass import quat_from_mat, quat_to_mat

        self.set_time(t)
        hand_pos = self.data.xpos[self.ee].copy()
        hand_quat = self.data.xquat[self.ee].copy()
        r_hand = quat_to_mat(hand_quat)
        tcp = hand_pos + r_hand @ self.TCP_LOCAL
        r_cup = r_hand @ self.CUP_TO_HAND.T
        cup_pos = tcp - r_cup @ self.GRASP_LOCAL
        return cup_pos, quat_from_mat(r_cup)

    def tilt_degrees(self, quat) -> float:
        """Cup tilt from vertical (deg): angle of the cup z-axis to world z."""
        from warpmpm.colliders.glass import quat_to_mat

        return float(np.degrees(np.arccos(np.clip(quat_to_mat(quat)[2, 2], -1.0, 1.0))))
