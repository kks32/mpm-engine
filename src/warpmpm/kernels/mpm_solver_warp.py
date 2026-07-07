import os
import warnings

import numpy as np

from warpmpm.kernels.warp_utils import *  # noqa: F401,F403
from warpmpm.kernels.mpm_utils import *  # noqa: F401,F403


MATERIAL_NAME_TO_ID = {
    "jelly": 0, "metal": 1, "sand": 2, "foam": 3,
    "snow": 4, "plasticine": 5, "fluid": 6, "stationary": 7, "rigid": 8,
    "mu_i_sand": 9, "newtonian": 10, "mu_i_phi": 11, "tabulated_viscous": 12,
    "tabulated_mu_i": 13,
}
MAX_MATERIALS = 16


@wp.func
def sdf_trilerp(vals: wp.array(dtype=float, ndim=3), res: int, fidx: wp.vec3):
    """Trilinear interpolation of a scalar voxel grid at continuous index fidx."""
    fi = wp.min(wp.max(fidx[0], 0.0), float(res) - 1.0001)
    fj = wp.min(wp.max(fidx[1], 0.0), float(res) - 1.0001)
    fk = wp.min(wp.max(fidx[2], 0.0), float(res) - 1.0001)
    i = int(wp.floor(fi)); j = int(wp.floor(fj)); k = int(wp.floor(fk))
    tx = fi - float(i); ty = fj - float(j); tz = fk - float(k)
    c00 = vals[i, j, k] * (1.0 - tx) + vals[i + 1, j, k] * tx
    c01 = vals[i, j, k + 1] * (1.0 - tx) + vals[i + 1, j, k + 1] * tx
    c10 = vals[i, j + 1, k] * (1.0 - tx) + vals[i + 1, j + 1, k] * tx
    c11 = vals[i, j + 1, k + 1] * (1.0 - tx) + vals[i + 1, j + 1, k + 1] * tx
    c0 = c00 * (1.0 - ty) + c10 * ty
    c1 = c01 * (1.0 - ty) + c11 * ty
    return c0 * (1.0 - tz) + c1 * tz


@wp.func
def sdf_trilerp_vec(vals: wp.array(dtype=wp.vec3, ndim=3), res: int, fidx: wp.vec3):
    """Trilinear interpolation of a vec3 voxel grid (the SDF gradient) at continuous index."""
    fi = wp.min(wp.max(fidx[0], 0.0), float(res) - 1.0001)
    fj = wp.min(wp.max(fidx[1], 0.0), float(res) - 1.0001)
    fk = wp.min(wp.max(fidx[2], 0.0), float(res) - 1.0001)
    i = int(wp.floor(fi)); j = int(wp.floor(fj)); k = int(wp.floor(fk))
    tx = fi - float(i); ty = fj - float(j); tz = fk - float(k)
    c00 = vals[i, j, k] * (1.0 - tx) + vals[i + 1, j, k] * tx
    c01 = vals[i, j, k + 1] * (1.0 - tx) + vals[i + 1, j, k + 1] * tx
    c10 = vals[i, j + 1, k] * (1.0 - tx) + vals[i + 1, j + 1, k] * tx
    c11 = vals[i, j + 1, k + 1] * (1.0 - tx) + vals[i + 1, j + 1, k + 1] * tx
    c0 = c00 * (1.0 - ty) + c10 * ty
    c1 = c01 * (1.0 - ty) + c11 * ty
    return c0 * (1.0 - tz) + c1 * tz


def _quat_mul(q1, q2):
    """Hamilton product of two (x, y, z, w) quaternions (numpy, host side)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def _omega_step_quat(q, omega, dt):
    """Advance orientation q (body->world, xyzw) by a world-frame angular velocity over dt
    using the exponential map (exact for constant omega); world omega left-multiplies."""
    w = np.asarray(omega, dtype=float)
    speed = float(np.linalg.norm(w))
    if speed * dt < 1e-12:
        return q
    half = 0.5 * speed * dt
    axis = w / speed
    dq = np.array([axis[0] * np.sin(half), axis[1] * np.sin(half), axis[2] * np.sin(half),
                   np.cos(half)])
    qn = _quat_mul(dq, q)
    return qn / np.linalg.norm(qn)


class MPM_Simulator_WARP:
    def __init__(self, n_particles, n_grid=100, grid_lim=1.0, device="cuda:0"):
        self.initialize(n_particles, n_grid, grid_lim, device=device)
        self.time_profile = {}

    def initialize(self, n_particles, n_grid=100, grid_lim=1.0, device="cuda:0"):
        self.n_particles = n_particles

        self.mpm_model = MPMModelStruct()
        # domain will be [0,grid_lim]*[0,grid_lim]*[0,grid_lim] !!!
        # domain will be [0,grid_lim]*[0,grid_lim]*[0,grid_lim] !!!
        # domain will be [0,grid_lim]*[0,grid_lim]*[0,grid_lim] !!!
        self.mpm_model.grid_lim = grid_lim
        self.mpm_model.n_grid = n_grid
        self.mpm_model.grid_dim_x = self.mpm_model.n_grid
        self.mpm_model.grid_dim_y = self.mpm_model.n_grid
        self.mpm_model.grid_dim_z = self.mpm_model.n_grid
        (
            self.mpm_model.dx,
            self.mpm_model.inv_dx,
        ) = self.mpm_model.grid_lim / self.mpm_model.n_grid, float(
            self.mpm_model.n_grid / self.mpm_model.grid_lim
        )

        # per-type material parameters (indexed by material type, not particle)
        self.mpm_model.E = wp.zeros(shape=MAX_MATERIALS, dtype=float, device=device)
        self.mpm_model.nu = wp.zeros(shape=MAX_MATERIALS, dtype=float, device=device)
        self.mpm_model.bulk = wp.zeros(shape=MAX_MATERIALS, dtype=float, device=device)
        self.mpm_model.hardening = wp.zeros(shape=MAX_MATERIALS, dtype=float, device=device)
        self.mpm_model.xi = wp.zeros(shape=MAX_MATERIALS, dtype=float, device=device)
        self.mpm_model.plastic_viscosity = wp.zeros(shape=MAX_MATERIALS, dtype=float, device=device)
        self.mpm_model.softening = wp.zeros(shape=MAX_MATERIALS, dtype=float, device=device)
        self.mpm_model.alpha = wp.zeros(shape=MAX_MATERIALS, dtype=float, device=device)

        # local mu(I) rheology (material 9, TrackEUCLID), per material type
        self.mpm_model.muI_mu_s = wp.zeros(shape=MAX_MATERIALS, dtype=float, device=device)
        self.mpm_model.muI_delta_mu = wp.zeros(shape=MAX_MATERIALS, dtype=float, device=device)
        self.mpm_model.muI_I0 = wp.zeros(shape=MAX_MATERIALS, dtype=float, device=device)
        self.mpm_model.muI_d = wp.zeros(shape=MAX_MATERIALS, dtype=float, device=device)
        self.mpm_model.muI_rho_s = wp.zeros(shape=MAX_MATERIALS, dtype=float, device=device)
        # compressible mu(I)-Phi(I) dilatancy (material 11), per material type
        self.mpm_model.muI_phi_init = wp.zeros(shape=MAX_MATERIALS, dtype=float, device=device)
        self.mpm_model.muI_phi_chi = wp.zeros(shape=MAX_MATERIALS, dtype=float, device=device)
        # default I0 = 1 and grain diameter/density = 1 to keep the bisection
        # well posed even for material types that never use mu(I); phi_init = 1
        # so Phi/Phi_init is well defined for non-mu_i_phi materials
        wp.launch(kernel=set_value_to_float_array, dim=MAX_MATERIALS,
                  inputs=[self.mpm_model.muI_I0, 1.0], device=device)
        wp.launch(kernel=set_value_to_float_array, dim=MAX_MATERIALS,
                  inputs=[self.mpm_model.muI_d, 1.0], device=device)
        wp.launch(kernel=set_value_to_float_array, dim=MAX_MATERIALS,
                  inputs=[self.mpm_model.muI_rho_s, 1.0], device=device)
        wp.launch(kernel=set_value_to_float_array, dim=MAX_MATERIALS,
                  inputs=[self.mpm_model.muI_phi_init, 1.0], device=device)

        # tabulated apparent viscosity (material 12): one global eta_app(gd) table on a
        # uniform log10(gd) grid. Always allocated (>=2 entries) so the struct field is
        # valid for every material; set via set_parameters_dict("eta_table", ...).
        self.mpm_model.eta_table = wp.zeros(shape=2, dtype=float, device=device)
        self.mpm_model.eta_table_smin = -1.0
        self.mpm_model.eta_table_smax = 2.0
        self.mpm_model.eta_table_n = 2

        # per-particle material parameters (can evolve during simulation)
        self.mpm_model.mu = wp.zeros(shape=n_particles, dtype=float, device=device)
        self.mpm_model.lam = wp.zeros(shape=n_particles, dtype=float, device=device)
        self.mpm_model.yield_stress = wp.zeros(shape=n_particles, dtype=float, device=device)

        self.mpm_model.update_cov_with_F = False

        # material is used to switch between different elastoplastic models. 0 is jelly
        self.mpm_model.material = 0

        # default friction_angle and alpha for all types
        self.mpm_model.friction_angle = 25.0
        sin_phi = wp.sin(self.mpm_model.friction_angle / 180.0 * 3.14159265)
        default_alpha = wp.sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi)
        wp.launch(kernel=set_value_to_float_array, dim=MAX_MATERIALS,
                  inputs=[self.mpm_model.alpha, default_alpha], device=device)
        wp.launch(kernel=set_value_to_float_array, dim=MAX_MATERIALS,
                  inputs=[self.mpm_model.softening, 0.1], device=device)

        self.mpm_model.gravitational_accelaration = wp.vec3(0.0, 0.0, 0.0)

        self.mpm_model.rpic_damping = 0.0  # 0.0 if no damping (apic). -1 if pic

        self.mpm_model.grid_v_damping_scale = 1.1  # globally applied

        self.mpm_state = MPMStateStruct()

        self.mpm_state.particle_x = wp.empty(
            shape=n_particles, dtype=wp.vec3, device=device
        )  # current position

        self.mpm_state.particle_v = wp.zeros(
            shape=n_particles, dtype=wp.vec3, device=device
        )  # particle velocity

        self.mpm_state.particle_F = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )  # particle F elastic

        self.mpm_state.particle_R = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )  # particle R rotation

        self.mpm_state.particle_init_cov = wp.zeros(
            shape=n_particles * 6, dtype=float, device=device
        )  # initial covariance matrix

        self.mpm_state.particle_cov = wp.zeros(
            shape=n_particles * 6, dtype=float, device=device
        )  # current covariance matrix

        self.mpm_state.particle_F_trial = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )  # apply return mapping will yield

        self.mpm_state.particle_stress = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )

        self.mpm_state.particle_L = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )  # velocity gradient from g2p, L_ij = dv_i/dx_j (TrackEUCLID dump)

        self.mpm_state.particle_vol = wp.zeros(
            shape=n_particles, dtype=float, device=device
        )  # particle volume
        self.mpm_state.particle_mass = wp.zeros(
            shape=n_particles, dtype=float, device=device
        )  # particle mass
        self.mpm_state.particle_density = wp.zeros(
            shape=n_particles, dtype=float, device=device
        )
        self.mpm_state.particle_C = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )
        self.mpm_state.particle_Jp = wp.zeros(
            shape=n_particles, dtype=float, device=device
        )

        self.mpm_state.particle_selection = wp.zeros(
            shape=n_particles, dtype=int, device=device
        )

        self.mpm_state.particle_material = wp.zeros(
            shape=n_particles, dtype=int, device=device
        )  # default 0 = jelly

        self.mpm_state.particle_rigid_id = wp.zeros(
            shape=n_particles, dtype=int, device=device
        )  # rigid body id; only meaningful when particle_material == 8
        self.mpm_state.particle_x_ref = wp.zeros(
            shape=n_particles, dtype=wp.vec3, device=device
        )  # body-frame reference position for rigid particles

        self.mpm_state.grid_m = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=float,
            device=device,
        )
        self.mpm_state.grid_v_in = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=wp.vec3,
            device=device,
        )
        self.mpm_state.grid_v_out = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=wp.vec3,
            device=device,
        )

        self.time = 0.0

        self.grid_postprocess = []
        self.collider_params = []
        self.modify_bc = []
        # per-collider host callables returning the current (lo_index, dim) grid box the
        # collider can touch, or None to skip the launch this substep; entry None means the
        # collider has no box (full-grid fallback). restrict_bc=False disables restriction
        # (used by the equivalence test).
        self.collider_aabbs = []
        self.restrict_bc = True
        # live particle grid box for the zero/normalize/damping sweeps, set per control
        # tick by the wrapper (core.Solver.step); None = full grid. Zeroing runs over the
        # union with the previous box so nodes leaving the box are cleared exactly once
        # (stale mass outside the box would corrupt collider force readouts).
        self.grid_launch_box = None
        self._prev_grid_box = None
        self.restrict_grid = True
        # per-phase ScopedTimers force a device sync each substep, which is pure
        # instrumentation (stream ordering guarantees kernel correctness; host reads
        # synchronize implicitly). profile=True restores the synced timers and fills
        # time_profile; default off so the GPU pipeline never stalls for a stopwatch.
        self.profile = False
        # CUDA graph capture of the fixed-shape substep segments (speedup_plan Step 3):
        # zero->stress->p2g->normalize(+damping) and g2p replay as two graphs; the BC
        # launches and host-side pose integration stay live between them. CUDA only;
        # the first substep runs live so modules JIT-load before capture; any capture
        # error falls back to live launches for the rest of the run.
        self.use_cuda_graph = True
        self._graph_A = None
        # active-block sparse compute (enable via rebuild_active_blocks, driven per tick
        # by core.Solver when Solver.sparse=True). Takes precedence over the CUDA-graph
        # path (its launch dims vary per tick) and over the particle-box restriction.
        self.sparse_ready = False
        self._blk = None
        self._graph_B = None
        self._graph_sig = None
        self._graph_warmup = 0

        self.tailored_struct_for_bc = MPMtailoredStruct()
        self.pre_p2g_operations = []
        self.impulse_params = []

        self.particle_velocity_modifiers = []
        self.particle_velocity_modifier_params = []

        # surfaces that apply restitution impulses to rigid bodies
        self.rigid_surface_colliders = []

        # rigid body state (populated by initialize_rigid_bodies / finalize_rigid_bodies)
        self.n_rigid_bodies = 0
        self.rigid_x_cm = None
        self.rigid_v_cm = None
        self.rigid_omega = None
        self.rigid_orientation = None
        self.rigid_mass = None
        self.rigid_inv_inertia_body = None
        self._rigid_linear_mom = None   # per-step accumulation buffer
        self._rigid_angular_mom = None  # per-step accumulation buffer

    # the h5 file should store particle initial position and volume.
    def load_from_sampling(
        self, sampling_h5, n_grid=100, grid_lim=1.0, device="cuda:0"
    ):
        if not os.path.exists(sampling_h5):
            print("h5 file cannot be found at ", os.getcwd() + sampling_h5)
            exit()

        import h5py  # optional: only needed by this unused-in-engine h5 loader
        h5file = h5py.File(sampling_h5, "r")
        x, particle_volume = h5file["x"], h5file["particle_volume"]

        x = x[()].transpose()  # np vector of x # shape now is (n_particles, dim)

        self.dim, self.n_particles = x.shape[1], x.shape[0]

        self.initialize(self.n_particles, n_grid, grid_lim, device=device)

        print(
            "Sampling particles are loaded from h5 file. Simulator is re-initialized for the correct n_particles"
        )
        particle_volume = np.squeeze(particle_volume, 0)

        self.mpm_state.particle_x = wp.from_numpy(
            x, dtype=wp.vec3, device=device
        )  # initialize warp array from np

        # initial velocity is default to zero
        wp.launch(
            kernel=set_vec3_to_zero,
            dim=self.n_particles,
            inputs=[self.mpm_state.particle_v],
            device=device,
        )
        # initial velocity is default to zero

        # initial deformation gradient is set to identity
        wp.launch(
            kernel=set_mat33_to_identity,
            dim=self.n_particles,
            inputs=[self.mpm_state.particle_F_trial],
            device=device,
        )
        # initial deformation gradient is set to identity

        self.mpm_state.particle_vol = wp.from_numpy(
            particle_volume, dtype=float, device=device
        )

        print("Particles initialized from sampling file.")
        print("Total particles: ", self.n_particles)

    # shape of tensor_x is (n, 3); shape of tensor_volume is (n,)
    def load_initial_data_from_torch(
        self,
        tensor_x,
        tensor_volume,
        tensor_cov = None,
        n_grid=100,
        grid_lim=1.0,
        device="cuda:0",
    ):
        self.dim, self.n_particles = tensor_x.shape[1], tensor_x.shape[0]
        assert tensor_x.shape[0] == tensor_volume.shape[0]
        # assert tensor_x.shape[0] == tensor_cov.reshape(-1, 6).shape[0]
        self.initialize(self.n_particles, n_grid, grid_lim, device=device)

        self.import_particle_x_from_torch(tensor_x, device=device)
        self.mpm_state.particle_vol = wp.from_numpy(
            tensor_volume.detach().clone().cpu().numpy(), dtype=float, device=device
        )
        if tensor_cov is not None:
            self.mpm_state.particle_init_cov = wp.from_numpy(
                tensor_cov.reshape(-1).detach().clone().cpu().numpy(),
                dtype=float,
                device=device,
            )

            if self.mpm_model.update_cov_with_F:
                self.mpm_state.particle_cov = self.mpm_state.particle_init_cov

        # initial velocity is default to zero
        wp.launch(
            kernel=set_vec3_to_zero,
            dim=self.n_particles,
            inputs=[self.mpm_state.particle_v],
            device=device,
        )
        # initial velocity is default to zero

        # initial deformation gradient is set to identity
        wp.launch(
            kernel=set_mat33_to_identity,
            dim=self.n_particles,
            inputs=[self.mpm_state.particle_F_trial],
            device=device,
        )
        # initial trial deformation gradient is set to identity

        print("Particles initialized from torch data.")
        print("Total particles: ", self.n_particles)

    # must give density. mass will be updated as density * volume
    def set_parameters(self, device="cuda:0", **kwargs):
        self.set_parameters_dict(device, kwargs)

    def _material_name_to_id(self, name):
        if isinstance(name, int):
            return name
        if name not in MATERIAL_NAME_TO_ID:
            raise TypeError(f"Undefined material type: {name}")
        return MATERIAL_NAME_TO_ID[name]

    def set_parameters_dict(self, kwargs={}, device="cuda:0"):
        if "material" in kwargs:
            mat_id = self._material_name_to_id(kwargs["material"])
            self.mpm_model.material = mat_id
            # broadcast material type to all particles
            wp.launch(
                kernel=set_value_to_int_array,
                dim=self.n_particles,
                inputs=[self.mpm_state.particle_material, mat_id],
                device=device,
            )

        if "grid_lim" in kwargs:
            self.mpm_model.grid_lim = kwargs["grid_lim"]
        if "n_grid" in kwargs:
            self.mpm_model.n_grid = kwargs["n_grid"]
        self.mpm_model.grid_dim_x = self.mpm_model.n_grid
        self.mpm_model.grid_dim_y = self.mpm_model.n_grid
        self.mpm_model.grid_dim_z = self.mpm_model.n_grid
        (
            self.mpm_model.dx,
            self.mpm_model.inv_dx,
        ) = self.mpm_model.grid_lim / self.mpm_model.n_grid, float(
            self.mpm_model.n_grid / self.mpm_model.grid_lim
        )
        self.mpm_state.grid_m = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=float,
            device=device,
        )
        self.mpm_state.grid_v_in = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=wp.vec3,
            device=device,
        )
        self.mpm_state.grid_v_out = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=wp.vec3,
            device=device,
        )

        mat_id = self.mpm_model.material  # current material type for per-type indexing

        # per-type parameters (E, nu, bulk written to model arrays at mat_id index)
        if "E" in kwargs:
            E_torch = wp.to_torch(self.mpm_model.E)
            E_torch[mat_id] = kwargs["E"]
        if "nu" in kwargs:
            nu_torch = wp.to_torch(self.mpm_model.nu)
            nu_torch[mat_id] = kwargs["nu"]
        if "bulk_modulus" in kwargs:
            bulk_torch = wp.to_torch(self.mpm_model.bulk)
            bulk_torch[mat_id] = kwargs["bulk_modulus"]

        # per-particle parameters
        if "yield_stress" in kwargs:
            val = kwargs["yield_stress"]
            wp.launch(
                kernel=set_value_to_float_array,
                dim=self.n_particles,
                inputs=[self.mpm_model.yield_stress, val],
                device=device,
            )

        # per-type plasticity parameters
        if "hardening" in kwargs:
            h_torch = wp.to_torch(self.mpm_model.hardening)
            h_torch[mat_id] = float(kwargs["hardening"])
        if "xi" in kwargs:
            xi_torch = wp.to_torch(self.mpm_model.xi)
            xi_torch[mat_id] = kwargs["xi"]
        if "friction_angle" in kwargs:
            self.mpm_model.friction_angle = kwargs["friction_angle"]
            sin_phi = wp.sin(self.mpm_model.friction_angle / 180.0 * 3.14159265)
            alpha_val = wp.sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi)
            alpha_torch = wp.to_torch(self.mpm_model.alpha)
            alpha_torch[mat_id] = alpha_val
        if "plastic_viscosity" in kwargs:
            pv_torch = wp.to_torch(self.mpm_model.plastic_viscosity)
            pv_torch[mat_id] = kwargs["plastic_viscosity"]
        if "softening" in kwargs:
            s_torch = wp.to_torch(self.mpm_model.softening)
            s_torch[mat_id] = kwargs["softening"]

        # tabulated apparent viscosity (material 12): eta_app samples on a uniform
        # log10(gd) grid in [eta_table_smin, eta_table_smax]
        if "eta_table" in kwargs:
            tab = np.asarray(kwargs["eta_table"], dtype=np.float32).reshape(-1)
            self.mpm_model.eta_table = wp.array(tab, dtype=float, device=device)
            self.mpm_model.eta_table_n = int(tab.shape[0])
        if "eta_table_smin" in kwargs:
            self.mpm_model.eta_table_smin = float(kwargs["eta_table_smin"])
        if "eta_table_smax" in kwargs:
            self.mpm_model.eta_table_smax = float(kwargs["eta_table_smax"])

        # local mu(I) rheology parameters (material 9 and 11), per material type
        for key, arr in (
            ("mu_s", self.mpm_model.muI_mu_s),
            ("delta_mu", self.mpm_model.muI_delta_mu),
            ("I0", self.mpm_model.muI_I0),
            ("grain_diameter", self.mpm_model.muI_d),
            ("grain_density", self.mpm_model.muI_rho_s),
            ("phi_init", self.mpm_model.muI_phi_init),
            ("phi_chi", self.mpm_model.muI_phi_chi),
        ):
            if key in kwargs:
                wp.to_torch(arr)[mat_id] = float(kwargs[key])

        if "g" in kwargs:
            self.set_gravity(kwargs["g"])

        if "density" in kwargs:
            density_value = kwargs["density"]
            wp.launch(
                kernel=set_value_to_float_array,
                dim=self.n_particles,
                inputs=[self.mpm_state.particle_density, density_value],
                device=device,
            )
            wp.launch(
                kernel=get_float_array_product,
                dim=self.n_particles,
                inputs=[
                    self.mpm_state.particle_density,
                    self.mpm_state.particle_vol,
                    self.mpm_state.particle_mass,
                ],
                device=device,
            )

        # compute per-particle mu/lam from per-type E/nu if E or nu was set
        if "E" in kwargs or "nu" in kwargs:
            wp.launch(
                kernel=compute_mu_lam_from_E_nu,
                dim=self.n_particles,
                inputs=[self.mpm_state, self.mpm_model],
                device=device,
            )

        if "rpic_damping" in kwargs:
            self.mpm_model.rpic_damping = kwargs["rpic_damping"]
        if "grid_v_damping_scale" in kwargs:
            self.mpm_model.grid_v_damping_scale = kwargs["grid_v_damping_scale"]

        if "additional_material_params" in kwargs:
            for params in kwargs["additional_material_params"]:
                param_modifier = MaterialParamsModifier()
                param_modifier.point = wp.vec3(params["point"])
                param_modifier.size = wp.vec3(params["size"])
                param_modifier.density = params["density"]
                wp.launch(
                    kernel=apply_additional_params,
                    dim=self.n_particles,
                    inputs=[self.mpm_state, self.mpm_model, param_modifier],
                    device=device,
                )

            wp.launch(
                kernel=get_float_array_product,
                dim=self.n_particles,
                inputs=[
                    self.mpm_state.particle_density,
                    self.mpm_state.particle_vol,
                    self.mpm_state.particle_mass,
                ],
                device=device,
            )


    def finalize_mu_lam(self, device="cuda:0"):
        wp.launch(kernel=compute_mu_lam_from_E_nu, dim=self.n_particles,
                  inputs=[self.mpm_state, self.mpm_model], device=device)

    # kept for backward compatibility
    def finalize_mu_lam_bulk(self, device="cuda:0"):
        self.finalize_mu_lam(device=device)

    def set_gravity(self, g):
        self.mpm_model.gravitational_accelaration = wp.vec3(g[0], g[1], g[2])

    # ------------------------------------------------------------------
    # Rigid body API
    # ------------------------------------------------------------------

    def initialize_rigid_bodies(self, n_bodies, device="cuda:0"):
        """Allocate per-body state arrays for n_bodies rigid bodies.
        Call this before finalize_rigid_bodies()."""
        self.n_rigid_bodies = n_bodies
        self.rigid_x_cm = wp.zeros(n_bodies, dtype=wp.vec3, device=device)
        self.rigid_v_cm = wp.zeros(n_bodies, dtype=wp.vec3, device=device)
        self.rigid_omega = wp.zeros(n_bodies, dtype=wp.vec3, device=device)
        self.rigid_orientation = wp.from_numpy(
            np.tile(np.eye(3, dtype=np.float32), (n_bodies, 1, 1)).reshape(n_bodies, 3, 3),
            dtype=wp.mat33, device=device,
        )
        self.rigid_mass = wp.zeros(n_bodies, dtype=float, device=device)
        self.rigid_inv_inertia_body = wp.zeros(n_bodies, dtype=wp.mat33, device=device)
        self._rigid_linear_mom = wp.zeros(n_bodies, dtype=wp.vec3, device=device)
        self._rigid_angular_mom = wp.zeros(n_bodies, dtype=wp.vec3, device=device)

    def finalize_rigid_bodies(self, device="cuda:0"):
        """Compute center of mass, inertia tensor, and reference positions for
        all rigid bodies from the current particle layout.  Must be called after
        all set_parameters_for_particles() calls that assign material="rigid" and
        obj_id, and before the first p2g2p() step."""
        x_np = self.mpm_state.particle_x.numpy()          # (n, 3)
        m_np = self.mpm_state.particle_mass.numpy()        # (n,)
        mat_np = self.mpm_state.particle_material.numpy()  # (n,)
        rid_np = self.mpm_state.particle_rigid_id.numpy()  # (n,)

        # auto-detect number of rigid bodies from assigned obj_ids
        rigid_mask = mat_np == 8
        if not np.any(rigid_mask):
            return
        n_bodies = int(rid_np[rigid_mask].max()) + 1
        self.initialize_rigid_bodies(n_bodies, device=device)

        x_cm_np = np.zeros((self.n_rigid_bodies, 3), dtype=np.float32)
        mass_np = np.zeros(self.n_rigid_bodies, dtype=np.float32)
        for b in range(self.n_rigid_bodies):
            idx = np.where((mat_np == 8) & (rid_np == b))[0]
            m_b = m_np[idx]
            mass_np[b] = float(m_b.sum())
            x_cm_np[b] = (x_np[idx] * m_b[:, None]).sum(axis=0) / mass_np[b]

        # inertia tensor in body frame (= world frame at t=0 since orientation = I)
        I_body_inv_np = np.zeros((self.n_rigid_bodies, 3, 3), dtype=np.float32)
        x_ref_np = np.zeros_like(x_np)
        for b in range(self.n_rigid_bodies):
            idx = np.where((mat_np == 8) & (rid_np == b))[0]
            r = x_np[idx] - x_cm_np[b]          # (n_b, 3)
            x_ref_np[idx] = r                    # body-frame reference positions
            m_b = m_np[idx]
            I = np.zeros((3, 3), dtype=np.float64)
            for i, (ri, mi) in enumerate(zip(r, m_b)):
                I += mi * (np.dot(ri, ri) * np.eye(3) - np.outer(ri, ri))
            I_body_inv_np[b] = np.linalg.inv(I).astype(np.float32)

        # push to GPU
        self.rigid_x_cm = wp.from_numpy(x_cm_np, dtype=wp.vec3, device=device)
        self.rigid_mass = wp.from_numpy(mass_np, dtype=float, device=device)
        self.rigid_inv_inertia_body = wp.from_numpy(
            I_body_inv_np.reshape(self.n_rigid_bodies, 3, 3), dtype=wp.mat33, device=device
        )
        self.mpm_state.particle_x_ref = wp.from_numpy(x_ref_np, dtype=wp.vec3, device=device)

    def set_rigid_body_velocity(self, body_id, v_cm=(0, 0, 0), omega=(0, 0, 0), device="cuda:0"):
        """Set the initial linear and angular velocity of a rigid body."""
        v_np = wp.to_torch(self.rigid_v_cm)
        o_np = wp.to_torch(self.rigid_omega)
        v_np[body_id] = torch.tensor(v_cm, dtype=torch.float32)
        o_np[body_id] = torch.tensor(omega, dtype=torch.float32)

    def _apply_rigid_restitution(self, dt):
        """Apply restitution impulses to rigid bodies for registered surfaces.
        Must be called after rigid_body_integrate and before rigid_particle_update."""
        if not self.rigid_surface_colliders or self.n_rigid_bodies == 0:
            return

        x_cm_np   = self.rigid_x_cm.numpy()              # (n_bodies, 3)
        v_cm_np   = self.rigid_v_cm.numpy()              # (n_bodies, 3)
        omega_np  = self.rigid_omega.numpy()             # (n_bodies, 3)
        R_np      = self.rigid_orientation.numpy()       # (n_bodies, 3, 3)
        M_np      = self.rigid_mass.numpy()              # (n_bodies,)
        I_inv_np  = self.rigid_inv_inertia_body.numpy()  # (n_bodies, 3, 3)
        x_ref_np  = self.mpm_state.particle_x_ref.numpy()       # (n_particles, 3)
        mat_np    = self.mpm_state.particle_material.numpy()     # (n_particles,)
        rid_np    = self.mpm_state.particle_rigid_id.numpy()     # (n_particles,)

        modified = False
        for b in range(self.n_rigid_bodies):
            idx = np.where((mat_np == 8) & (rid_np == b))[0]
            if len(idx) == 0:
                continue

            R     = R_np[b]       # (3, 3)
            x_cm  = x_cm_np[b]   # (3,)
            M     = float(M_np[b])
            I_inv = I_inv_np[b]   # (3, 3)

            # World-frame inverse inertia: I_world_inv = R * I_body_inv * R^T
            I_world_inv = R @ I_inv @ R.T

            # New particle world positions after rigid_body_integrate
            x_refs  = x_ref_np[idx]              # (n_b, 3)
            x_world = x_cm + (R @ x_refs.T).T   # (n_b, 3)

            for surf in self.rigid_surface_colliders:
                if self.time < surf["start_time"] or self.time >= surf["end_time"]:
                    continue

                pt = np.array(surf["point"],  dtype=np.float64)
                n  = np.array(surf["normal"], dtype=np.float64)
                e  = float(surf["restitution"])
                mu = float(surf["friction"])

                # Signed distance of each particle from the plane (negative = inside)
                dists = (x_world - pt) @ n  # (n_b,)
                min_i = int(np.argmin(dists))
                if dists[min_i] >= 0.0:
                    continue  # no penetration

                # Contact point and lever arm
                x_contact = x_world[min_i].astype(np.float64)
                r = x_contact - x_cm.astype(np.float64)

                # Contact velocity at the contact point
                v_contact = v_cm_np[b].astype(np.float64) + np.cross(omega_np[b].astype(np.float64), r)
                v_n = float(np.dot(v_contact, n))
                if v_n >= 0.0:
                    continue  # already separating

                # Effective inverse mass at contact point along normal
                r_cross_n = np.cross(r, n)
                denom = 1.0 / M + np.dot(np.cross(I_world_inv @ r_cross_n, r), n)

                # Normal impulse magnitude (positive = outward)
                J_n = -(1.0 + e) * v_n / denom

                # Apply normal impulse
                v_cm_np[b]  = v_cm_np[b].astype(np.float64) + (J_n / M) * n
                omega_np[b] = omega_np[b].astype(np.float64) + I_world_inv @ (J_n * r_cross_n)

                # Friction impulse (Coulomb, opposing tangential contact velocity)
                if mu > 0.0:
                    v_t_vec = v_contact - v_n * n
                    v_t_mag = float(np.linalg.norm(v_t_vec))
                    if v_t_mag > 1e-10:
                        t_hat = v_t_vec / v_t_mag
                        r_cross_t = np.cross(r, t_hat)
                        denom_t = 1.0 / M + np.dot(np.cross(I_world_inv @ r_cross_t, r), t_hat)
                        # Impulse to fully stop sliding, capped by Coulomb limit
                        J_t = min(v_t_mag / denom_t, mu * J_n)
                        v_cm_np[b]  = v_cm_np[b] - (J_t / M) * t_hat
                        omega_np[b] = omega_np[b] - I_world_inv @ (J_t * r_cross_t)

                modified = True

        if modified:
            v_cm_torch    = wp.to_torch(self.rigid_v_cm)
            omega_torch   = wp.to_torch(self.rigid_omega)
            v_cm_torch[:] = torch.from_numpy(v_cm_np.astype(np.float32)).to(v_cm_torch.device)
            omega_torch[:] = torch.from_numpy(omega_np.astype(np.float32)).to(omega_torch.device)

    def rebuild_active_blocks(self, device="cpu"):
        """Rebuild the 4^3 active-block list from current particle positions (call once
        per control tick). Marks each particle's stencil-base block, dilates by one block
        (stencil crossing + intra-tick motion), compacts to a current list and a union
        list with the previous tick (union feeds zeroing so departing nodes clear once).
        One small device-to-host readback per call (the two counts)."""
        if self._blk is None:
            bd = (int(np.ceil(self.mpm_model.grid_dim_x / 4)),
                  int(np.ceil(self.mpm_model.grid_dim_y / 4)),
                  int(np.ceil(self.mpm_model.grid_dim_z / 4)))
            nb = bd[0] * bd[1] * bd[2]
            self._blk = {
                "bd": bd,
                "raw": wp.zeros(bd, dtype=int, device=device),
                "cur": wp.zeros(bd, dtype=int, device=device),
                "prev": wp.zeros(bd, dtype=int, device=device),
                "cur_list": wp.zeros(nb, dtype=int, device=device),
                "union_list": wp.zeros(nb, dtype=int, device=device),
                "counts": wp.zeros(2, dtype=int, device=device),
                "cur_n": 0,
                "union_n": 0,
            }
        b = self._blk
        b["prev"], b["cur"] = b["cur"], b["prev"]
        b["raw"].zero_()
        bd_v = wp.vec3i(b["bd"][0], b["bd"][1], b["bd"][2])
        wp.launch(kernel=blocks_mark, dim=self.n_particles,
                  inputs=[self.mpm_state.particle_x, self.mpm_model.inv_dx, bd_v,
                          b["raw"]], device=device)
        wp.launch(kernel=blocks_dilate, dim=b["bd"],
                  inputs=[b["raw"], bd_v, b["cur"]], device=device)
        b["counts"].zero_()
        wp.launch(kernel=blocks_compact, dim=b["bd"],
                  inputs=[b["cur"], b["prev"], bd_v, b["cur_list"], b["union_list"],
                          b["counts"]], device=device)
        c = b["counts"].numpy()
        b["cur_n"], b["union_n"] = int(c[0]), int(c[1])
        self.sparse_ready = True

    def active_block_fraction(self):
        """Fraction of grid blocks in the current active set (diagnostics/benchmarks)."""
        if not self.sparse_ready:
            return 1.0
        bd = self._blk["bd"]
        return self._blk["cur_n"] / float(bd[0] * bd[1] * bd[2])

    def _capture_substep_graphs(self, dt, device, grid_size):
        """CUDA-graph capture of the fixed-shape substep segments (speedup_plan Step 3).
        Segment A: zero -> stress -> p2g -> normalize (+ damping when enabled); segment
        B: g2p. Captured at FULL grid dims so the graph shape never changes (the particle
        box restriction stays a CPU optimization; dense sweeps are cheap on GPU). The BC
        launches and host-side pose integration stay live between the two graphs because
        their inputs (time, poses, restricted dims) change per substep."""
        full_lo = wp.vec3i(0, 0, 0)
        wp.capture_begin(device)
        try:
            wp.launch(kernel=zero_grid, dim=grid_size,
                      inputs=[self.mpm_state, self.mpm_model, full_lo], device=device)
            wp.launch(kernel=compute_stress_from_F_trial, dim=self.n_particles,
                      inputs=[self.mpm_state, self.mpm_model, dt], device=device)
            wp.launch(kernel=p2g_apic_with_stress, dim=self.n_particles,
                      inputs=[self.mpm_state, self.mpm_model, dt], device=device)
            wp.launch(kernel=grid_normalization_and_gravity, dim=grid_size,
                      inputs=[self.mpm_state, self.mpm_model, dt, full_lo], device=device)
            if self.mpm_model.grid_v_damping_scale < 1.0:
                wp.launch(kernel=add_damping_via_grid, dim=grid_size,
                          inputs=[self.mpm_state, self.mpm_model.grid_v_damping_scale,
                                  full_lo], device=device)
        finally:
            self._graph_A = wp.capture_end(device)
        wp.capture_begin(device)
        try:
            wp.launch(kernel=g2p, dim=self.n_particles,
                      inputs=[self.mpm_state, self.mpm_model, dt], device=device)
        finally:
            self._graph_B = wp.capture_end(device)

    def p2g2p(self, step, dt, device="cuda:0"):
        grid_size = (
            self.mpm_model.grid_dim_x,
            self.mpm_model.grid_dim_y,
            self.mpm_model.grid_dim_z,
        )
        # CUDA graph fast path: replay the fixed-shape inner segments as graphs. The
        # first substep runs live so warp modules JIT-load before capture; particle
        # modifiers take self.time per substep so their presence disables capture.
        sparse = self.sparse_ready and self.restrict_grid
        use_graph = (
            self.use_cuda_graph
            and not sparse
            and self._graph_warmup > 0
            and str(device).startswith("cuda")
            and not self.pre_p2g_operations
            and not self.particle_velocity_modifiers
        )
        if use_graph:
            sig = (float(dt), bool(self.mpm_model.grid_v_damping_scale < 1.0))
            if self._graph_sig != sig:
                try:
                    self._capture_substep_graphs(dt, device, grid_size)
                    self._graph_sig = sig
                except Exception:
                    self._graph_A = self._graph_B = self._graph_sig = None
                    self.use_cuda_graph = False
                    use_graph = False
        self._graph_warmup = 1

        if sparse:
            self._prev_grid_box = None
            b = self._blk
            bd_v = wp.vec3i(b["bd"][0], b["bd"][1], b["bd"][2])
            if b["union_n"] > 0:
                wp.launch(kernel=zero_grid_blocks, dim=b["union_n"] * 64,
                          inputs=[self.mpm_state, self.mpm_model, b["union_list"], bd_v],
                          device=device)
        elif use_graph:
            # graphs zero and normalize the FULL grid (dense sweeps are cheap on GPU;
            # the box restriction stays a CPU optimization), so no union bookkeeping
            self._prev_grid_box = None
            wp.capture_launch(self._graph_A)
        else:
            # restricted grid sweeps: zero/normalize/damping run over the live particle
            # box when the wrapper has set one; zeroing uses the union with the previous
            # box so nodes leaving the box are cleared exactly once
            gbox = (self.grid_launch_box
                    if (self.restrict_grid and self.grid_launch_box) else None)
            if gbox is None:
                g_lo, g_dims = wp.vec3i(0, 0, 0), grid_size
                z_lo, z_dims = g_lo, g_dims
                self._prev_grid_box = None
            else:
                g_lo, g_dims = wp.vec3i(*gbox[0]), gbox[1]
                prev = self._prev_grid_box
                if prev is None:
                    z_lo, z_dims = g_lo, g_dims
                else:
                    zl = tuple(min(prev[0][i], gbox[0][i]) for i in range(3))
                    zh = tuple(max(prev[0][i] + prev[1][i], gbox[0][i] + gbox[1][i])
                               for i in range(3))
                    z_lo = wp.vec3i(*zl)
                    z_dims = tuple(zh[i] - zl[i] for i in range(3))
                self._prev_grid_box = gbox
            wp.launch(
                kernel=zero_grid,
                dim=z_dims,
                inputs=[self.mpm_state, self.mpm_model, z_lo],
                device=device,
            )

        # apply pre-p2g operations on particles
        for k in range(len(self.pre_p2g_operations)):
            wp.launch(
                kernel=self.pre_p2g_operations[k],
                dim=self.n_particles,
                inputs=[self.time, dt, self.mpm_state, self.impulse_params[k]],
                device=device,
            )
        # apply dirichlet particle v modifier
        for k in range(len(self.particle_velocity_modifiers)):
            wp.launch(
                kernel = self.particle_velocity_modifiers[k],
                dim = self.n_particles,
                inputs=[self.time, self.mpm_state, self.particle_velocity_modifier_params[k]],
                device=device,
            )

        if not use_graph:
            # compute stress = stress(returnMap(F_trial))
            with wp.ScopedTimer(
                "compute_stress_from_F_trial",
                synchronize=self.profile, active=self.profile,
                print=False,
                dict=self.time_profile,
            ):
                wp.launch(
                    kernel=compute_stress_from_F_trial,
                    dim=self.n_particles,
                    inputs=[self.mpm_state, self.mpm_model, dt],
                    device=device,
                )  # F and stress are updated

            # p2g
            with wp.ScopedTimer(
                "p2g",
                synchronize=self.profile, active=self.profile,
                print=False,
                dict=self.time_profile,
            ):
                wp.launch(
                    kernel=p2g_apic_with_stress,
                    dim=self.n_particles,
                    inputs=[self.mpm_state, self.mpm_model, dt],
                    device=device,
                )  # apply p2g'

            # grid update
            with wp.ScopedTimer(
                "grid_update", synchronize=self.profile, active=self.profile, print=False, dict=self.time_profile
            ):
                if sparse:
                    b = self._blk
                    bd_v = wp.vec3i(b["bd"][0], b["bd"][1], b["bd"][2])
                    if b["cur_n"] > 0:
                        wp.launch(kernel=grid_normalization_and_gravity_blocks,
                                  dim=b["cur_n"] * 64,
                                  inputs=[self.mpm_state, self.mpm_model, dt,
                                          b["cur_list"], bd_v], device=device)
                else:
                    wp.launch(
                        kernel=grid_normalization_and_gravity,
                        dim=g_dims,
                        inputs=[self.mpm_state, self.mpm_model, dt, g_lo],
                        device=device,
                    )

            if self.mpm_model.grid_v_damping_scale < 1.0:
                if sparse:
                    b = self._blk
                    bd_v = wp.vec3i(b["bd"][0], b["bd"][1], b["bd"][2])
                    if b["cur_n"] > 0:
                        wp.launch(kernel=add_damping_via_grid_blocks,
                                  dim=b["cur_n"] * 64,
                                  inputs=[self.mpm_state, self.mpm_model,
                                          self.mpm_model.grid_v_damping_scale,
                                          b["cur_list"], bd_v], device=device)
                else:
                    wp.launch(
                        kernel=add_damping_via_grid,
                        dim=g_dims,
                        inputs=[self.mpm_state, self.mpm_model.grid_v_damping_scale, g_lo],
                        device=device,
                    )

        # apply BC on grid
        with wp.ScopedTimer(
            "apply_BC_on_grid", synchronize=self.profile, active=self.profile, print=False, dict=self.time_profile
        ):
            for k in range(len(self.grid_postprocess)):
                # restrict the launch to the collider's current grid box when it has one;
                # a None box means the collider is outside the domain (skip the launch but
                # still integrate its pose below)
                lo_v = wp.vec3i(0, 0, 0)
                dims = grid_size
                skip = False
                fn = self.collider_aabbs[k] if k < len(self.collider_aabbs) else None
                if self.restrict_bc and fn is not None:
                    box = fn()
                    if box is None:
                        skip = True
                    else:
                        lo_v = wp.vec3i(int(box[0][0]), int(box[0][1]), int(box[0][2]))
                        dims = box[1]
                if not skip:
                    wp.launch(
                        kernel=self.grid_postprocess[k],
                        dim=dims,
                        inputs=[
                            self.time,
                            dt,
                            self.mpm_state,
                            self.mpm_model,
                            self.collider_params[k],
                            lo_v,
                        ],
                        device=device,
                    )
                if self.modify_bc[k] is not None:
                    self.modify_bc[k](self.time, dt, self.collider_params[k])

        # g2p
        with wp.ScopedTimer(
            "g2p", synchronize=self.profile, active=self.profile, print=False, dict=self.time_profile
        ):
            if use_graph:
                wp.capture_launch(self._graph_B)
            else:
                wp.launch(
                    kernel=g2p,
                    dim=self.n_particles,
                    inputs=[self.mpm_state, self.mpm_model, dt],
                    device=device,
                )  # x, v, C, F_trial are updated

        # rigid body step (skipped when no rigid bodies are present)
        if self.n_rigid_bodies > 0:
            with wp.ScopedTimer("rigid_body", synchronize=self.profile, active=self.profile, print=False, dict=self.time_profile):
                # zero accumulation buffers
                wp.launch(kernel=set_vec3_to_zero, dim=self.n_rigid_bodies,
                          inputs=[self._rigid_linear_mom], device=device)
                wp.launch(kernel=set_vec3_to_zero, dim=self.n_rigid_bodies,
                          inputs=[self._rigid_angular_mom], device=device)
                # gather grid momentum weighted by particle mass
                wp.launch(kernel=rigid_g2p_accumulate, dim=self.n_particles,
                          inputs=[self.mpm_state, self.mpm_model,
                                  self.rigid_x_cm, self._rigid_linear_mom,
                                  self._rigid_angular_mom],
                          device=device)
                # integrate rigid body EOM and update orientation
                wp.launch(kernel=rigid_body_integrate, dim=self.n_rigid_bodies,
                          inputs=[self.rigid_x_cm, self.rigid_v_cm, self.rigid_omega,
                                  self.rigid_orientation, self.rigid_mass,
                                  self.rigid_inv_inertia_body,
                                  self._rigid_linear_mom, self._rigid_angular_mom, dt],
                          device=device)
                # apply restitution impulses to rigid bodies (surface colliders with e > 0)
                self._apply_rigid_restitution(dt)
                # push updated state back to particles
                wp.launch(kernel=rigid_particle_update, dim=self.n_particles,
                          inputs=[self.mpm_state, self.rigid_x_cm, self.rigid_v_cm,
                                  self.rigid_omega, self.rigid_orientation],
                          device=device)

        #### CFL check ####
        # particle_v = self.mpm_state.particle_v.numpy()
        # if np.max(np.abs(particle_v)) > self.mpm_model.dx / dt:
        #     print("max particle v: ", np.max(np.abs(particle_v)))
        #     print("max allowed  v: ", self.mpm_model.dx / dt)
        #     print("does not allow v*dt>dx")
        #     input()
        #### CFL check ####
        self.time = self.time + dt

    # set particle densities to all_particle_densities, 
    def reset_densities_and_update_masses(self, all_particle_densities, device = "cuda:0"):
        all_particle_densities = all_particle_densities.clone().detach()
        self.mpm_state.particle_density = torch2warp_float(all_particle_densities, dvc=device)
        wp.launch(
                kernel=get_float_array_product,
                dim=self.n_particles,
                inputs=[
                    self.mpm_state.particle_density,
                    self.mpm_state.particle_vol,
                    self.mpm_state.particle_mass,
                ],
                device=device,
            )

    # clone = True makes a copy, not necessarily needed
    def import_particle_x_from_torch(self, tensor_x, clone=True, device="cuda:0"):
        if tensor_x is not None:
            if clone:
                tensor_x = tensor_x.clone().detach()
            self.mpm_state.particle_x = torch2warp_vec3(tensor_x, dvc=device)

    # clone = True makes a copy, not necessarily needed
    def import_particle_v_from_torch(self, tensor_v, clone=True, device="cuda:0"):
        if tensor_v is not None:
            if clone:
                tensor_v = tensor_v.clone().detach()
            self.mpm_state.particle_v = torch2warp_vec3(tensor_v, dvc=device)

    # clone = True makes a copy, not necessarily needed
    def import_particle_F_from_torch(self, tensor_F, clone=True, device="cuda:0"):
        if tensor_F is not None:
            if clone:
                tensor_F = tensor_F.clone().detach()
            tensor_F = torch.reshape(tensor_F, (-1, 3, 3))  # arranged by rowmajor
            self.mpm_state.particle_F = torch2warp_mat33(tensor_F, dvc=device)

    # clone = True makes a copy, not necessarily needed
    def import_particle_C_from_torch(self, tensor_C, clone=True, device="cuda:0"):
        if tensor_C is not None:
            if clone:
                tensor_C = tensor_C.clone().detach()
            tensor_C = torch.reshape(tensor_C, (-1, 3, 3))  # arranged by rowmajor
            self.mpm_state.particle_C = torch2warp_mat33(tensor_C, dvc=device)
            
    def import_particle_selection_from_torch(self, tensor_selection, clone=True, device="cuda:0"):
        if tensor_selection is not None:
            if clone:
                tensor_selection = tensor_selection.clone().detach()
            self.mpm_state.particle_selection = torch2warp_int(tensor_selection, dvc=device)

    def import_particle_material_from_torch(self, tensor_material, clone=True, device="cuda:0"):
        if tensor_material is not None:
            if clone:
                tensor_material = tensor_material.clone().detach()
            self.mpm_state.particle_material = torch2warp_int(tensor_material, dvc=device)

    def export_particle_material_to_torch(self):
        return wp.to_torch(self.mpm_state.particle_material)

    def set_parameters_for_particles(self, start_idx, end_idx, params_dict, device="cuda:0"):
        """Set material type and parameters for particles in range [start_idx, end_idx)."""
        import torch

        mat_id = None
        if "material" in params_dict:
            mat_id = self._material_name_to_id(params_dict["material"])
            # set particle_material for the range
            mat_torch = wp.to_torch(self.mpm_state.particle_material)
            mat_torch[start_idx:end_idx] = mat_id

        if mat_id is None:
            mat_id = self.mpm_model.material

        # per-type parameters (write to model arrays at mat_id index)
        if "E" in params_dict:
            E_torch = wp.to_torch(self.mpm_model.E)
            E_torch[mat_id] = params_dict["E"]
        if "nu" in params_dict:
            nu_torch = wp.to_torch(self.mpm_model.nu)
            nu_torch[mat_id] = params_dict["nu"]
        if "bulk_modulus" in params_dict:
            bulk_torch = wp.to_torch(self.mpm_model.bulk)
            bulk_torch[mat_id] = params_dict["bulk_modulus"]
        if "hardening" in params_dict:
            h_torch = wp.to_torch(self.mpm_model.hardening)
            h_torch[mat_id] = float(params_dict["hardening"])
        if "xi" in params_dict:
            xi_torch = wp.to_torch(self.mpm_model.xi)
            xi_torch[mat_id] = params_dict["xi"]
        if "friction_angle" in params_dict:
            sin_phi = wp.sin(params_dict["friction_angle"] / 180.0 * 3.14159265)
            alpha_val = wp.sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi)
            alpha_torch = wp.to_torch(self.mpm_model.alpha)
            alpha_torch[mat_id] = alpha_val
        if "plastic_viscosity" in params_dict:
            pv_torch = wp.to_torch(self.mpm_model.plastic_viscosity)
            pv_torch[mat_id] = params_dict["plastic_viscosity"]
        if "softening" in params_dict:
            s_torch = wp.to_torch(self.mpm_model.softening)
            s_torch[mat_id] = params_dict["softening"]

        # local mu(I) rheology parameters (material 9 and 11), per material type
        for key, arr in (
            ("mu_s", self.mpm_model.muI_mu_s),
            ("delta_mu", self.mpm_model.muI_delta_mu),
            ("I0", self.mpm_model.muI_I0),
            ("grain_diameter", self.mpm_model.muI_d),
            ("grain_density", self.mpm_model.muI_rho_s),
            ("phi_init", self.mpm_model.muI_phi_init),
            ("phi_chi", self.mpm_model.muI_phi_chi),
        ):
            if key in params_dict:
                wp.to_torch(arr)[mat_id] = float(params_dict[key])

        # rigid body id — only meaningful when material == "rigid" (8)
        if "obj_id" in params_dict:
            rid_torch = wp.to_torch(self.mpm_state.particle_rigid_id)
            rid_torch[start_idx:end_idx] = int(params_dict["obj_id"])

        # per-particle parameters (set for the range only)
        if "density" in params_dict:
            density_torch = wp.to_torch(self.mpm_state.particle_density)
            density_torch[start_idx:end_idx] = params_dict["density"]
            mass_torch = wp.to_torch(self.mpm_state.particle_mass)
            vol_torch = wp.to_torch(self.mpm_state.particle_vol)
            mass_torch[start_idx:end_idx] = density_torch[start_idx:end_idx] * vol_torch[start_idx:end_idx]

        if "yield_stress" in params_dict:
            ys_torch = wp.to_torch(self.mpm_model.yield_stress)
            ys_torch[start_idx:end_idx] = params_dict["yield_stress"]

        # compute mu/lam for this range from per-type E/nu
        if "E" in params_dict or "nu" in params_dict:
            E_torch = wp.to_torch(self.mpm_model.E)
            nu_torch = wp.to_torch(self.mpm_model.nu)
            E_val = float(E_torch[mat_id])
            nu_val = float(nu_torch[mat_id])
            mu_val = E_val / (2.0 * (1.0 + nu_val))
            lam_val = E_val * nu_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val))
            mu_torch = wp.to_torch(self.mpm_model.mu)
            lam_torch = wp.to_torch(self.mpm_model.lam)
            mu_torch[start_idx:end_idx] = mu_val
            lam_torch[start_idx:end_idx] = lam_val

    def export_particle_x_to_torch(self):
        return wp.to_torch(self.mpm_state.particle_x)

    def export_particle_v_to_torch(self):
        return wp.to_torch(self.mpm_state.particle_v)
    
    def export_particle_vol_to_torch(self):
        return wp.to_torch(self.mpm_state.particle_vol)
    
    def export_particle_stress_to_torch(self):
        stress_tensor = wp.to_torch(self.mpm_state.particle_stress)
        stress_tensor = stress_tensor.reshape(-1, 9)
        return stress_tensor

    def export_particle_F_to_torch(self):
        F_tensor = wp.to_torch(self.mpm_state.particle_F)
        F_tensor = F_tensor.reshape(-1, 9)
        return F_tensor

    def export_particle_L_to_torch(self):
        # velocity gradient L_ij = dv_i/dx_j from the most recent g2p
        L_tensor = wp.to_torch(self.mpm_state.particle_L)
        return L_tensor.reshape(-1, 9)

    def export_particle_R_to_torch(self, device="cuda:0"):
        with wp.ScopedTimer(
            "compute_R_from_F",
            synchronize=self.profile, active=self.profile,
            print=False,
            dict=self.time_profile,
        ):
            wp.launch(
                kernel=compute_R_from_F,
                dim=self.n_particles,
                inputs=[self.mpm_state, self.mpm_model],
                device=device,
            )

        R_tensor = wp.to_torch(self.mpm_state.particle_R)
        R_tensor = R_tensor.reshape(-1, 9)
        return R_tensor

    def export_particle_C_to_torch(self):
        C_tensor = wp.to_torch(self.mpm_state.particle_C)
        C_tensor = C_tensor.reshape(-1, 9)
        return C_tensor

    def export_particle_cov_to_torch(self, device="cuda:0"):
        if not self.mpm_model.update_cov_with_F:
            with wp.ScopedTimer(
                "compute_cov_from_F",
                synchronize=self.profile, active=self.profile,
                print=False,
                dict=self.time_profile,
            ):
                wp.launch(
                    kernel=compute_cov_from_F,
                    dim=self.n_particles,
                    inputs=[self.mpm_state, self.mpm_model],
                    device=device,
                )

        cov = wp.to_torch(self.mpm_state.particle_cov)
        return cov
    def export_particle_selection_to_torch(self):
        selection_tensor = wp.to_torch(self.mpm_state.particle_selection)
        return selection_tensor
    def export_grid_v_out_to_torch(self):
        grid_v_tensor = wp.to_torch(self.mpm_state.grid_v_out)
        return grid_v_tensor
    def export_grid_v_in_to_torch(self):
        grid_v_tensor = wp.to_torch(self.mpm_state.grid_v_in)
        return grid_v_tensor
    def export_grid_m_to_torch(self):
        grid_m_tensor = wp.to_torch(self.mpm_state.grid_m)
        return grid_m_tensor
    
    def print_time_profile(self):
        print("MPM Time profile:")
        for key, value in self.time_profile.items():
            print(key, sum(value))

    # a surface specified by a point and the normal vector
    def _grid_box(self, lo_w, hi_w, halo=1):
        """World-space box -> clamped grid-index (lo, dim) for a restricted BC launch, or
        None when the box misses the grid entirely."""
        dx = self.mpm_model.dx
        n = (self.mpm_model.grid_dim_x, self.mpm_model.grid_dim_y, self.mpm_model.grid_dim_z)
        lo = [max(0, int(np.floor(float(lo_w[i]) / dx)) - halo) for i in range(3)]
        hi = [min(n[i], int(np.ceil(float(hi_w[i]) / dx)) + halo + 1) for i in range(3)]
        dims = (hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])
        if dims[0] <= 0 or dims[1] <= 0 or dims[2] <= 0:
            return None
        return (tuple(lo), dims)

    def add_surface_collider(
        self,
        point,
        normal,
        surface="sticky",
        friction=0.0,
        restitution=0.0,
        start_time=0.0,
        end_time=999.0,
    ):
        point = list(point)
        # Normalize normal
        normal_scale = 1.0 / wp.sqrt(float(sum(x**2 for x in normal)))
        normal = list(normal_scale * x for x in normal)

        collider_param = Dirichlet_collider()
        collider_param.start_time = start_time
        collider_param.end_time = end_time

        collider_param.point = wp.vec3(point[0], point[1], point[2])
        collider_param.normal = wp.vec3(normal[0], normal[1], normal[2])

        if surface == "sticky" and friction != 0:
            raise ValueError("friction must be 0 on sticky surfaces.")
        if surface == "sticky":
            collider_param.surface_type = 0
        elif surface == "slip":
            collider_param.surface_type = 1
        elif surface == "cut":
            collider_param.surface_type = 11
        else:
            collider_param.surface_type = 2
        # frictional
        collider_param.friction = friction

        if restitution != 0.0:
            if not (0.0 < restitution <= 1.0):
                raise ValueError("restitution must be between 0 (exclusive) and 1 (inclusive).")
            self.rigid_surface_colliders.append({
                "point": list(point),
                "normal": list(normal),
                "friction": friction,
                "restitution": restitution,
                "start_time": start_time,
                "end_time": end_time,
            })

        self.collider_params.append(collider_param)

        @wp.kernel
        def collide(
            time: float,
            dt: float,
            state: MPMStateStruct,
            model: MPMModelStruct,
            param: Dirichlet_collider,
            lo: wp.vec3i,
        ):
            grid_x, grid_y, grid_z = wp.tid()
            grid_x = grid_x + lo[0]
            grid_y = grid_y + lo[1]
            grid_z = grid_z + lo[2]
            if time >= param.start_time and time < param.end_time:
                offset = wp.vec3(
                    float(grid_x) * model.dx - param.point[0],
                    float(grid_y) * model.dx - param.point[1],
                    float(grid_z) * model.dx - param.point[2],
                )
                n = wp.vec3(param.normal[0], param.normal[1], param.normal[2])
                dotproduct = wp.dot(offset, n)

                if dotproduct < 0.0:
                    if param.surface_type == 0:
                        state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                            0.0, 0.0, 0.0
                        )
                    elif param.surface_type == 11:
                        if (
                            float(grid_z) * model.dx < 0.4
                            or float(grid_z) * model.dx > 0.53
                        ):
                            state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                                0.0, 0.0, 0.0
                            )
                        else:
                            v_in = state.grid_v_out[grid_x, grid_y, grid_z]
                            state.grid_v_out[grid_x, grid_y, grid_z] = (
                                wp.vec3(v_in[0], 0.0, v_in[2]) * 0.3
                            )
                    else:
                        v = state.grid_v_out[grid_x, grid_y, grid_z]
                        normal_component = wp.dot(v, n)
                        if param.surface_type == 1:
                            v = (
                                v - normal_component * n
                            )  # Project out all normal component
                        else:
                            v = (
                                v - wp.min(normal_component, 0.0) * n
                            )  # Project out only inward normal component
                        if normal_component < 0.0 and wp.length(v) > 1e-20:
                            v = wp.max(
                                0.0, wp.length(v) + normal_component * param.friction
                            ) * wp.normalize(
                                v
                            )  # apply friction here
                        state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                            v[0], v[1], v[2]
                        )

        # restricted-launch box: an axis-aligned plane affects only the half-space
        # behind it, a thin slab when the plane sits near a domain face (the floor).
        # General planes keep the full-grid fallback (entry None).
        _axis = next((i for i in range(3) if abs(abs(normal[i]) - 1.0) < 1e-6), -1)
        if _axis >= 0:
            _p_ax, _sgn = float(point[_axis]), float(normal[_axis])

            def plane_aabb(axis=_axis, p_ax=_p_ax, sgn=_sgn, self=self):
                big = 1.0e9
                lo_w = [-big, -big, -big]
                hi_w = [big, big, big]
                if sgn > 0.0:
                    hi_w[axis] = p_ax
                else:
                    lo_w[axis] = p_ax
                return self._grid_box(lo_w, hi_w, halo=1)

            self.collider_aabbs.append(plane_aabb)
        else:
            self.collider_aabbs.append(None)
        self.grid_postprocess.append(collide)
        self.modify_bc.append(None)

    # a cubiod is a rectangular cube'
    # centered at `point`
    # dimension is x: point[0]±size[0]
    #              y: point[1]±size[1]
    #              z: point[2]±size[2]
    # all grid nodes lie within the cubiod will have their speed set to velocity
    # the cuboid itself is also moving with const speed = velocity
    # set the speed to zero to fix BC
    def set_velocity_on_cuboid(
        self,
        point,
        size,
        velocity,
        start_time=0.0,
        end_time=999.0,
        reset=0,
    ):
        point = list(point)

        collider_param = Dirichlet_collider()
        collider_param.start_time = start_time
        collider_param.end_time = end_time
        collider_param.point = wp.vec3(point[0], point[1], point[2])
        collider_param.size = size
        collider_param.velocity = wp.vec3(velocity[0], velocity[1], velocity[2])
        # collider_param.threshold = threshold
        collider_param.reset = reset
        # Newton-exact reaction-impulse accumulator: each substep the collide kernel adds
        # sum_nodes m*(v_free - v_imposed) (the impulse the material delivers to the cuboid)
        # BEFORE it overwrites the node velocity. Reaction force = (this) / elapsed dt.
        collider_param.force = wp.zeros(1, dtype=wp.vec3, device=self.mpm_state.grid_m.device)
        self.collider_params.append(collider_param)

        @wp.kernel
        def collide(
            time: float,
            dt: float,
            state: MPMStateStruct,
            model: MPMModelStruct,
            param: Dirichlet_collider,
            lo: wp.vec3i,
        ):
            grid_x, grid_y, grid_z = wp.tid()
            grid_x = grid_x + lo[0]
            grid_y = grid_y + lo[1]
            grid_z = grid_z + lo[2]
            if time >= param.start_time and time < param.end_time:
                offset = wp.vec3(
                    float(grid_x) * model.dx - param.point[0],
                    float(grid_y) * model.dx - param.point[1],
                    float(grid_z) * model.dx - param.point[2],
                )
                if (
                    wp.abs(offset[0]) < param.size[0]
                    and wp.abs(offset[1]) < param.size[1]
                    and wp.abs(offset[2]) < param.size[2]
                ):
                    # capture the exact reaction impulse before imposing the BC velocity
                    m = state.grid_m[grid_x, grid_y, grid_z]
                    v_free = state.grid_v_out[grid_x, grid_y, grid_z]
                    wp.atomic_add(param.force, 0, m * (v_free - param.velocity))
                    state.grid_v_out[grid_x, grid_y, grid_z] = param.velocity
            elif param.reset == 1:
                if time < param.end_time + 15.0 * dt:
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)

        def modify(time, dt, param: Dirichlet_collider):
            if time >= param.start_time and time < param.end_time:
                param.point = wp.vec3(
                    param.point[0] + dt * param.velocity[0],
                    param.point[1] + dt * param.velocity[1],
                    param.point[2] + dt * param.velocity[2],
                )  # param.point + dt * param.velocity

        def box_aabb(param=collider_param, self=self):
            p, s = param.point, param.size
            return self._grid_box((p[0] - s[0], p[1] - s[1], p[2] - s[2]),
                                  (p[0] + s[0], p[1] + s[1], p[2] + s[2]), halo=1)

        self.collider_aabbs.append(box_aabb)
        self.grid_postprocess.append(collide)
        self.modify_bc.append(modify)

    def add_bounding_box(self, start_time=0.0, end_time=999.0):
        collider_param = Dirichlet_collider()
        collider_param.start_time = start_time
        collider_param.end_time = end_time

        self.collider_params.append(collider_param)

        @wp.kernel
        def collide(
            time: float,
            dt: float,
            state: MPMStateStruct,
            model: MPMModelStruct,
            param: Dirichlet_collider,
            lo: wp.vec3i,
        ):
            grid_x, grid_y, grid_z = wp.tid()
            grid_x = grid_x + lo[0]
            grid_y = grid_y + lo[1]
            grid_z = grid_z + lo[2]
            padding = 3
            if time >= param.start_time and time < param.end_time:
                if grid_x < padding and state.grid_v_out[grid_x, grid_y, grid_z][0] < 0:
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                        0.0,
                        state.grid_v_out[grid_x, grid_y, grid_z][1],
                        state.grid_v_out[grid_x, grid_y, grid_z][2],
                    )
                if (
                    grid_x >= model.grid_dim_x - padding
                    and state.grid_v_out[grid_x, grid_y, grid_z][0] > 0
                ):
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                        0.0,
                        state.grid_v_out[grid_x, grid_y, grid_z][1],
                        state.grid_v_out[grid_x, grid_y, grid_z][2],
                    )

                if grid_y < padding and state.grid_v_out[grid_x, grid_y, grid_z][1] < 0:
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                        state.grid_v_out[grid_x, grid_y, grid_z][0],
                        0.0,
                        state.grid_v_out[grid_x, grid_y, grid_z][2],
                    )
                if (
                    grid_y >= model.grid_dim_y - padding
                    and state.grid_v_out[grid_x, grid_y, grid_z][1] > 0
                ):
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                        state.grid_v_out[grid_x, grid_y, grid_z][0],
                        0.0,
                        state.grid_v_out[grid_x, grid_y, grid_z][2],
                    )

                if grid_z < padding and state.grid_v_out[grid_x, grid_y, grid_z][2] < 0:
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                        state.grid_v_out[grid_x, grid_y, grid_z][0],
                        state.grid_v_out[grid_x, grid_y, grid_z][1],
                        0.0,
                    )
                if (
                    grid_z >= model.grid_dim_z - padding
                    and state.grid_v_out[grid_x, grid_y, grid_z][2] > 0
                ):
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                        state.grid_v_out[grid_x, grid_y, grid_z][0],
                        state.grid_v_out[grid_x, grid_y, grid_z][1],
                        0.0,
                    )

        self.collider_aabbs.append(None)   # boundary shells: full-grid fallback
        self.grid_postprocess.append(collide)
        self.modify_bc.append(None)

    # particle_v += force/particle_mass * dt
    # this is applied from start_dt, ends after num_dt p2g2p's
    # particle velocity is changed before p2g at each timestep
    def add_impulse_on_particles(self, force, dt, point =[1,1,1], size = [1,1,1], num_dt = 1, start_time=0.0, device = "cuda:0"):
        impulse_param = Impulse_modifier()
        impulse_param.start_time = start_time
        impulse_param.end_time = start_time + dt * num_dt

        impulse_param.point = wp.vec3(point[0], point[1], point[2])
        impulse_param.size = wp.vec3(size[0], size[1], size[2])
        impulse_param.mask = wp.zeros(
            shape=self.n_particles, dtype=int, device=device)

        impulse_param.force = wp.vec3(
            force[0],
            force[1],
            force[2],
        )

        wp.launch(
                kernel=selection_add_impulse_on_particles,
                dim=self.n_particles,
                inputs=[
                    self.mpm_state,
                    impulse_param
                ],
                device=device,
            )

        self.impulse_params.append(impulse_param)

        @wp.kernel
        def apply_force(
            time: float, dt: float, state: MPMStateStruct, param: Impulse_modifier
        ):
            p = wp.tid()
            if time >= param.start_time and time < param.end_time:
                if param.mask[p] == 1:
                    impulse = wp.vec3(
                        param.force[0] / state.particle_mass[p],
                        param.force[1] / state.particle_mass[p],
                        param.force[2] / state.particle_mass[p],
                    )
                    state.particle_v[p] = state.particle_v[p] + impulse * dt

        self.pre_p2g_operations.append(apply_force)



    def enforce_particle_velocity_translation(self, point, size, velocity, start_time, end_time, device = "cuda:0"):

        # first select certain particles based on position

        velocity_modifier_params = ParticleVelocityModifier()

        velocity_modifier_params.point = wp.vec3(point[0], point[1], point[2])
        velocity_modifier_params.size = wp.vec3(size[0], size[1], size[2])

        velocity_modifier_params.velocity = wp.vec3(velocity[0], velocity[1], velocity[2])

        velocity_modifier_params.start_time = start_time
        velocity_modifier_params.end_time = end_time

        velocity_modifier_params.mask = wp.zeros(
            shape=self.n_particles, dtype=int, device=device)
        
        wp.launch(
                kernel=selection_enforce_particle_velocity_translation,
                dim=self.n_particles,
                inputs=[
                    self.mpm_state,
                    velocity_modifier_params
                ],
                device=device,
            )
        self.particle_velocity_modifier_params.append(velocity_modifier_params)
        
        @wp.kernel
        def modify_particle_v_before_p2g(time: float,
                                        state: MPMStateStruct,
                                        velocity_modifier_params: ParticleVelocityModifier):
            p = wp.tid()
            if time >= velocity_modifier_params.start_time and time < velocity_modifier_params.end_time:
                if velocity_modifier_params.mask[p] == 1:
                    state.particle_v[p] = velocity_modifier_params.velocity

        
        self.particle_velocity_modifiers.append(modify_particle_v_before_p2g)


    # define a cylinder with center point, half_height, radius, normal
    # particles within the cylinder are rotating along the normal direction
    # may also have a translational velocity along the normal direction
    def enforce_particle_velocity_rotation(self, point, normal, 
                        half_height_and_radius, rotation_scale, translation_scale, start_time, end_time, device = "cuda:0"):

        normal_scale = 1.0 / wp.sqrt(float(normal[0]**2 + normal[1]**2 + normal[2]**2))
        normal = list(normal_scale * x for x in normal)

        velocity_modifier_params = ParticleVelocityModifier()

        velocity_modifier_params.point = wp.vec3(point[0], point[1], point[2])
        velocity_modifier_params.half_height_and_radius = wp.vec2(half_height_and_radius[0], half_height_and_radius[1])
        velocity_modifier_params.normal = wp.vec3(normal[0], normal[1], normal[2])

        horizontal_1 = wp.vec3(1.0,1.0,1.0)
        if wp.abs(wp.dot(velocity_modifier_params.normal, horizontal_1)) < 0.01:
            horizontal_1 = wp.vec3(0.72, 0.37, -0.67)
        horizontal_1 = horizontal_1 - wp.dot(horizontal_1, velocity_modifier_params.normal) * velocity_modifier_params.normal
        horizontal_1 = horizontal_1 * (1.0 / wp.length(horizontal_1))
        horizontal_2 = wp.cross(horizontal_1, velocity_modifier_params.normal)

        velocity_modifier_params.horizontal_axis_1 = horizontal_1
        velocity_modifier_params.horizontal_axis_2 = horizontal_2

        velocity_modifier_params.rotation_scale = rotation_scale
        velocity_modifier_params.translation_scale = translation_scale

        velocity_modifier_params.start_time = start_time
        velocity_modifier_params.end_time = end_time

        velocity_modifier_params.mask = wp.zeros(
            shape=self.n_particles, dtype=int, device=device)
        
        wp.launch(
                kernel=selection_enforce_particle_velocity_cylinder,
                dim=self.n_particles,
                inputs=[
                    self.mpm_state,
                    velocity_modifier_params
                ],
                device=device,
            )
        self.particle_velocity_modifier_params.append(velocity_modifier_params)
        
        @wp.kernel
        def modify_particle_v_before_p2g(time: float,
                                        state: MPMStateStruct,
                                        velocity_modifier_params: ParticleVelocityModifier):
            p = wp.tid()
            if time >= velocity_modifier_params.start_time and time < velocity_modifier_params.end_time:
                if velocity_modifier_params.mask[p] == 1:
                    offset = state.particle_x[p] - velocity_modifier_params.point
                    horizontal_distance = wp.length(offset - wp.dot(offset, velocity_modifier_params.normal) * velocity_modifier_params.normal)
                    cosine = wp.dot(offset, velocity_modifier_params.horizontal_axis_1) / horizontal_distance
                    theta = wp.acos(cosine)
                    if wp.dot(offset, velocity_modifier_params.horizontal_axis_2) > 0:
                        theta = theta
                    else:
                        theta = -theta
                    axis1_scale = - horizontal_distance * wp.sin(theta) * velocity_modifier_params.rotation_scale
                    axis2_scale = horizontal_distance * wp.cos(theta) * velocity_modifier_params.rotation_scale
                    axis_vertical_scale = translation_scale
                    state.particle_v[p] = axis1_scale * velocity_modifier_params.horizontal_axis_1 + axis2_scale * velocity_modifier_params.horizontal_axis_2 + axis_vertical_scale * velocity_modifier_params.normal 
                        

        
        self.particle_velocity_modifiers.append(modify_particle_v_before_p2g)
        

    # Add a point cloud as a static collider.
    # point_cloud_np: numpy array of shape (N, 3) with positions in world coordinates.
    # padding: number of grid cells to dilate around each occupied cell (fills gaps in sparse clouds).
    def add_point_cloud_collider(self, point_cloud_np, padding=1, start_time=0.0, end_time=999.0, device="cuda:0"):
        import numpy as np

        n_grid = self.mpm_model.n_grid
        inv_dx = self.mpm_model.inv_dx

        # Convert point positions to grid indices
        indices = (point_cloud_np * inv_dx).astype(np.int32)
        indices = np.clip(indices, 0, n_grid - 1)

        # Build occupancy grid
        occupancy = np.zeros((n_grid, n_grid, n_grid), dtype=np.int32)
        occupancy[indices[:, 0], indices[:, 1], indices[:, 2]] = 1

        # Dilate occupancy to fill gaps (without wrap-around)
        if padding > 0:
            for _ in range(padding):
                dilated = occupancy.copy()
                for axis in range(3):
                    slices_src = [slice(None)] * 3
                    slices_dst = [slice(None)] * 3
                    slices_src[axis] = slice(0, -1)
                    slices_dst[axis] = slice(1, None)
                    dilated[tuple(slices_dst)] |= occupancy[tuple(slices_src)]
                    slices_src = [slice(None)] * 3
                    slices_dst = [slice(None)] * 3
                    slices_src[axis] = slice(1, None)
                    slices_dst[axis] = slice(0, -1)
                    dilated[tuple(slices_dst)] |= occupancy[tuple(slices_src)]
                occupancy = dilated

        num_occupied = int(occupancy.sum())
        print(f"Point cloud collider: {len(point_cloud_np)} points -> {num_occupied} occupied grid cells (padding={padding})")

        collider_param = PointCloudCollider()
        collider_param.occupancy_grid = wp.from_numpy(occupancy, dtype=int, device=device)
        collider_param.start_time = start_time
        collider_param.end_time = end_time

        self.collider_params.append(collider_param)

        @wp.kernel
        def collide(
            time: float,
            dt: float,
            state: MPMStateStruct,
            model: MPMModelStruct,
            param: PointCloudCollider,
            lo: wp.vec3i,
        ):
            grid_x, grid_y, grid_z = wp.tid()
            grid_x = grid_x + lo[0]
            grid_y = grid_y + lo[1]
            grid_z = grid_z + lo[2]
            if time >= param.start_time and time < param.end_time:
                if param.occupancy_grid[grid_x, grid_y, grid_z] == 1:
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)

        _occ = np.argwhere(occupancy == 1)
        if len(_occ):
            _pc_box = (tuple(int(v) for v in _occ.min(0)),
                       tuple(int(_occ.max(0)[i] - _occ.min(0)[i] + 1) for i in range(3)))
        else:
            _pc_box = None
        self.collider_aabbs.append(lambda box=_pc_box: box)
        self.grid_postprocess.append(collide)
        self.modify_bc.append(None)

    # A watertight mesh as a moving/rotating signed-distance-field collider. The robot (or a
    # scripted pour) drives it via set_sdf_pose; the material is blocked from the solid and the
    # exact reaction wrench (force + torque) is read back from the grid impulse.
    def add_sdf_collider(self, sdf_values, sdf_grads, origin, cell, center,
                         quat=(0.0, 0.0, 0.0, 1.0), velocity=(0.0, 0.0, 0.0),
                         omega=(0.0, 0.0, 0.0), band=None, surface="separable",
                         friction=0.4, start_time=0.0, end_time=999.0, device="cpu"):
        res = int(sdf_values.shape[0])
        if band is None:
            band = float(self.mpm_model.dx)
        surface_id = {"sticky": 0, "slip": 1, "separable": 2}[surface]

        # Containment guard: the collide kernel only queries nodes whose body-frame position
        # falls inside the SDF grid box, so every point within `band` of the mesh surface must
        # lie inside that box. The minimum stored SDF value on the box's six faces IS the
        # worst-case surface-to-box-edge margin; if band reaches it, a shell of near-surface
        # space outside the box gets no constraint and the material can leak through unseen.
        vals_np = np.asarray(sdf_values)
        boundary_min = float(min(vals_np[0].min(), vals_np[-1].min(),
                                 vals_np[:, 0, :].min(), vals_np[:, -1, :].min(),
                                 vals_np[:, :, 0].min(), vals_np[:, :, -1].min()))
        if band >= boundary_min:
            raise ValueError(
                f"SDF collider contact band ({band * 1e3:.1f} mm) reaches or exceeds the SDF "
                f"grid margin ({boundary_min * 1e3:.1f} mm): near-surface space outside the "
                f"stored grid would get no collision. Rebuild the SDF with more margin_cells "
                f"or reduce band.")

        # Tunneling guard data: the largest body-frame lever arm from the pivot (body origin)
        # to any corner of the SDF box bounds the surface speed |v| + |omega| * r_max.
        ext = (res - 1) * float(cell)
        _corners = np.array([[float(origin[0]) + dx_ * ext,
                              float(origin[1]) + dy_ * ext,
                              float(origin[2]) + dz_ * ext]
                             for dx_ in (0, 1) for dy_ in (0, 1) for dz_ in (0, 1)])
        r_max = float(np.linalg.norm(_corners, axis=1).max())
        if not hasattr(self, "_sdf_guard"):
            self._sdf_guard = {}
        self._sdf_guard[len(self.collider_params)] = {"r_max": r_max, "band": float(band),
                                                      "warned": False}

        param = SDFCollider()
        param.sdf_val = wp.from_numpy(
            np.ascontiguousarray(sdf_values, dtype=np.float32), dtype=float, device=device)
        param.sdf_grad = wp.from_numpy(
            np.ascontiguousarray(sdf_grads, dtype=np.float32), dtype=wp.vec3, device=device)
        param.res = res
        param.origin = wp.vec3(float(origin[0]), float(origin[1]), float(origin[2]))
        param.cell = float(cell)
        param.center = wp.vec3(float(center[0]), float(center[1]), float(center[2]))
        param.quat = wp.quat(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        param.velocity = wp.vec3(float(velocity[0]), float(velocity[1]), float(velocity[2]))
        param.omega = wp.vec3(float(omega[0]), float(omega[1]), float(omega[2]))
        param.band = float(band)
        param.surface_type = surface_id
        param.friction = float(friction)
        param.start_time = start_time
        param.end_time = end_time
        param.force = wp.zeros(1, dtype=wp.vec3, device=device)
        param.torque = wp.zeros(1, dtype=wp.vec3, device=device)
        self.collider_params.append(param)

        @wp.kernel
        def collide(
            time: float,
            dt: float,
            state: MPMStateStruct,
            model: MPMModelStruct,
            param: SDFCollider,
            lo: wp.vec3i,
        ):
            gx, gy, gz = wp.tid()
            gx = gx + lo[0]
            gy = gy + lo[1]
            gz = gz + lo[2]
            if time >= param.start_time and time < param.end_time:
                xw = wp.vec3(float(gx) * model.dx, float(gy) * model.dx, float(gz) * model.dx)
                rel = xw - param.center
                body = wp.quat_rotate_inv(param.quat, rel)
                fidx = wp.vec3(
                    (body[0] - param.origin[0]) / param.cell,
                    (body[1] - param.origin[1]) / param.cell,
                    (body[2] - param.origin[2]) / param.cell,
                )
                rf = float(param.res) - 1.0
                inside_grid = (
                    fidx[0] >= 0.0 and fidx[1] >= 0.0 and fidx[2] >= 0.0
                    and fidx[0] <= rf and fidx[1] <= rf and fidx[2] <= rf
                )
                if inside_grid:
                    sd = sdf_trilerp(param.sdf_val, param.res, fidx)
                    if sd <= param.band:
                        g = sdf_trilerp_vec(param.sdf_grad, param.res, fidx)
                        if wp.length(g) > 1.0e-12:
                            n_world = wp.quat_rotate(param.quat, wp.normalize(g))
                            v_surf = param.velocity + wp.cross(param.omega, rel)
                            v_free = state.grid_v_out[gx, gy, gz]
                            v_rel = v_free - v_surf
                            vn = wp.dot(v_rel, n_world)
                            v_new = v_free
                            if param.surface_type == 0:        # sticky: move with the surface
                                v_new = v_surf
                            elif param.surface_type == 1:      # slip: kill normal relative vel
                                v_new = v_surf + (v_rel - vn * n_world)
                            else:                              # separable + Coulomb friction
                                if vn < 0.0:
                                    v_tan = v_rel - vn * n_world
                                    tlen = wp.length(v_tan)
                                    if tlen > 1.0e-12:
                                        scale = wp.max(0.0, tlen + param.friction * vn) / tlen
                                        v_tan = v_tan * scale
                                    v_new = v_surf + v_tan
                            m = state.grid_m[gx, gy, gz]
                            impulse = m * (v_free - v_new)
                            wp.atomic_add(param.force, 0, impulse)
                            wp.atomic_add(param.torque, 0, wp.cross(rel, impulse))
                            state.grid_v_out[gx, gy, gz] = v_new

        guard = self._sdf_guard[len(self.collider_params) - 1]

        def modify(time, dt, param: SDFCollider):
            if time >= param.start_time and time < param.end_time:
                c = param.center
                v = param.velocity
                w = np.array([param.omega[0], param.omega[1], param.omega[2]], dtype=float)
                # tunneling guard: per-substep surface sweep must stay inside the contact band
                # or a node can jump from outside the band to inside the solid unconstrained
                speed = float(np.hypot(np.hypot(v[0], v[1]), v[2])) \
                    + float(np.linalg.norm(w)) * guard["r_max"]
                if speed * dt > guard["band"] and not guard["warned"]:
                    guard["warned"] = True
                    warnings.warn(
                        f"SDF collider surface sweeps {speed * dt * 1e3:.2f} mm per substep, "
                        f"more than the contact band ({guard['band'] * 1e3:.2f} mm): fast "
                        f"material can tunnel through. Reduce dt, velocity/omega, or raise "
                        f"band (and the SDF margin).", RuntimeWarning, stacklevel=2)
                param.center = wp.vec3(c[0] + dt * v[0], c[1] + dt * v[1], c[2] + dt * v[2])
                q = np.array([param.quat[0], param.quat[1], param.quat[2], param.quat[3]],
                             dtype=float)
                q = _omega_step_quat(q, w, dt)
                param.quat = wp.quat(float(q[0]), float(q[1]), float(q[2]), float(q[3]))

        def sdf_aabb(param=param, corners=_corners, self=self):
            qx, qy, qz, qw = param.quat[0], param.quat[1], param.quat[2], param.quat[3]
            R = np.array([
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ])
            c = np.array([param.center[0], param.center[1], param.center[2]])
            pts = c + corners @ R.T
            return self._grid_box(pts.min(axis=0), pts.max(axis=0), halo=1)

        self.collider_aabbs.append(sdf_aabb)
        self.grid_postprocess.append(collide)
        self.modify_bc.append(modify)
        return len(self.collider_params) - 1

    def set_sdf_pose(self, handle, center=None, quat=None, velocity=None, omega=None):
        """Command an SDF collider with its START-of-tick pose and per-tick velocity/omega; the
        modify_bc integrates center += dt*velocity and rotates the quat by omega every substep,
        mirroring the box-collider contract (drive with v = (target - prev)/dt_ctrl). Setting
        center/quat directly teleports the collider between ticks; a jump whose surface sweep
        exceeds the contact band can tunnel through material, so it warns once."""
        p = self.collider_params[handle]
        guard = getattr(self, "_sdf_guard", {}).get(handle)
        if guard is not None and not guard["warned"]:
            jump = 0.0
            if center is not None:
                jump += float(np.linalg.norm(
                    np.array([float(center[0]) - p.center[0],
                              float(center[1]) - p.center[1],
                              float(center[2]) - p.center[2]])))
            if quat is not None:
                dot = abs(float(quat[0]) * p.quat[0] + float(quat[1]) * p.quat[1]
                          + float(quat[2]) * p.quat[2] + float(quat[3]) * p.quat[3])
                jump += 2.0 * float(np.arccos(min(1.0, dot))) * guard["r_max"]
            if jump > guard["band"]:
                guard["warned"] = True
                warnings.warn(
                    f"set_sdf_pose jumped the collider surface by up to {jump * 1e3:.2f} mm in "
                    f"one tick, more than the contact band ({guard['band'] * 1e3:.2f} mm): "
                    f"material inside the swept region can be tunnelled through. Drive the pose "
                    f"in smaller per-tick steps (or via velocity/omega).",
                    RuntimeWarning, stacklevel=2)
        if center is not None:
            p.center = wp.vec3(float(center[0]), float(center[1]), float(center[2]))
        if quat is not None:
            p.quat = wp.quat(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        if velocity is not None:
            p.velocity = wp.vec3(float(velocity[0]), float(velocity[1]), float(velocity[2]))
        if omega is not None:
            p.omega = wp.vec3(float(omega[0]), float(omega[1]), float(omega[2]))

    # given normal direction, say [0,0,1]
    # gradually release grid velocities from start position to end position
    def release_particles_sequentially(self, normal, start_position, end_position, num_layers, start_time, end_time):
        num_layers = 50
        point = [0,0,0]
        size = [0,0,0]
        axis = -1
        for i in range(3):
            if normal[i] == 0:
                point[i] = 1
                size[i] = 1
            else:
                axis = i
                point[i] = end_position
            
        half_length_portion = wp.abs(start_position - end_position)/num_layers
        end_time_portion = end_time / num_layers
        for i in range(num_layers):
            size[axis] = half_length_portion * (num_layers - i)
            self.enforce_particle_velocity_translation(point=point, size=size, velocity = [0,0,0], start_time=start_time, end_time=end_time_portion * (i+1))




        


