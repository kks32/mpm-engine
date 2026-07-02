import warp
import warp as wp
try:  # warp.torch was a separate submodule import in warp<=1.10; gone/needless in 1.14+
    import warp.torch  # noqa: F401
except Exception:
    pass
import torch


@wp.struct
class MPMModelStruct:
    ####### essential #######
    grid_lim: float
    n_particles: int
    n_grid: int
    dx: float
    inv_dx: float
    grid_dim_x: int
    grid_dim_y: int
    grid_dim_z: int
    mu: wp.array(dtype=float)
    lam: wp.array(dtype=float)
    E: wp.array(dtype=float)
    nu: wp.array(dtype=float)
    bulk: wp.array(dtype=float)
    material: int

    ######## for plasticity ####
    yield_stress: wp.array(dtype=float)
    friction_angle: float
    alpha: wp.array(dtype=float)
    gravitational_accelaration: wp.vec3
    hardening: wp.array(dtype=float)
    xi: wp.array(dtype=float)
    plastic_viscosity: wp.array(dtype=float)
    softening: wp.array(dtype=float)

    ######## local mu(I) rheology (material 9, TrackEUCLID), per-type ####
    muI_mu_s: wp.array(dtype=float)        # static friction coefficient
    muI_delta_mu: wp.array(dtype=float)    # mu_2 - mu_s (0 for constant mu)
    muI_I0: wp.array(dtype=float)          # Pouliquen reference inertial number
    muI_d: wp.array(dtype=float)           # grain diameter
    muI_rho_s: wp.array(dtype=float)       # grain density
    # compressible mu(I)-Phi(I) dilatancy (material 11), per-type
    muI_phi_init: wp.array(dtype=float)    # rest (densest) solid fraction Phi_init
    muI_phi_chi: wp.array(dtype=float)     # max dilatancy fraction: Phi_c(I)=Phi_init*(1-chi*I/(I+I0))

    ######## tabulated apparent viscosity (material 12, TrackEUCLID FE rollout) ####
    # eta_app(gd) read from a uniform log10(gd) table with clamped linear interp, so an
    # FE-recovered eta_app curve can be re-simulated directly (no parametric fit). One
    # global table (a single tabulated material per sim); s = log10(gd) in [smin, smax].
    eta_table: wp.array(dtype=float)       # (n,) eta_app samples on the log10(gd) grid
    eta_table_smin: float
    eta_table_smax: float
    eta_table_n: int

    ####### for damping
    rpic_damping: float
    grid_v_damping_scale: float

    ####### for PhysGaussian: covariance
    update_cov_with_F: int


@wp.struct
class MPMStateStruct:
    ###### essential #####
    # particle
    particle_x: wp.array(dtype=wp.vec3)  # current position
    particle_v: wp.array(dtype=wp.vec3)  # particle velocity
    particle_F: wp.array(dtype=wp.mat33)  # particle elastic deformation gradient
    particle_init_cov: wp.array(dtype=float)  # initial covariance matrix
    particle_cov: wp.array(dtype=float)  # current covariance matrix
    particle_F_trial: wp.array(
        dtype=wp.mat33
    )  # apply return mapping on this to obtain elastic def grad
    particle_R: wp.array(dtype=wp.mat33)  # rotation matrix
    particle_stress: wp.array(dtype=wp.mat33)  # Kirchoff stress, elastic stress
    particle_C: wp.array(dtype=wp.mat33)
    particle_vol: wp.array(dtype=float)  # current volume
    particle_mass: wp.array(dtype=float)  # mass
    particle_density: wp.array(dtype=float)  # density
    particle_Jp: wp.array(dtype=float)

    particle_L: wp.array(dtype=wp.mat33)  # velocity gradient from g2p, L_ij = dv_i/dx_j (TrackEUCLID dump)

    particle_selection: wp.array(dtype=int) # only particle_selection[p] = 0 will be simulated
    particle_material: wp.array(dtype=int)  # material type per particle (0=jelly,1=metal,2=sand,3=foam,4=snow,5=plasticine,6=fluid,7=stationary,8=rigid)
    particle_rigid_id: wp.array(dtype=int)  # rigid body id for mat==8 particles; -1 for non-rigid
    particle_x_ref: wp.array(dtype=wp.vec3) # body-frame reference position for rigid particles

    # grid
    grid_m: wp.array(dtype=float, ndim=3)
    grid_v_in: wp.array(dtype=wp.vec3, ndim=3)  # grid node momentum/velocity
    grid_v_out: wp.array(
        dtype=wp.vec3, ndim=3
    )  # grid node momentum/velocity, after grid update


# for various boundary conditions
@wp.struct
class Dirichlet_collider:
    point: wp.vec3
    normal: wp.vec3
    direction: wp.vec3

    start_time: float
    end_time: float

    friction: float
    surface_type: int

    velocity: wp.vec3

    threshold: float
    reset: int
    index: int

    x_unit: wp.vec3
    y_unit: wp.vec3
    radius: float
    v_scale: float
    width: float
    height: float
    length: float
    R: float

    size: wp.vec3

    horizontal_axis_1: wp.vec3
    horizontal_axis_2: wp.vec3
    half_height_and_radius: wp.vec2

    # optional Newton-exact reaction-impulse accumulator (set only by colliders that
    # measure force, e.g. set_velocity_on_cuboid); harmless/unused for the others
    force: wp.array(dtype=wp.vec3)

@wp.struct
class PointCloudCollider:
    occupancy_grid: wp.array(dtype=int, ndim=3)
    start_time: float
    end_time: float


@wp.struct
class SDFCollider:
    # a watertight mesh represented as a stored signed-distance field in its body frame:
    # negative inside the solid, positive outside. The collider may translate and rotate; the
    # field is queried by mapping each grid node into the body frame and trilinearly
    # interpolating sdf_val (distance) and sdf_grad (outward gradient).
    sdf_val: wp.array(dtype=float, ndim=3)
    sdf_grad: wp.array(dtype=wp.vec3, ndim=3)
    res: int
    origin: wp.vec3          # body-frame coordinate of voxel index (0,0,0)
    cell: float              # body-frame metres per voxel (isotropic)

    center: wp.vec3          # world position of the body-frame origin (pivot)
    quat: wp.quat            # orientation body -> world
    velocity: wp.vec3        # linear velocity of the pivot (world)
    omega: wp.vec3           # angular velocity (world)

    band: float              # contact band thickness (world metres); constrain nodes with sd < band
    surface_type: int        # 0 sticky, 1 slip (frictionless), 2 separable + Coulomb friction
    friction: float

    start_time: float
    end_time: float

    # Newton-exact reaction accumulators (impulse the material delivers to the collider, world
    # frame): force = sum_nodes m*(v_free - v_new); torque = sum_nodes (x - center) x impulse.
    # Reaction wrench = (force, torque) / elapsed dt.
    force: wp.array(dtype=wp.vec3)
    torque: wp.array(dtype=wp.vec3)



@wp.struct
class Impulse_modifier:
    # this needs to be changed for each different BC!
    point: wp.vec3
    normal: wp.vec3
    start_time: float
    end_time: float
    force: wp.vec3
    forceTimesDt: wp.vec3
    numsteps: int

    point: wp.vec3
    size: wp.vec3
    mask: wp.array(dtype=int)


@wp.struct
class MPMtailoredStruct:
    # this needs to be changed for each different BC!
    point: wp.vec3
    normal: wp.vec3
    start_time: float
    end_time: float
    friction: float
    surface_type: int
    velocity: wp.vec3
    threshold: float
    reset: int

    point_rotate: wp.vec3
    normal_rotate: wp.vec3
    x_unit: wp.vec3
    y_unit: wp.vec3
    radius: float
    v_scale: float
    width: float
    point_plane: wp.vec3
    normal_plane: wp.vec3
    velocity_plane: wp.vec3
    threshold_plane: float

@wp.struct
class MaterialParamsModifier:
    point: wp.vec3
    size: wp.vec3
    E: float
    nu: float
    density: float

@wp.struct
class ParticleVelocityModifier:
    point: wp.vec3
    normal: wp.vec3
    half_height_and_radius: wp.vec2
    rotation_scale: float
    translation_scale: float

    size: wp.vec3

    horizontal_axis_1: wp.vec3
    horizontal_axis_2: wp.vec3
    
    start_time: float

    end_time: float

    velocity: wp.vec3

    mask: wp.array(dtype=int)




@wp.kernel
def set_vec3_to_zero(target_array: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    target_array[tid] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def set_mat33_to_identity(target_array: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    target_array[tid] = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


@wp.kernel
def add_identity_to_mat33(target_array: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    target_array[tid] = wp.add(
        target_array[tid], wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    )


@wp.kernel
def subtract_identity_to_mat33(target_array: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    target_array[tid] = wp.sub(
        target_array[tid], wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    )


@wp.kernel
def add_vec3_to_vec3(
    first_array: wp.array(dtype=wp.vec3), second_array: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()
    first_array[tid] = wp.add(first_array[tid], second_array[tid])


@wp.kernel
def set_value_to_float_array(target_array: wp.array(dtype=float), value: float):
    tid = wp.tid()
    target_array[tid] = value


@wp.kernel
def set_value_to_int_array(target_array: wp.array(dtype=int), value: int):
    tid = wp.tid()
    target_array[tid] = value


@wp.kernel
def get_float_array_product(
    arrayA: wp.array(dtype=float),
    arrayB: wp.array(dtype=float),
    arrayC: wp.array(dtype=float),
):
    tid = wp.tid()
    arrayC[tid] = arrayA[tid] * arrayB[tid]


def _torch_on_device(t, dvc):
    dev = torch.device(str(dvc))
    if t.device != dev:
        t = t.to(dev)
    if not t.is_contiguous():
        t = t.contiguous()
    return t


def torch2warp_quat(t, copy=False, dtype=wp.float32, dvc="cuda:0"):
    t = _torch_on_device(t, dvc)
    if t.dtype != torch.float32 and t.dtype != torch.int32:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
        )
    assert t.shape[1] == 4
    a = wp.array(
        ptr=t.data_ptr(),
        dtype=wp.quat,
        shape=t.shape[0],
        copy=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc,
    )
    a.tensor = t
    return a

def torch2warp_int(t, copy=False, dtype=wp.int32, dvc="cuda:0"):
    t = _torch_on_device(t, dvc)
    if t.dtype != torch.float32 and t.dtype != torch.int32:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
        )
    a = wp.array(
        ptr=t.data_ptr(),
        dtype=wp.int32,
        shape=t.shape[0],
        copy=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc,
    )
    a.tensor = t
    return a

def torch2warp_float(t, copy=False, dtype=wp.float32, dvc="cuda:0"):
    t = _torch_on_device(t, dvc)
    if t.dtype != torch.float32 and t.dtype != torch.int32:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
        )
    a = wp.array(
        ptr=t.data_ptr(),
        dtype=wp.float32,
        shape=t.shape[0],
        copy=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc,
    )
    a.tensor = t
    return a

def torch2warp_vec3(t, copy=False, dtype=wp.float32, dvc="cuda:0"):
    t = _torch_on_device(t, dvc)
    if t.dtype != torch.float32 and t.dtype != torch.int32:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
        )
    assert t.shape[1] == 3
    a = wp.array(
        ptr=t.data_ptr(),
        dtype=wp.vec3,
        shape=t.shape[0],
        copy=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc,
    )
    a.tensor = t
    return a


def torch2warp_mat33(t, copy=False, dtype=wp.float32, dvc="cuda:0"):
    t = _torch_on_device(t, dvc)
    if t.dtype != torch.float32 and t.dtype != torch.int32:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
        )
    assert t.shape[1] == 3
    a = wp.array(
        ptr=t.data_ptr(),
        dtype=wp.mat33,
        shape=t.shape[0],
        copy=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc,
    )
    a.tensor = t
    return a
