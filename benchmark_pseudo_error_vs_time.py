from pathlib import Path

import basix.ufl
import numpy as np
import pandas as pd
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, geometry, mesh
from dolfinx.fem.petsc import assemble_matrix, assemble_vector

# ===== Function space / state helper functions ===============
def build_state_space(domain, degree=1):
    element = basix.ufl.element("DG", domain.basix_cell(), degree, shape=(3,))
    return fem.functionspace(domain, element)

def split_state(U):
    return U[0], U[1], U[2]


# ===== Physical SWE fluxes ===================================
def flux_x(U, gravity, h_floor):
    h, hu, hv = split_state(U)
    h_safe = ufl.max_value(h, h_floor)
    u = hu / h_safe
    return ufl.as_vector(
        [
            hu,
            hu * u + 0.5 * gravity * h * h,
            hv * u,
        ]
    )

def flux_y(U, gravity, h_floor):
    h, hu, hv = split_state(U)
    h_safe = ufl.max_value(h, h_floor)
    v = hv / h_safe
    return ufl.as_vector(
        [
            hv,
            hu * v,
            hv * v + 0.5 * gravity * h * h,
        ]
    )

def normal_flux(U, normal, gravity, h_floor):
    return flux_x(U, gravity, h_floor) * normal[0] + flux_y(U, gravity, h_floor) * normal[1]

def max_wave_speed(U, normal, gravity, h_floor):
    h, hu, hv = split_state(U)
    h_safe = ufl.max_value(h, h_floor)
    un = (hu * normal[0] + hv * normal[1]) / h_safe
    c = ufl.sqrt(gravity * h_safe)
    return abs(un) + c

def lax_friedrichs_flux(U_minus, U_plus, normal, gravity, h_floor):
    flux_avg = 0.5 * (
        normal_flux(U_minus, normal, gravity, h_floor)
        + normal_flux(U_plus, normal, gravity, h_floor)
    )
    lambda_max = ufl.max_value(
        max_wave_speed(U_minus, normal, gravity, h_floor),
        max_wave_speed(U_plus, normal, gravity, h_floor),
    )
    return flux_avg - 0.5 * lambda_max * (U_plus - U_minus)


# ===== Boundary/Reflective state =============================
def reflective_exterior_state(U, normal):
    h, hu, hv = split_state(U)
    momentum = ufl.as_vector([hu, hv])
    normal_momentum = ufl.dot(momentum, normal)
    reflected_momentum = momentum - 2.0 * normal_momentum * normal
    return ufl.as_vector([h, reflected_momentum[0], reflected_momentum[1]])


# DG weak form
def build_rhs_form(domain, V, U_state, zb, gravity, h_floor):
    W = ufl.TestFunction(V)
    n = ufl.FacetNormal(domain)

    fx = flux_x(U_state, gravity, h_floor)
    fy = flux_y(U_state, gravity, h_floor)
    physical_flux = ufl.as_tensor(
        [
            [fx[0], fy[0]],
            [fx[1], fy[1]],
            [fx[2], fy[2]],
        ]
    )

    interior_flux = lax_friedrichs_flux(U_state("-"), U_state("+"), n("-"), gravity, h_floor)
    exterior_state = reflective_exterior_state(U_state, n)
    boundary_flux = lax_friedrichs_flux(U_state, exterior_state, n, gravity, h_floor)

    source = ufl.as_vector(
        [
            0.0,
            -gravity * U_state[0] * zb.dx(0),
            -gravity * U_state[0] * zb.dx(1),
        ]
    )

    return (
        ufl.inner(physical_flux, ufl.grad(W)) * ufl.dx
        - ufl.inner(interior_flux, W("-") - W("+")) * ufl.dS
        - ufl.inner(boundary_flux, W) * ufl.ds
        + ufl.inner(source, W) * ufl.dx
    )


# ===== Time stepping =========================================
def compute_time_step(U, h_dofs, hu_dofs, hv_dofs, gravity, cfl, h_min, comm):
    h_vals = U.x.array[h_dofs]
    hu_vals = U.x.array[hu_dofs]
    hv_vals = U.x.array[hv_dofs]
    h_safe = np.maximum(h_vals, 1.0e-8)
    speed = np.sqrt((hu_vals / h_safe) ** 2 + (hv_vals / h_safe) ** 2) + np.sqrt(gravity * h_safe)
    local_speed = float(np.max(speed)) if speed.size else 0.0
    max_speed = comm.allreduce(local_speed, op=MPI.MAX)
    if max_speed <= 1.0e-12:
        return 1.0e-3
    return cfl * h_min / max_speed

def solve_stage(ksp, rhs_form, stage_dt, U_eval, U_in, U_out, rhs_vec):
    U_eval.x.scatter_forward()
    rhs_vec.set(0.0)
    assemble_vector(rhs_vec, rhs_form)
    rhs_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    rhs_vec.scale(stage_dt)
    ksp.solve(rhs_vec, U_out.x.petsc_vec)
    U_out.x.scatter_forward()
    U_out.x.array[:] += U_in.x.array


# ===== Evaluation on arbitrary points =========================
def evaluate_on_points(domain, function, points_xy):
    points = np.zeros((3, len(points_xy)), dtype=np.float64)
    points[0, :] = points_xy[:, 0]
    points[1, :] = points_xy[:, 1]

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    values = np.full(len(points_xy), np.nan, dtype=np.float64)
    local_points = []
    local_cells = []
    local_ids = []

    for i, p in enumerate(points.T):
        cells = colliding_cells.links(i)
        if len(cells) > 0:
            local_points.append(p)
            local_cells.append(cells[0])
            local_ids.append(i)

    if local_points:
        local_points = np.array(local_points, dtype=np.float64)
        evaluated = function.eval(local_points, local_cells)
        values[np.array(local_ids, dtype=np.int32)] = evaluated[:, 0]

    gathered = domain.comm.gather(values, root=0)
    if domain.comm.rank == 0:
        merged = np.full(len(points_xy), np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            merged[mask] = arr[mask]
        return merged
    return None


def compute_depth_errors(domain, U_n, Q_out, x_ref, h_ref, Ly):
    h_fun = fem.Function(Q_out)
    h_expr = fem.Expression(U_n.sub(0), Q_out.element.interpolation_points)
    h_fun.interpolate(h_expr)
    h_fun.x.scatter_forward()

    points_xy = np.column_stack([x_ref, np.full_like(x_ref, 0.5 * Ly)])
    h_num = evaluate_on_points(domain, h_fun, points_xy)

    if domain.comm.rank == 0:
        err = h_num - h_ref
        l1 = np.mean(np.abs(err))
        l2 = np.sqrt(np.mean(err**2))
        linf = np.max(np.abs(err))
        return l1, l2, linf
    return None, None, None


def main():
    comm = MPI.COMM_WORLD
    Path("output").mkdir(exist_ok=True)

    df = pd.read_csv("/home/c_a00/SWE_flooding_project/swashes_pseudo2d_subcritical.csv", index_col=0) # PseudoTwoDimensional(1, 1, 1, 400)
    x_ref = df.index.to_numpy(dtype=float)
    h_ref = df["depth"].to_numpy(dtype=float)
    zb_ref = df["gd_elev"].to_numpy(dtype=float)

    Lx = 200
    Ly = 1.0
    nx = 400
    ny = 1

    final_time = 10
    cfl = 0.08
    gravity = 9.81
    h_floor = 1.0e-6
    degree = 1
    save_every = 10

    domain = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0]), np.array([Lx, Ly])],
        [nx, ny],
    )
    V = build_state_space(domain, degree=degree)

    times = []
    l1_errors = []
    l2_errors = []
    linf_errors = []

    U_n = fem.Function(V, name="U")
    U_0 = fem.Function(V, name="U_old")
    U_1 = fem.Function(V, name="U_stage_1")
    U_2 = fem.Function(V, name="U_stage_2")
    U_eval = fem.Function(V, name="U_eval")

    Q_h, h_dofs = V.sub(0).collapse()
    Q_hu, hu_dofs = V.sub(1).collapse()
    Q_hv, hv_dofs = V.sub(2).collapse()

    h_init = fem.Function(Q_h)
    hu_init = fem.Function(Q_hu)
    hv_init = fem.Function(Q_hv)

    def initial_depth(x):
        values = np.empty((1, x.shape[1]), dtype=PETSc.ScalarType)
        values[0] = np.interp(x[0], x_ref, h_ref)
        return values

    def zero_field(x):
        return np.zeros((1, x.shape[1]), dtype=PETSc.ScalarType)

    h_init.interpolate(initial_depth)
    hu_init.interpolate(zero_field)
    hv_init.interpolate(zero_field)

    U_n.x.array[h_dofs] = h_init.x.array
    U_n.x.array[hu_dofs] = hu_init.x.array
    U_n.x.array[hv_dofs] = hv_init.x.array
    U_n.x.scatter_forward()

    Q_b = fem.functionspace(domain, ("DG", degree))
    zb = fem.Function(Q_b, name="bathymetry")

    def bathymetry_profile(x):
        values = np.empty((1, x.shape[1]), dtype=PETSc.ScalarType)
        values[0] = np.interp(x[0], x_ref, zb_ref)
        return values

    zb.interpolate(bathymetry_profile)
    zb.x.scatter_forward()

    trial = ufl.TrialFunction(V)
    test = ufl.TestFunction(V)
    mass_form = fem.form(ufl.inner(trial, test) * ufl.dx)
    mass_matrix = assemble_matrix(mass_form)
    mass_matrix.assemble()

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(mass_matrix)
    ksp.setType("preonly")
    ksp.getPC().setType("bjacobi")

    rhs_form = fem.form(build_rhs_form(domain, V, U_eval, zb, gravity, h_floor))
    rhs_vec = assemble_vector(rhs_form)

    Q_out = fem.functionspace(domain, ("DG", degree))
    h_out = fem.Function(Q_out, name="depth")

    def update_depth_output():
        h_expr = fem.Expression(U_n.sub(0), Q_out.element.interpolation_points)
        h_out.interpolate(h_expr)
        h_out.x.scatter_forward()

    update_depth_output()

    t = 0.0
    step = 0
    h_min = min(Lx / nx, Ly / ny)

    while t < final_time - 1.0e-12:
        dt = compute_time_step(U_n, h_dofs, hu_dofs, hv_dofs, gravity, cfl, h_min, comm)
        dt = min(dt, final_time - t)

        U_0.x.array[:] = U_n.x.array

        U_eval.x.array[:] = U_0.x.array
        solve_stage(ksp, rhs_form, dt, U_eval, U_0, U_1, rhs_vec)

        U_eval.x.array[:] = U_1.x.array
        solve_stage(ksp, rhs_form, dt, U_eval, U_1, U_2, rhs_vec)
        U_2.x.array[:] = 0.75 * U_0.x.array + 0.25 * U_2.x.array

        U_eval.x.array[:] = U_2.x.array
        solve_stage(ksp, rhs_form, dt, U_eval, U_2, U_n, rhs_vec)
        U_n.x.array[:] = (1.0 / 3.0) * U_0.x.array + (2.0 / 3.0) * U_n.x.array

        U_n.x.array[h_dofs] = np.maximum(U_n.x.array[h_dofs], h_floor)
        U_n.x.scatter_forward()

        t += dt
        step += 1

        if step % save_every == 0 or t >= final_time - 1.0e-12:
            update_depth_output()

            l1, l2, linf = compute_depth_errors(domain, U_n, Q_out, x_ref, h_ref, Ly)
            if comm.rank == 0:
                times.append(t)
                l1_errors.append(l1)
                l2_errors.append(l2)
                linf_errors.append(linf)
                h_vals = U_n.x.array[h_dofs]
                print(
                    f"step={step:05d} t={t:.6f} dt={dt:.3e} "
                    f"h_min={h_vals.min():.6f} h_max={h_vals.max():.6f} "
                    f"L1={l1:.6e} L2={l2:.6e} Linf={linf:.6e}"
                )
    
    if comm.rank == 0:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.plot(times, l1_errors, label="L1 error")
        plt.plot(times, l2_errors, label="L2 error")
        plt.plot(times, linf_errors, label="Linf error")
        plt.yscale("log")
        plt.xlabel("Time [s]")
        plt.ylabel("Error")
        plt.title("Pseudo-2D Error Norms vs Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("output/error_vs_time.png", dpi=200)
        plt.show()



if __name__ == "__main__":
    main()
