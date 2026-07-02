from pathlib import Path

import basix.ufl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from scipy.interpolate import griddata

from dolfinx import fem, geometry, mesh
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from dolfinx.io import VTXWriter


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


# ===== TVB slope limiter helper functions ====================
def minmod(values):
    if np.all(values > 0.0):
        return np.min(values)
    if np.all(values < 0.0):
        return np.max(values)
    return 0.0

def tvb_minmod(a0, a1, M, h):
    if abs(a0) <= M * h * h:
        return a0
    return minmod(np.array([a0, a1], dtype=float))

def fit_cell_gradient(points, values):
    vandermonde = np.column_stack([np.ones(points.shape[0]), points])
    coeffs, _, rank, _ = np.linalg.lstsq(vandermonde, values, rcond=None) # More robust than solve, returns zero gradient if singular cell
    if rank < vandermonde.shape[1]:
        return np.zeros(points.shape[1])
    return coeffs[1:]

def fit_neighbor_gradient(center, neighbors, mean_value, cell_means, centers, gdim):
    if neighbors.size < gdim:
        return np.zeros(gdim)

    displacements = centers[neighbors] - center
    mean_differences = cell_means[neighbors] - mean_value
    gradient, *_ = np.linalg.lstsq(displacements, mean_differences, rcond=None)
    return gradient

def build_tvb_limiter_data(domain, scalar_space):
    tdim = domain.topology.dim
    fdim = tdim - 1

    domain.topology.create_connectivity(tdim, fdim)
    domain.topology.create_connectivity(fdim, tdim)
    cell_map = domain.topology.index_map(tdim)
    num_cells = cell_map.size_local + cell_map.num_ghosts

    dof_coords = scalar_space.tabulate_dof_coordinates().reshape((-1, domain.geometry.dim))
    cell_dofs = scalar_space.dofmap.list
    num_local_dofs = cell_dofs.shape[1]
    if num_local_dofs != 3:
        raise NotImplementedError("The TVB limiter below is implemented for DG(1) triangles only.")

    cell_centers = np.zeros((num_cells, domain.geometry.dim))
    edge_midpoints = np.zeros((num_cells, 3, domain.geometry.dim))
    cell_neighbors = []

    c_to_f = domain.topology.connectivity(tdim, fdim)
    f_to_c = domain.topology.connectivity(fdim, tdim)

    for cell in range(num_cells):
        local_dofs = cell_dofs[cell]
        points = dof_coords[local_dofs]
        cell_centers[cell] = np.mean(points, axis=0)

        edge_midpoints[cell, 0] = 0.5 * (points[1] + points[2])
        edge_midpoints[cell, 1] = 0.5 * (points[2] + points[0])
        edge_midpoints[cell, 2] = 0.5 * (points[0] + points[1])

        neighbors = []
        for facet in c_to_f.links(cell):
            attached_cells = f_to_c.links(facet)
            for other in attached_cells:
                if other != cell and other not in neighbors:
                    neighbors.append(other)
        cell_neighbors.append(np.array(neighbors, dtype=np.int32))

    return {
        "cell_dofs": cell_dofs,
        "dof_coords": dof_coords,
        "cell_centers": cell_centers,
        "edge_midpoints": edge_midpoints,
        "cell_neighbors": cell_neighbors,
        "gdim": domain.geometry.dim,
    }

def limit_scalar_component(component_array, limiter_data, tvb_M, tvb_nu, cell_size, limit_cells=None):
    cell_dofs = limiter_data["cell_dofs"]
    dof_coords = limiter_data["dof_coords"]
    cell_centers = limiter_data["cell_centers"]
    edge_midpoints = limiter_data["edge_midpoints"]
    cell_neighbors = limiter_data["cell_neighbors"]
    gdim = limiter_data["gdim"]

    cell_values = component_array[cell_dofs]
    cell_means = np.mean(cell_values, axis=1)
    limited_values = cell_values.copy()

    for cell, local_dofs in enumerate(cell_dofs):
        if limit_cells is not None and not limit_cells[cell]:
            continue

        local_points = dof_coords[local_dofs]
        local_values = cell_values[cell]
        mean_value = cell_means[cell]
        center = cell_centers[cell]

        gradient = fit_cell_gradient(local_points, local_values)
        midpoint_offsets = edge_midpoints[cell] - center
        midpoint_deviation = midpoint_offsets @ gradient

        neighbor_gradient = fit_neighbor_gradient(
            center, cell_neighbors[cell], mean_value, cell_means, cell_centers, gdim
        )
        estimated_deviation = midpoint_offsets @ neighbor_gradient

        limited_midpoint_deviation = np.array(
            [
                tvb_minmod(midpoint_deviation[i], tvb_nu * estimated_deviation[i], tvb_M, cell_size)
                for i in range(3)
            ]
        )

        if abs(np.sum(limited_midpoint_deviation)) > 1.0e-14:
            positive = np.sum(np.maximum(limited_midpoint_deviation, 0.0))
            negative = -np.sum(np.minimum(limited_midpoint_deviation, 0.0))
            theta_pos = min(1.0, negative / positive) if positive > 0.0 else 1.0
            theta_neg = min(1.0, positive / negative) if negative > 0.0 else 1.0
            limited_midpoint_deviation = (
                theta_pos * np.maximum(limited_midpoint_deviation, 0.0)
                + theta_neg * np.minimum(limited_midpoint_deviation, 0.0)
            )

        limited_gradient, *_ = np.linalg.lstsq(midpoint_offsets, limited_midpoint_deviation, rcond=None)
        limited_values[cell] = mean_value + (local_points - center) @ limited_gradient

    component_array[cell_dofs.reshape(-1)] = limited_values.reshape(-1)

def apply_tvb_limiter(U, limiter_data, component_dofs, tvb_M, tvb_nu, cell_size, limit_cells=None):
    for dofs in component_dofs:
        component_values = U.x.array[dofs].copy()
        limit_scalar_component(
            component_values, limiter_data, tvb_M, tvb_nu, cell_size, limit_cells=limit_cells
        )
        U.x.array[dofs] = component_values
    U.x.scatter_forward()
    return U


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


def compute_depth_errors_2d(domain, U_n, Q_out, points_xy, h_ref):
    h_fun = fem.Function(Q_out)
    h_expr = fem.Expression(U_n.sub(0), Q_out.element.interpolation_points)
    h_fun.interpolate(h_expr)
    h_fun.x.scatter_forward()

    h_num = evaluate_on_points(domain, h_fun, points_xy)
    if domain.comm.rank == 0:
        err = h_num - h_ref
        l1 = np.mean(np.abs(err))
        l2 = np.sqrt(np.mean(err**2))
        linf = np.max(np.abs(err))
        return l1, l2, linf
    return None, None, None


def load_reference_data(csv_path):
    df = pd.read_csv(csv_path)
    x_ref = df["x"].to_numpy(dtype=float)
    y_ref = df["y"].to_numpy(dtype=float)
    h_ref = df["depth"].to_numpy(dtype=float)
    u_ref = df["u"].to_numpy(dtype=float)
    v_ref = df["v"].to_numpy(dtype=float)
    zb_ref = df["gd_elev"].to_numpy(dtype=float)

    xy_ref = np.column_stack([x_ref, y_ref])
    xmin, xmax = float(np.min(x_ref)), float(np.max(x_ref))
    ymin, ymax = float(np.min(y_ref)), float(np.max(y_ref))

    return {
        "x_ref": x_ref,
        "y_ref": y_ref,
        "h_ref": h_ref,
        "u_ref": u_ref,
        "v_ref": v_ref,
        "zb_ref": zb_ref,
        "xy_ref": xy_ref,
        "points_xy": xy_ref.copy(),
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
    }


def run_benchmark(reference, nx, ny, final_time, degree=1, cfl=None, fixed_dt=None, save_output=False):
    comm = MPI.COMM_WORLD
    gravity = 9.81
    h_floor = 1.0e-6

    x_ref = reference["x_ref"]
    y_ref = reference["y_ref"]
    h_ref = reference["h_ref"]
    u_ref = reference["u_ref"]
    v_ref = reference["v_ref"]
    zb_ref = reference["zb_ref"]
    xy_ref = reference["xy_ref"]
    points_xy = reference["points_xy"]
    xmin = reference["xmin"]
    xmax = reference["xmax"]
    ymin = reference["ymin"]
    ymax = reference["ymax"]

    domain = mesh.create_rectangle(
        comm,
        [np.array([xmin, ymin]), np.array([xmax, ymax])],
        [nx, ny],
    )
    V = build_state_space(domain, degree=degree)

    U_n = fem.Function(V, name="U")
    U_0 = fem.Function(V, name="U_old")
    U_1 = fem.Function(V, name="U_stage_1")
    U_2 = fem.Function(V, name="U_stage_2")
    U_eval = fem.Function(V, name="U_eval")

    Q_h, h_dofs = V.sub(0).collapse()
    Q_hu, hu_dofs = V.sub(1).collapse()
    Q_hv, hv_dofs = V.sub(2).collapse()

    limiter_data = build_tvb_limiter_data(domain, Q_h)
    component_dofs = [h_dofs, hu_dofs, hv_dofs]
    tvb_M = 0.5
    tvb_nu = 1.5
    limit_cells = None

    h_init = fem.Function(Q_h)
    hu_init = fem.Function(Q_hu)
    hv_init = fem.Function(Q_hv)

    def interp2d(xq, yq, values):
        pts = np.column_stack([xq, yq])
        vals = griddata(xy_ref, values, pts, method="linear")
        if np.any(np.isnan(vals)):
            vals_nn = griddata(xy_ref, values, pts[np.isnan(vals)], method="nearest")
            vals[np.isnan(vals)] = vals_nn
        return vals

    def initial_depth(x):
        values = np.empty((1, x.shape[1]), dtype=PETSc.ScalarType)
        values[0] = np.maximum(interp2d(x[0], x[1], h_ref), 0.0)
        return values

    def initial_hu(x):
        values = np.empty((1, x.shape[1]), dtype=PETSc.ScalarType)
        values[0] = interp2d(x[0], x[1], h_ref) * interp2d(x[0], x[1], u_ref)
        return values

    def initial_hv(x):
        values = np.empty((1, x.shape[1]), dtype=PETSc.ScalarType)
        values[0] = interp2d(x[0], x[1], h_ref) * interp2d(x[0], x[1], v_ref)
        return values

    h_init.interpolate(initial_depth)
    hu_init.interpolate(initial_hu)
    hv_init.interpolate(initial_hv)

    U_n.x.array[h_dofs] = h_init.x.array
    U_n.x.array[hu_dofs] = hu_init.x.array
    U_n.x.array[hv_dofs] = hv_init.x.array
    U_n.x.scatter_forward()

    Q_b = fem.functionspace(domain, ("DG", degree))
    zb = fem.Function(Q_b, name="bathymetry")

    def bathymetry_profile(x):
        values = np.empty((1, x.shape[1]), dtype=PETSc.ScalarType)
        values[0] = interp2d(x[0], x[1], zb_ref)
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

    writer = None
    if save_output:
        Path("output").mkdir(exist_ok=True)
        update_depth_output()
        writer = VTXWriter(comm, f"output/thacker_{nx}x{ny}.bp", [h_out], engine="BP4")
        writer.write(0.0)

    h_mesh = min((xmax - xmin) / nx, (ymax - ymin) / ny)
    t = 0.0
    last_dt = None

    while t < final_time - 1.0e-12:
        if fixed_dt is not None:
            dt = min(fixed_dt, final_time - t)
        else:
            dt = compute_time_step(U_n, h_dofs, hu_dofs, hv_dofs, gravity, cfl, h_mesh, comm)
            dt = min(dt, final_time - t)

        last_dt = dt

        U_0.x.array[:] = U_n.x.array

        U_eval.x.array[:] = U_0.x.array
        solve_stage(ksp, rhs_form, dt, U_eval, U_0, U_1, rhs_vec)
        #apply_tvb_limiter(U_1, limiter_data, component_dofs, tvb_M, tvb_nu, h_mesh, limit_cells=limit_cells)

        U_eval.x.array[:] = U_1.x.array
        solve_stage(ksp, rhs_form, dt, U_eval, U_1, U_2, rhs_vec)
        U_2.x.array[:] = 0.75 * U_0.x.array + 0.25 * U_2.x.array
        #apply_tvb_limiter(U_2, limiter_data, component_dofs, tvb_M, tvb_nu, h_mesh, limit_cells=limit_cells)

        U_eval.x.array[:] = U_2.x.array
        solve_stage(ksp, rhs_form, dt, U_eval, U_2, U_n, rhs_vec)
        U_n.x.array[:] = (1.0 / 3.0) * U_0.x.array + (2.0 / 3.0) * U_n.x.array
        #apply_tvb_limiter(U_n, limiter_data, component_dofs, tvb_M, tvb_nu, h_mesh, limit_cells=limit_cells)

        U_n.x.array[h_dofs] = np.maximum(U_n.x.array[h_dofs], h_floor)
        U_n.x.scatter_forward()

        t += dt

    l1, l2, linf = compute_depth_errors_2d(domain, U_n, Q_out, points_xy, h_ref)

    if writer is not None:
        update_depth_output()
        writer.write(t)
        writer.close()

    return h_mesh, last_dt, l1, l2, linf

def plot_convergence(x, l1, l2, linf, xlabel, title, filename):
    plt.figure(figsize=(8, 5))
    plt.loglog(x, l1, "o-", label="L1")
    plt.loglog(x, l2, "s-", label="L2")
    plt.loglog(x, linf, "^-", label="Linf")
    plt.xlabel(xlabel)
    plt.ylabel("Error")
    plt.title(title)
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()

def grid_convergence_study(reference):
    mesh_sizes = [25, 50, 100]
    hs, l1s, l2s, linfs = [], [], [], []

    for n in mesh_sizes:
        h, dt, l1, l2, linf = run_benchmark(
            reference,
            nx=n,
            ny=n,
            final_time=0.2,
            cfl=0.08,
            fixed_dt=None,
            save_output=False,
        )
        hs.append(h)
        l1s.append(l1)
        l2s.append(l2)
        linfs.append(linf)

        if MPI.COMM_WORLD.rank == 0:
            print(f"grid study n={n}: h={h:.4e}, dt={dt:.4e}, L1={l1:.4e}, L2={l2:.4e}, Linf={linf:.4e}")

    return np.array(hs), np.array(l1s), np.array(l2s), np.array(linfs)

def main():
    Path("output").mkdir(exist_ok=True)
    reference = load_reference_data("/home/c_a00/SWE_flooding_project/swashes_2d_thacker_radial.csv")

    hs, l1_h, l2_h, linf_h = grid_convergence_study(reference)
    if MPI.COMM_WORLD.rank == 0:
        plot_convergence(
            hs,
            l1_h,
            l2_h,
            linf_h,
            xlabel="Grid size h",
            title="Error vs Grid Size",
            filename="output/swashes_2d_thacker_with_limiter_error_vs_h.png",
        )
    

if __name__ == "__main__":
    main()
