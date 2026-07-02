# Code that simulates waves in an urban setting (Avenyn in Gothenburg) using the shallow water equations with a TVB limiter. 
# The mesh is loaded from an XDMF file, and building walls are treated as reflective boundaries. 
# The code outputs the water depth at regular time intervals for visualization as .xdmf and .h5 files.

from pathlib import Path

import basix.ufl
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, mesh
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from dolfinx.io import XDMFFile

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
    normal_xy = ufl.as_vector([normal[0], normal[1]])
    return flux_x(U, gravity, h_floor) * normal_xy[0] + flux_y(U, gravity, h_floor) * normal_xy[1]

def max_wave_speed(U, normal, gravity, h_floor):
    h, hu, hv = split_state(U)
    h_safe = ufl.max_value(h, h_floor)
    normal_xy = ufl.as_vector([normal[0], normal[1]])
    un = (hu * normal_xy[0] + hv * normal_xy[1]) / h_safe
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
    normal_xy = ufl.as_vector([normal[0], normal[1]])
    normal_momentum = ufl.dot(momentum, normal_xy)
    reflected_momentum = momentum - 2.0 * normal_momentum * normal_xy
    return ufl.as_vector([h, reflected_momentum[0], reflected_momentum[1]])


# DG weak form
def build_rhs_form(domain, V, U_state, gravity, h_floor, wall_facets=None):
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

    grad_xy = ufl.as_tensor(
        [
            [W[0].dx(0), W[0].dx(1)],
            [W[1].dx(0), W[1].dx(1)],
            [W[2].dx(0), W[2].dx(1)],
        ]
    )

    exterior_state = reflective_exterior_state(U_state, n)
    boundary_flux = lax_friedrichs_flux(U_state, exterior_state, n, gravity, h_floor)

    form = ufl.inner(physical_flux, grad_xy) * ufl.dx - ufl.inner(boundary_flux, W) * ufl.ds

    if wall_facets is None:
        interior_flux = lax_friedrichs_flux(U_state("-"), U_state("+"), n("-"), gravity, h_floor)
        form += -ufl.inner(interior_flux, W("-") - W("+")) * ufl.dS
    else:
        dS_all = ufl.Measure("dS", domain=domain, subdomain_data=wall_facets)

        # Regular interior facets: marker 0
        interior_flux = lax_friedrichs_flux(U_state("-"), U_state("+"), n("-"), gravity, h_floor)
        form += -ufl.inner(interior_flux, W("-") - W("+")) * dS_all(0)

        # Building wall facets: marker 1
        wall_state_minus = reflective_exterior_state(U_state("-"), n("-"))
        wall_state_plus = reflective_exterior_state(U_state("+"), -n("-"))

        wall_flux_minus = lax_friedrichs_flux(U_state("-"), wall_state_minus, n("-"), gravity, h_floor)
        wall_flux_plus = lax_friedrichs_flux(U_state("+"), wall_state_plus, -n("-"), gravity, h_floor)

        form += -ufl.inner(wall_flux_minus, W("-")) * dS_all(1)
        form += -ufl.inner(wall_flux_plus, W("+")) * dS_all(1)

    return form


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


# ===== Mesh/Geometry helper functions ========================
def domain_bounds(domain):
    coords = domain.geometry.x
    mins = np.min(coords, axis=0)
    maxs = np.max(coords, axis=0)
    return mins, maxs

def build_building_wall_facets(domain, cell_tags):
    tdim = domain.topology.dim
    fdim = tdim - 1

    domain.topology.create_connectivity(fdim, tdim)
    f_to_c = domain.topology.connectivity(fdim, tdim)

    num_cells = domain.topology.index_map(tdim).size_local + domain.topology.index_map(tdim).num_ghosts
    num_facets = domain.topology.index_map(fdim).size_local + domain.topology.index_map(fdim).num_ghosts

    cell_marker_array = np.full(num_cells, -999, dtype=np.int32)
    cell_marker_array[cell_tags.indices] = cell_tags.values

    facet_indices = []
    facet_values = []

    # Mark facets between building and non-building cells as reflective wall interfaces
    for facet in range(num_facets):
        adjacent_cells = f_to_c.links(facet)

        # only interior facets
        if len(adjacent_cells) != 2:
            continue

        c0, c1 = adjacent_cells
        m0 = cell_marker_array[c0]
        m1 = cell_marker_array[c1]

        is_building_0 = m0 >= 0
        is_building_1 = m1 >= 0

        facet_indices.append(facet)
        facet_values.append(1 if is_building_0 != is_building_1 else 0)

    facet_indices = np.array(facet_indices, dtype=np.int32)
    facet_values = np.array(facet_values, dtype=np.int32)

    return mesh.meshtags(domain, fdim, facet_indices, facet_values)

def build_limiter_cell_mask(domain, wall_facets):
    tdim = domain.topology.dim
    fdim = tdim - 1

    domain.topology.create_connectivity(fdim, tdim)
    f_to_c = domain.topology.connectivity(fdim, tdim)

    num_cells = domain.topology.index_map(tdim).size_local + domain.topology.index_map(tdim).num_ghosts
    limit_cells = np.zeros(num_cells, dtype=bool)

    if wall_facets is None:
        limit_cells[:] = True
        return limit_cells

    # Limit only cells adjacent to building-wall facets when an urban mesh is used
    for facet, marker in zip(wall_facets.indices, wall_facets.values):
        if marker != 1:
            continue
        for cell in f_to_c.links(facet):
            limit_cells[cell] = True

    return limit_cells


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


# ===== Main Simulation =======================================
def main():
    comm = MPI.COMM_WORLD
    Path("output").mkdir(exist_ok=True)

    mesh_path = Path("/home/c_a00/SWE_flooding_project/dtcc-sim/sandbox/output/flat_mesh_gbg.xdmf")
    use_city_mesh = True
    nx = 100
    ny = 100
    Lx = 3.0
    Ly = 1.0
    # Top quarter of Avenyn = (319996, 6399170), Poseidon = (320023.0, 6398980.0), Bottom quarter of Avenyn = (319951.0, 6398780.0)
    x_center = 1 #None
    y_center = 0.75 #None

    final_time = 1.0
    cfl = 0.08
    gravity = 9.81
    h0 = 1.0
    h_floor = 1.0e-6
    degree = 1
    tvb_M = 0.5
    tvb_nu = 1.5
    n_outputs = 120
    output_dt = final_time / n_outputs
    next_output_time = output_dt

    if use_city_mesh:
        if not mesh_path.exists():
            raise FileNotFoundError(
                f"City mesh '{mesh_path}' was not found. Run build_flat_mesh_gbg.py first "
                "or set use_city_mesh = False."
            )

        with XDMFFile(comm, str(mesh_path), "r") as xdmf:
            domain = xdmf.read_mesh(name="mesh")
            cell_tags = xdmf.read_meshtags(domain, name="boundary_markers")

        if comm.rank == 0:
            print(f"Loaded city mesh from {mesh_path}")

    else:
        domain = mesh.create_rectangle(
            comm,
            [np.array([0.0, 0.0]), np.array([Lx, Ly])],
            [nx, ny],
        )
        cell_tags = None

    mins, maxs = domain_bounds(domain)
    if x_center is None:
        x_center = 0.5 * (mins[0] + maxs[0])
    if y_center is None:
        y_center = 0.5 * (mins[1] + maxs[1])

    V = build_state_space(domain, degree=degree)

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

    if use_city_mesh:
        wall_facets = build_building_wall_facets(domain, cell_tags)
    else:
        wall_facets = None
        limit_cells = None


    limit_cells = build_limiter_cell_mask(domain, wall_facets)

    def initial_depth(x):
        sigma = 0.1
        amplitude = 0.2
        r2 = (x[0] - x_center) ** 2 + (x[1] - y_center) ** 2
        values = np.empty((1, x.shape[1]), dtype=PETSc.ScalarType)
        values[0] = h0 + amplitude * np.exp(-r2 / (2.0 * sigma**2))
        return values

    def zero_field(x):
        values = np.zeros((1, x.shape[1]), dtype=PETSc.ScalarType)
        return values

    h_init.interpolate(initial_depth)
    hu_init.interpolate(zero_field)
    hv_init.interpolate(zero_field)

    limiter_data = build_tvb_limiter_data(domain, Q_h)
    component_dofs = [h_dofs, hu_dofs, hv_dofs]

    U_n.x.array[h_dofs] = h_init.x.array
    U_n.x.array[hu_dofs] = hu_init.x.array
    U_n.x.array[hv_dofs] = hv_init.x.array
    U_n.x.scatter_forward()

    trial = ufl.TrialFunction(V)
    test = ufl.TestFunction(V)
    mass_form = fem.form(ufl.inner(trial, test) * ufl.dx)
    mass_matrix = assemble_matrix(mass_form)
    mass_matrix.assemble()

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(mass_matrix)
    ksp.setType("preonly")
    ksp.getPC().setType("bjacobi")

    rhs_form = fem.form(
        build_rhs_form(
            domain,
            V,
            U_eval,
            gravity,
            h_floor,
            wall_facets=wall_facets,
        )
    )

    rhs_vec = assemble_vector(rhs_form)

    # Project DG depth to CG space for easier XDMF visualization
    Q_out = fem.functionspace(domain, ("CG", 1))
    h_out = fem.Function(Q_out, name="depth")

    def update_depth_output():
        h_tmp = U_n.sub(0).collapse()
        h_out.interpolate(h_tmp)
        h_out.x.scatter_forward()
    
    update_depth_output()

    with XDMFFile(comm, "output/new_test_with_limiter_2.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(h_out, 0.0)

        t = 0.0
        step = 0
        if use_city_mesh:
            num_cells = domain.topology.index_map(domain.topology.dim).size_global
            coords = domain.geometry.x
            mins = np.min(coords, axis=0)
            maxs = np.max(coords, axis=0)
            area = (maxs[0] - mins[0]) * (maxs[1] - mins[1])
            h_min = np.sqrt(area / max(num_cells, 1))
        else:
            h_min = min(Lx / nx, Ly / ny)

        while t < final_time - 1.0e-12:
            dt = compute_time_step(U_n, h_dofs, hu_dofs, hv_dofs, gravity, cfl, h_min, comm)
            dt = min(dt, final_time - t)

            U_0.x.array[:] = U_n.x.array

            # SSP-RK3 stages, comment apply_tvb_limiter if running without limiter
            U_eval.x.array[:] = U_0.x.array
            solve_stage(ksp, rhs_form, dt, U_eval, U_0, U_1, rhs_vec)
            apply_tvb_limiter(U_1, limiter_data, component_dofs, tvb_M, tvb_nu, h_min, limit_cells=limit_cells)

            U_eval.x.array[:] = U_1.x.array
            solve_stage(ksp, rhs_form, dt, U_eval, U_1, U_2, rhs_vec)
            U_2.x.array[:] = (3.0 / 4.0) * U_0.x.array + (1.0 / 4.0) * U_2.x.array
            apply_tvb_limiter(U_2, limiter_data, component_dofs, tvb_M, tvb_nu, h_min, limit_cells=limit_cells)

            U_eval.x.array[:] = U_2.x.array
            solve_stage(ksp, rhs_form, dt, U_eval, U_2, U_n, rhs_vec)
            U_n.x.array[:] = (1.0 / 3.0) * U_0.x.array + (2.0 / 3.0) * U_n.x.array
            apply_tvb_limiter(U_n, limiter_data, component_dofs, tvb_M, tvb_nu, h_min, limit_cells=limit_cells)

            U_n.x.array[h_dofs] = np.maximum(U_n.x.array[h_dofs], h_floor)
            U_n.x.scatter_forward()
            
            h_vals = np.maximum(U_n.x.array[h_dofs], h_floor)
            hu_vals = U_n.x.array[hu_dofs]
            hv_vals = U_n.x.array[hv_dofs]

            u_vals = hu_vals / h_vals
            v_vals = hv_vals / h_vals
            Fr_vals = np.sqrt(u_vals**2 + v_vals**2) / np.sqrt(gravity * h_vals)
            Fr_max = comm.allreduce(float(np.max(Fr_vals)), op=MPI.MAX)

            t += dt
            step += 1

            if comm.rank == 0:
                h_vals = U_n.x.array[h_dofs]
                print(
                    f"step={step:05d} t={t:.5f} dt={dt:.3e} "
                    f"h_min={h_vals.min():.6f} h_max={h_vals.max():.6f} "
                    f"Fr_max={Fr_max:.6f}"
                )

            if t >= next_output_time or t >= final_time - 1.0e-12:
                update_depth_output()
                print(f"write t={t:.5f}, h min={h_out.x.array.min()}, h max={h_out.x.array.max()}")
                xdmf.write_function(h_out, t)
                next_output_time += output_dt

if __name__ == "__main__":
    main()
