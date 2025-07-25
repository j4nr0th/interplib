"""Solving the actual system."""

from collections.abc import Sequence
from typing import cast

import numpy as np
import numpy.typing as npt
import pyvista as pv
import scipy.sparse as sp
from scipy.sparse import linalg as sla

from interplib import kforms
from interplib._mimetic import compute_element_explicit, compute_element_matrices
from interplib.element import (
    ElementCollection,
    FixedElementArray,
    FlexibleElementArray,
    UnknownOrderings,
    call_per_element_fix,
    call_per_element_flex,
    compute_dof_sizes,
    compute_lagrange_sizes,
)
from interplib.kforms.eval import CompiledSystem
from interplib.mimetic.mimetic2d import BasisCache, Mesh2D
from interplib.solve_system import (
    Constraint,
    ElementConstraint,
    RefinementSettings,
    SolutionStatisticsUnsteady,
    SolverSettings,
    SystemSettings,
    TimeSettings,
    _find_time_carry_indices,
    assemble_matrix,
    assemble_vector,
    assign_leaves,
    check_and_refine,
    compute_element_rhs,
    compute_leaf_element_matrices,
    compute_vector_fields_nonlin,
    divide_old,
    extract_carry,
    mesh_boundary_conditions,
    mesh_continuity_constraints,
    non_linear_solve_run,
    reconstruct_mesh_from_solution,
)


def solve_system_2d(
    mesh: Mesh2D,
    system_settings: SystemSettings,
    refinement_settings: RefinementSettings = RefinementSettings(
        refinement_levels=0,
        division_predicate=None,
        division_function=divide_old,
    ),
    solver_settings: SolverSettings = SolverSettings(
        maximum_iterations=100,
        relaxation=1,
        absolute_tolerance=1e-6,
        relative_tolerance=1e-5,
    ),
    time_settings: TimeSettings | None = None,
    *,
    recon_order: int | None = None,
    print_residual: bool = False,
) -> tuple[Sequence[pv.UnstructuredGrid], SolutionStatisticsUnsteady]:
    """Solve the unsteady system on the specified mesh.

    Parameters
    ----------
    mesh : Mesh2D
        Mesh on which to solve the system on.

    system_settings : SystemSettings
        Settings specifying the system of equations and boundary conditions to solve for.

    refinement_settings : RefinementSettings, optional
        Settings specifying refinement of the mesh.

    solver_settings : SolverSettings, optional
        Settings specifying the behavior of the solver

    time_settings : TimeSettings or None, default: None
        When set to ``None``, the equations are solved without time dependence (steady
        state). Otherwise, it specifies which equations are time derivative related and
        time step count and size.

    recon_order : int, optional
        When specified, all elements will be reconstructed using this polynomial order.
        Otherwise, they are reconstructed with their own order.

    print_residual : bool, default: False
        Print the maximum of the absolute value of the residual for each iteration of the
        solver.

    Returns
    -------
    grids : Sequence of pyvista.UnstructuredGrid
        Reconstructed solution as an unstructured grid of VTK's "lagrange quadrilateral"
        cells. This reconstruction is done on the nodal basis for all unknowns.
    stats : SolutionStatisticsNonLin
        Statistics about the solution. This can be used for convergence tests or timing.
    """
    system = system_settings.system

    constrained_forms = system_settings.constrained_forms
    boundary_conditions = system_settings.boundary_conditions

    for _, form in constrained_forms:
        if form not in system.unknown_forms:
            raise ValueError(
                f"Form {form} which is to be zeroed is not involved in the system."
            )

        if boundary_conditions and form in (bc.form for bc in boundary_conditions):
            raise ValueError(
                f"Form {form} can not be zeroed because it is involved in a strong "
                "boundary condition."
            )

    # Make elements into a rectree
    lists = [
        check_and_refine(
            refinement_settings.division_predicate,
            refinement_settings.division_function,
            mesh.get_element(ie),
            0,
            refinement_settings.refinement_levels,
        )
        for ie in range(mesh.n_elements)
    ]
    element_list = sum(lists, start=[])
    element_collection = ElementCollection(element_list)

    # Make element matrices and vectors
    cache: dict[int, BasisCache] = dict()
    unique_orders = element_collection.orders_array.unique()
    for order in (int(order) for order in unique_orders):
        cache[order] = BasisCache(order, order + 2)

    if recon_order is not None and recon_order not in cache:
        cache[int(recon_order)] = BasisCache(int(recon_order), int(recon_order))

    vector_fields = system.vector_fields

    # Create modified system to make it work with time marching.
    if time_settings is not None:
        if time_settings.sample_rate < 1:
            raise ValueError("Sample rate can not be less than 1.")

        if len(time_settings.time_march_relations) < 1:
            raise ValueError("Problem has no time march relations.")

        for w, u in time_settings.time_march_relations.items():
            if u not in system.unknown_forms:
                raise ValueError(f"Unknown form {u} is not in the system.")
            if w not in system.weight_forms:
                raise ValueError(f"Weight form {w} is not in the system.")
            if u.primal_order != w.primal_order:
                raise ValueError(
                    f"Forms {u} and {w} in the time march relation can not be used, as "
                    f"they have differing primal orders ({u.primal_order} vs "
                    f"{w.primal_order})."
                )

        time_march_indices = tuple(
            (
                system.unknown_forms.index(time_settings.time_march_relations[eq.weight])
                if eq.weight in time_settings.time_march_relations
                else None
            )
            for eq in system.equations
        )

        new_equations: list[kforms.KEquation] = list()
        for ie, (eq, m_idx) in enumerate(zip(system.equations, time_march_indices)):
            if m_idx is None:
                new_equations.append(eq)
            else:
                new_equations.append(
                    eq.left
                    + 2
                    / time_settings.dt
                    * (system.weight_forms[m_idx] * system.unknown_forms[m_idx])
                    == eq.right
                )

        system = kforms.KFormSystem(*new_equations)
        del new_equations

    compiled_system = CompiledSystem(system)

    # Make a system that can be used to perform an L2 projection for the initial
    # conditions.
    project_equations: list[kforms.KEquation] = list()
    for ie, eq in enumerate(system.equations):
        base_form = eq.weight.base_form
        proj_rhs = (
            0
            if base_form not in system_settings.initial_conditions
            else eq.weight @ system_settings.initial_conditions[base_form]
        )
        proj_lhs = eq.weight * eq.weight.base_form
        project_equations.append(proj_lhs == proj_rhs)  # type: ignore

    projection_system = kforms.KFormSystem(*project_equations)
    projection_compiled = CompiledSystem(projection_system)
    projection_codes = projection_compiled.linear_codes
    del project_equations

    # Explicit right side
    explicit_vec: npt.NDArray[np.float64]

    # Prepare for evaluation of matrices/vectors
    corners = np.stack([v for v in element_collection.corners_array], axis=0)
    leaf_elements = np.flatnonzero(
        np.concatenate(element_collection.child_count_array.values) == 0
    )
    bl = corners[leaf_elements, 0]
    br = corners[leaf_elements, 1]
    tr = corners[leaf_elements, 2]
    tl = corners[leaf_elements, 3]
    # NOTE: does not work with differing orders yet
    orde = np.array(element_collection.orders_array)[leaf_elements, 0]
    c_ser = tuple(cache[o].c_serialization() for o in cache if o in orde)

    # Release cache element memory. If they will be needed in the future,
    # they will be recomputed, but they consume LOTS of memory

    linear_vectors = call_per_element_flex(
        element_collection.com,
        1,
        np.float64,
        compute_element_rhs,
        system,
        cache,
        element_collection.corners_array,
        element_collection.orders_array,
        element_collection.child_count_array,
    )

    unknown_ordering = UnknownOrderings(*(form.order for form in system.unknown_forms))
    dof_sizes = compute_dof_sizes(element_collection, unknown_ordering)
    lagrange_counts = compute_lagrange_sizes(element_collection, unknown_ordering)
    dof_offsets = call_per_element_fix(
        element_collection.com,
        np.uint32,
        dof_sizes.shape[0] + 1,
        lambda i, x: np.pad(np.cumsum(x[i]), (1, 0)),
        dof_sizes,
    )
    total_dof_counts = call_per_element_fix(
        element_collection.com,
        np.uint32,
        1,
        lambda i, x, y: x[i][-1] + y[i],
        dof_offsets,
        lagrange_counts,
    )

    solution = FlexibleElementArray(element_collection.com, np.float64, total_dof_counts)

    if system_settings.initial_conditions:
        initial_vectors = call_per_element_flex(
            element_collection.com,
            1,
            np.float64,
            compute_element_rhs,
            projection_system,
            cache,
            element_collection.corners_array,
            element_collection.orders_array,
            element_collection.child_count_array,
        )

        projection_matrices = compute_element_matrices(
            [f.order for f in system.unknown_forms],
            projection_compiled.linear_codes,
            bl,
            br,
            tr,
            tl,
            orde,
            tuple(),
            np.zeros((orde.size + 1), np.uint64),
            c_ser,
        )
        distributed_projections = assign_leaves(
            element_collection, 2, np.float64, projection_matrices
        )  # TODO: factor
        del projection_matrices

        def _inverse_for_leaves(
            ie: int,
            child_counts: FixedElementArray[np.uint32],
            mat: FlexibleElementArray[np.float64, np.uint32],
            vec: FlexibleElementArray[np.float64, np.uint32],
            total_dofs: FixedElementArray[np.uint32],
        ) -> npt.NDArray[np.float64]:
            """Compute inverse for each leaf element."""
            n_dofs = int(total_dofs[ie][0])
            if int(child_counts[ie][0]) != 0:
                return np.zeros(n_dofs, np.float64)

            res = np.astype(np.linalg.solve(mat[ie], vec[ie]), np.float64, copy=False)
            assert res.size == n_dofs
            return res

        initial_solution = call_per_element_flex(
            element_collection.com,
            1,
            np.float64,
            _inverse_for_leaves,
            element_collection.child_count_array,
            distributed_projections,
            initial_vectors,
            total_dof_counts,
        )
        del distributed_projections
    else:
        initial_vectors = None
        initial_solution = None

    if time_settings is not None:
        time_carry_index_array = call_per_element_flex(
            element_collection.com,
            1,
            np.uint32,
            _find_time_carry_indices,
            tuple(
                sorted(
                    system.weight_forms.index(form)
                    for form in time_settings.time_march_relations
                )
            ),
            dof_offsets,
            element_collection.child_count_array,
        )
        if initial_vectors and initial_solution:
            # compute carry
            old_solution_carry = extract_carry(
                element_collection, time_carry_index_array, initial_vectors
            )
            solution = initial_solution
        else:
            old_solution_carry = FlexibleElementArray(
                element_collection.com, np.float64, time_carry_index_array.shapes
            )
    else:
        time_carry_index_array = None
        old_solution_carry = None

    del initial_solution, initial_vectors

    assert compiled_system.linear_codes

    # Compute vector fields at integration points for leaf elements
    vec_field_offsets, vec_fields = compute_vector_fields_nonlin(
        system,
        leaf_elements,
        cache,
        vector_fields,
        element_collection.corners_array,
        element_collection.orders_array,
        element_collection.orders_array,
        dof_offsets,
        solution,
    )

    linear_element_matrices = compute_leaf_element_matrices(
        unknown_ordering,
        element_collection,
        compiled_system.linear_codes,
        bl,
        br,
        tr,
        tl,
        orde,
        c_ser,
        vec_field_offsets,
        vec_fields,
    )

    main_mat = assemble_matrix(
        unknown_ordering,
        element_collection,
        dof_offsets,
        linear_element_matrices,
    )
    main_vec = assemble_vector(
        unknown_ordering,
        element_collection,
        dof_offsets,
        lagrange_counts,
        linear_vectors,
    )

    def _find_constrained_indices(
        ie: int,
        i_unknown: int,
        child_count: FixedElementArray[np.uint32],
        dof_offsets: FixedElementArray[np.uint32],
    ) -> npt.NDArray[np.uint32]:
        """Find indices of DoFs that should be constrained for an element."""
        if int(child_count[ie][0]) != 0:
            return np.zeros(0, np.uint32)
        offsets = dof_offsets[ie]
        return np.arange(offsets[i_unknown], offsets[i_unknown + 1], dtype=np.uint32)

    # Generate constraints that force the specified for to have the (child element) sum
    # equal to a prescribed value.
    constrained_form_constaints = {
        form: Constraint(
            k,
            *(
                ElementConstraint(ie, dofs, np.ones_like(dofs, dtype=np.float64))
                for ie, dofs in enumerate(
                    call_per_element_flex(
                        element_collection.com,
                        1,
                        np.uint32,
                        _find_constrained_indices,
                        system.unknown_forms.index(form),
                        element_collection.child_count_array,
                        dof_offsets,
                    )
                )
            ),
        )
        for k, form in constrained_forms
    }

    if boundary_conditions is None:
        boundary_conditions = list()

    top_indices = np.astype(
        np.flatnonzero(np.array(element_collection.parent_array) == 0),
        np.uint32,
        copy=False,
    )

    strong_bc_constraints, weak_bc_constraints = mesh_boundary_conditions(
        [eq.right for eq in system.equations],
        mesh,
        unknown_ordering,
        element_collection,
        dof_offsets,
        top_indices,
        [
            [bc for bc in boundary_conditions if bc.form == eq.weight.base_form]
            for eq in system.equations
        ],
        cache,
    )

    continuity_constraints = mesh_continuity_constraints(
        system,
        mesh,
        top_indices,
        unknown_ordering,
        element_collection,
        unique_orders,
        dof_offsets,
    )

    element_offset = np.astype(
        np.pad(np.array(total_dof_counts, np.uint32).flatten().cumsum(), (1, 0)),
        np.uint32,
        copy=False,
    )

    constraint_rows: list[npt.NDArray[np.uint32]] = list()
    constraint_cols: list[npt.NDArray[np.uint32]] = list()
    constraint_coef: list[npt.NDArray[np.float64]] = list()
    constraint_vals: list[float] = list()
    # Continuity constraints
    ic = 0
    for constraint in continuity_constraints:
        constraint_vals.append(constraint.rhs)
        for ec in constraint.element_constraints:
            offset = int(element_offset[ec.i_e])
            constraint_cols.append(ec.dofs + offset)
            constraint_rows.append(np.full_like(ec.dofs, ic))
            constraint_coef.append(ec.coeffs)
        ic += 1

    # Form constraining
    for form in constrained_form_constaints:
        constraint = constrained_form_constaints[form]
        constraint_vals.append(constraint.rhs)
        for ec in constraint.element_constraints:
            offset = int(element_offset[ec.i_e])
            constraint_cols.append(ec.dofs + offset)
            constraint_rows.append(np.full_like(ec.dofs, ic))
            constraint_coef.append(ec.coeffs)
        ic += 1

    # Strong BC constraints
    for ec in strong_bc_constraints:
        offset = int(element_offset[ec.i_e])
        for ci, cv in zip(ec.dofs, ec.coeffs, strict=True):
            constraint_rows.append(np.array([ic]))
            constraint_cols.append(np.array([ci + offset]))
            constraint_coef.append(np.array([1.0]))
            constraint_vals.append(float(cv))

            ic += 1

    # Weak BC constraints/additions
    for ec in weak_bc_constraints:
        offset = element_offset[ec.i_e]
        main_vec[ec.dofs] += ec.coeffs
    if constraint_coef:
        lagrange_mat = sp.csr_array(
            (
                np.concatenate(constraint_coef),
                (np.concatenate(constraint_rows), np.concatenate(constraint_cols)),
            )
        )
        lagrange_mat.resize((ic, element_offset[-1]))
        main_mat = cast(
            sp.csr_array,
            sp.block_array(
                ((main_mat, lagrange_mat.T), (lagrange_mat, None)), format="csr"
            ),
        )
        lagrange_vec = np.array(constraint_vals, np.float64)
        main_vec = np.concatenate((main_vec, lagrange_vec))
    else:
        lagrange_mat = None
        lagrange_vec = np.zeros(0, np.float64)

    # # TODO: Delet dis
    # from matplotlib import pyplot as plt

    # plt.figure()
    # plt.imshow(main_mat.toarray())
    # # plt.spy(main_mat)
    # plt.show()

    linear_matrix = sp.csc_array(main_mat)
    explicit_vec = main_vec

    if time_settings is not None:
        assert time_carry_index_array is not None
        time_carry_term = extract_carry(
            element_collection, time_carry_index_array, linear_vectors
        )
    else:
        time_carry_term = None
    del main_mat, main_vec

    system_decomp = sla.splu(linear_matrix)

    resulting_grids: list[pv.UnstructuredGrid] = list()

    grid = reconstruct_mesh_from_solution(
        system,
        recon_order,
        element_collection,
        cache,
        dof_offsets,
        solution,
    )
    grid.field_data["time"] = (0.0,)
    resulting_grids.append(grid)

    global_lagrange = np.zeros_like(lagrange_vec)
    max_mag = np.abs(explicit_vec).max()

    max_iterations = solver_settings.maximum_iterations
    relax = solver_settings.relaxation
    atol = solver_settings.absolute_tolerance
    rtol = solver_settings.relative_tolerance

    if time_settings is not None:
        nt = time_settings.nt
        dt = time_settings.dt
        changes = np.zeros(nt, np.float64)
        iters = np.zeros(nt, np.uint32)

        for time_index in range(nt):
            max_residual = np.inf
            # 2 / dt * old_solution_carry + time_carry_term
            current_carry = call_per_element_flex(
                element_collection.com,
                1,
                np.float64,
                lambda ie, x, y: 2 / dt * x[ie] + y[ie],
                old_solution_carry,
                time_carry_term,
            )
            new_solution, global_lagrange, iter_cnt, max_residual = non_linear_solve_run(
                system,
                max_iterations,
                relax,
                atol,
                rtol,
                print_residual,
                unknown_ordering,
                element_collection,
                leaf_elements,
                cache,
                compiled_system,
                explicit_vec,
                bl,
                br,
                tr,
                tl,
                orde,
                c_ser,
                dof_offsets,
                element_offset,
                linear_element_matrices,
                time_carry_index_array,
                current_carry,
                solution,
                global_lagrange,
                max_mag,
                vector_fields,
                system_decomp,
                lagrange_mat,
            )

            changes[time_index] = float(max_residual)
            iters[time_index] = iter_cnt
            updated_derivative = assign_leaves(
                element_collection,
                1,
                np.float64,
                compute_element_explicit(
                    np.concatenate(new_solution, dtype=np.float64),
                    element_offset[leaf_elements],
                    [f.order for f in system.unknown_forms],
                    projection_codes,
                    bl,
                    br,
                    tr,
                    tl,
                    orde,
                    vec_fields,
                    vec_field_offsets,
                    c_ser,
                ),
            )
            assert time_carry_index_array is not None
            new_solution_carry = extract_carry(
                element_collection, time_carry_index_array, updated_derivative
            )
            # Compute time carry
            new_time_carry_term = call_per_element_flex(
                element_collection.com,
                1,
                np.float64,
                lambda ie, x, y, z: 2 / dt * (x[ie] - y[ie]) - z[ie],
                new_solution_carry,
                old_solution_carry,
                time_carry_term,
            )
            # 2 / dt * (new_solution_carry - old_solution_carry) - time_carry_term

            solution = new_solution
            time_carry_term = new_time_carry_term
            old_solution_carry = new_solution_carry
            del new_solution_carry, new_time_carry_term, new_solution, updated_derivative

            if (time_index % time_settings.sample_rate) == 0 or time_index + 1 == nt:
                # Prepare to build up the 1D Splines

                grid = reconstruct_mesh_from_solution(
                    system,
                    recon_order,
                    element_collection,
                    cache,
                    dof_offsets,
                    solution,
                )
                grid.field_data["time"] = (float((time_index + 1) * dt),)
                resulting_grids.append(grid)

            if print_residual:
                print(
                    f"Time step {time_index:d} finished in {iter_cnt:d} iterations with"
                    f" residual of {max_residual:.5e}"
                )
    else:
        new_solution, global_lagrange, iter_cnt, changes = non_linear_solve_run(
            system,
            max_iterations,
            relax,
            atol,
            rtol,
            print_residual,
            unknown_ordering,
            element_collection,
            leaf_elements,
            cache,
            compiled_system,
            explicit_vec,
            bl,
            br,
            tr,
            tl,
            orde,
            c_ser,
            dof_offsets,
            element_offset,
            linear_element_matrices,
            None,
            None,
            solution,
            global_lagrange,
            max_mag,
            vector_fields,
            system_decomp,
            lagrange_mat,
        )
        iters = np.array((iter_cnt,), np.uint32)  # type: ignore

        solution = new_solution
        del new_solution

        # Prepare to build up the 1D Splines

        grid = reconstruct_mesh_from_solution(
            system,
            recon_order,
            element_collection,
            cache,
            dof_offsets,
            solution,
        )

        resulting_grids.append(grid)

    del c_ser, bl, br, tr, tl, orde
    # TODO: solution statistics
    orders, counts = np.unique(
        np.array(element_collection.orders_array), return_counts=True
    )
    stats = SolutionStatisticsUnsteady(
        element_orders={int(order): int(count) for order, count in zip(orders, counts)},
        n_total_dofs=explicit_vec.size,
        n_lagrange=int(lagrange_vec.size + np.array(lagrange_counts).sum()),
        n_elems=element_collection.com.element_cnt,
        n_leaves=len(leaf_elements),
        n_leaf_dofs=sum(int(total_dof_counts[int(ie)][0]) for ie in leaf_elements),
        iter_history=iters,
        residual_history=changes,
    )

    return tuple(resulting_grids), stats
