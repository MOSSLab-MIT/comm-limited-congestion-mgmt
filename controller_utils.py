import grid_pkg
import controller_pkg
import numpy as np
import torch
import torch.nn as nn
import time
import cvxpy as cp

import torch

import numpy as np
import scipy as sp

from scipy.sparse import csc_matrix,csr_matrix,coo_matrix

def csc_torch_to_scipy(A):
    return csc_matrix((A.values().detach().numpy(),A.row_indices().numpy(),A.ccol_indices().numpy()),shape=A.size())

def csc_scipy_to_torch(A):
    return torch.sparse_csc_tensor(torch.tensor(A.indptr,dtype=torch.int64),torch.tensor(A.indices,dtype=torch.int64),torch.tensor(A.data),size=np.shape(A),dtype=torch.float64)

def coo_torch_to_scipy(A):
    # A must be coalesced
    return coo_matrix((A.values().detach().numpy(),(A._indices()[0,:].numpy(),A._indices()[1,:].numpy())),shape=A.size())

def coo_scipy_to_torch(A):
    i = torch.tensor(np.vstack((A.row,A.col)),dtype=torch.long)
    v = torch.tensor(A.data,dtype=torch.float64)
    return torch.sparse_coo_tensor(i,v,size=np.shape(A))

def initialize_torch_from_npz(filename):
    data = np.load(filename, allow_pickle=True)
    Qnp, qnp, Gnp, hnp, Anp, bnp = data["Q"][()], data["q"][()], data["G"][()], data["h"][()], data["A"][()], data["b"][()]

    Qnp, Gnp = Qnp.tocoo(), Gnp.tocoo()
    __to_coo = lambda M: torch.sparse_coo_tensor(torch.stack([torch.tensor(M.row), torch.tensor(M.col)]),
                                                 torch.tensor(M.data), M.shape, dtype=torch.float64, requires_grad=True)
    Q, G = __to_coo(Qnp).to_sparse_csc(), __to_coo(Gnp).to_sparse_csc()

    q = torch.tensor(qnp, requires_grad=True)
    h = torch.tensor(hnp, requires_grad=True)
    if Anp is not None:
        Anp = Anp.tocoo()
        A = __to_coo(Anp).to_sparse_csc()
        b = torch.tensor(bnp, requires_grad=True)

    return Q,q,G,h,A,b

class sparse_row_norm(torch.autograd.Function):
    """Differentiable sparse row-wise 1 or 2 norm (not natively supported in torch 2.3.1)."""

    @staticmethod
    def forward(ctx, A, p):
        assert(p == 1 or p == 2)

        ctx.p = p

        A = csc_torch_to_scipy(A) # TODO : convert to list of sparse matrices if sparse
        ctx.A = A

        N = np.expand_dims(sp.sparse.linalg.norm(A,ord=p,axis=1),-1)
        ctx.N = N

        return torch.tensor(N,dtype=torch.float64)

    @staticmethod
    def backward(ctx,grad_output):
        if ctx.p == 1:
            dN = ctx.A.sign().multiply(grad_output.numpy())
        elif ctx.p == 2:
            dN = ctx.A.multiply(grad_output.numpy() * np.power(ctx.N,-1))

        # return torch.tensor(dN.todense(),dtype=torch.float64), None # TODO : sparse or not?
        return csc_scipy_to_torch(csc_matrix(dN)), None


class sparse_row_normalize(torch.autograd.Function):
    """Differentiable sparse row normalization (not natively supported in torch)."""

    @staticmethod
    def forward(ctx, A, N):
        N = N.numpy()
        ctx.N = N
        ctx.A = csc_torch_to_scipy(A)

        A = csr_matrix(ctx.A)
        A.data /= N[A.nonzero()[0],0] # normalize rows in-place
        A = csc_matrix(A)

        return csc_scipy_to_torch(A)

    @staticmethod
    def backward(ctx,grad_output):
        if grad_output.layout is torch.strided:
            dL = grad_output.numpy()
        else:
            dL = grad_output.to_dense().numpy()

        dA_norm = np.multiply(np.power(ctx.N, -1), dL)
        dN = -np.multiply(np.expand_dims(np.diag(ctx.A @ dL.T), -1), np.power(ctx.N, -2))

        return torch.tensor(dA_norm, dtype=torch.float64), torch.tensor(dN, dtype=torch.float64)


def test_normalization():
    # p = 2
    p = 1

    rng = np.random.default_rng()
    S = sp.sparse.random(3, 4, density=0.5, random_state=rng, format="csc")

    S = csc_scipy_to_torch(S)
    S.requires_grad_()

    D = S.clone().detach().to_dense().numpy()
    D = torch.tensor(D,dtype=torch.float64,requires_grad=True)

    print("Inputs: ")
    print(D)
    print(S.to_dense())
    print("\n")

    row_norm = sparse_row_norm.apply
    row_normalize = sparse_row_normalize.apply

    db = torch.rand(3,1,dtype=torch.float64,requires_grad=True)
    sb = db.clone().detach().requires_grad_()

    N_D = torch.unsqueeze(torch.linalg.vector_norm(D, ord=p, dim=-1),-1)
    D_new = torch.div(D, N_D)
    db_new = torch.div(db, N_D)

    N_S = row_norm(S,p)
    S_new = row_normalize(S,N_S)
    sb_new = torch.div(sb, N_S)

    # S = S.to_sparse_coo()
    # mat = torch_sparse.SparseTensor(row=S._indices()[0, :], col=S._indices()[1, :], value=S._values())

    print("Check outputs: ")
    print(N_D)
    print(N_S)
    print(db)
    print(sb)
    print("Check outputs: ")
    print("\n")

    print("Check gradients: ")
    l_D = D_new
    l_S = S_new
    l_D.backward(torch.ones(3,4))
    l_S.backward(torch.ones(3,4))
    print("Check gradients: ")
    print(D.grad)
    print(S.grad)

    return


def get_RTE_noise_values(file_path, grid, T, H, power_scale, bus_idx_gap, offset=0):
    """Generates grid disturbance values from an open-source RTE power injection dataset.

    Args:
        file_path (str): Path to the RTE noise data file.
        grid (Grid): Grid that the disturbances are generated for.
        T (int): Number of control timesteps.
        H (int): Size of prediction window.
        power_scale (float): Scale factor applied to disturbances.
        bus_idx_gap (int): Controls spatial correlation; larger values reduce correlation between distant buses.
        offset (int): Index offset into the disturbance trajectory.

    Returns:
        torch.Tensor: Disturbance values of shape (T+H, num_buses).

    Raises:
        ValueError: If offset or bus_idx_gap causes an index overflow.
    """
    data = np.loadtxt(file_path)
    vec = torch.tensor(data)
    disturbance_length = T + H
    all_disturbances = torch.zeros(disturbance_length, grid.num_buses)

    for i in range(grid.num_curt):
        bus = grid.bus_with_curt[i]
        bus_max_gen = grid.curt_max_limits[i]
        start_idx = offset + bus * bus_idx_gap
        idx = torch.arange(start_idx, start_idx + grid.delta_t*(disturbance_length+1), grid.delta_t, dtype=torch.long)

        if start_idx + grid.delta_t*(disturbance_length+1) >= vec.shape[0]:
            raise ValueError("Offset / bus gap is too large, index overflow occurred.")

        bus_power = bus_max_gen * vec[idx]
        bus_disturbances = torch.diff(bus_power)
        all_disturbances[:,bus] = bus_disturbances
    
    # normalize
    all_disturbances = power_scale * (all_disturbances / torch.mean(torch.abs(all_disturbances)))

    return all_disturbances


def get_iid_noise_values(T, H, num_buses, max_noise, max_trend=0):
    """Generates i.i.d. disturbance values across the network.

    Args:
        T (int): Number of control timesteps.
        H (int): Size of prediction window.
        num_buses (int): Number of buses in the network.
        max_noise (float): Maximum noise magnitude at a bus.
        max_trend (float): Maximum trend magnitude at a bus.

    Returns:
        torch.Tensor: Disturbance values of shape (T+H, num_buses).
    """
    trend = 2 * max_trend * (torch.rand(num_buses) - 0.5)
    trend_values = trend.unsqueeze(0).repeat(T+H,1)
    noise_values = 2 * max_noise * (torch.rand(T+H,num_buses)-0.5)
    return trend_values + noise_values


def create_base_controllers(grid, num_controllers, partition, T, H, batt_cost, curt_change_cost, curt_net_cost, bus_slack_cost, line_slack_cost):
    """Initializes base controller agents on a grid given a desired bus partition.

    Args:
        grid (Grid): Full grid the agents act on.
        num_controllers (int): Number of controllers.
        partition (list of lists): Buses each agent controls; agent i controls buses in partition[i].
        T (int): Number of control timesteps.
        H (int): Size of prediction window.
        batt_cost (float): Cost on battery charge deviation from target. Must be non-negative.
        curt_change_cost (float): Cost on changes in curtailed power. Must be non-negative.
        curt_net_cost (float): Cost on net curtailed power. Must be non-negative.
        bus_slack_cost (float): Bus slack penalty. Must be non-negative.
        line_slack_cost (float): Line slack penalty. Must be non-negative.

    Returns:
        list[Base_Controller]: Initialized controllers in partition order.

    Raises:
        ValueError: If partition count mismatches num_controllers, partition is invalid, or costs are negative.
    """
    if not len(partition) == num_controllers:
        raise ValueError(f"Number of partitions {len(partition)} does not match number of agents {num_controllers}")
    if not is_partition(partition, [i for i in range(grid.num_buses)]):
        raise ValueError(f"Partition is not a valid partition of the grid")
    if curt_change_cost < 0:
            raise ValueError(f"curt_change_cost {curt_change_cost} is negative.")
    if curt_net_cost < 0:
        raise ValueError(f"init_curt_net_cost {curt_net_cost} is negative.")

    controller_list = list()
    for i in range(num_controllers):
        buses_in_area = torch.tensor(partition[i], dtype=torch.int)
        controller_list.append(controller_pkg.Base_Controller(grid, H, buses_in_area, batt_cost, curt_change_cost, curt_net_cost,
                                        bus_slack_cost, line_slack_cost))
        
    return controller_list


def create_split_constraint_controllers(grid, num_controllers, partition, T, H, batt_cost, curt_change_cost, curt_net_cost,
                                  bus_slack_cost, line_slack_cost):
    """Initializes split-constraint controller agents on a grid given a desired bus partition.

    Args:
        grid (Grid): Full grid the agents act on.
        num_controllers (int): Number of split constraint controllers.
        partition (list of lists): Buses each agent controls; agent i controls buses in partition[i].
        T (int): Number of control timesteps.
        H (int): Size of prediction window.
        batt_cost (float): Cost on battery charge deviation from target. Must be non-negative.
        curt_change_cost (float): Cost on changes in curtailed power. Must be non-negative.
        curt_net_cost (float): Cost on net curtailed power. Must be non-negative.
        bus_slack_cost (float): Bus slack penalty. Must be non-negative.
        line_slack_cost (float): Line slack penalty. Must be non-negative.

    Returns:
        list[Split_Constraint_Controller]: Initialized controllers in partition order.

    Raises:
        ValueError: If partition count mismatches num_controllers, partition is invalid, or costs are negative.
    """
    if not len(partition) == num_controllers:
        raise ValueError(f"Number of partitions {len(partition)} does not match number of agents {num_controllers}")
    if not is_partition(partition, [i for i in range(grid.num_buses)]):
        raise ValueError(f"Partition is not a valid partition of the grid")
    if curt_change_cost < 0:
            raise ValueError(f"curt_change_cost {curt_change_cost} is negative.")
    if curt_net_cost < 0:
        raise ValueError(f"init_curt_net_cost {curt_net_cost} is negative.")

    controller_list = list()

    max_margin = torch.zeros(T+H, grid.num_lines)
    min_margin = torch.zeros(T+H, grid.num_lines)
    max_margin[0] = ( grid.line_data[:,5] - grid.init_state[0:grid.num_lines]) / num_controllers
    min_margin[0] = (-grid.line_data[:,5] - grid.init_state[0:grid.num_lines]) / num_controllers

    for i in range(num_controllers):
        buses_in_area = torch.tensor(partition[i], dtype=torch.int)
        controller_list.append(controller_pkg.Split_Constraint_Controller(grid, H, buses_in_area, max_margin, min_margin, batt_cost,
                                                    curt_change_cost, curt_net_cost, bus_slack_cost, line_slack_cost))
    return controller_list


def assign_line_limit_changes(controller_list, line_max_changes, line_min_changes):
    """Assigns line limit change trajectories to each controller.

    Args:
        controller_list (list[Split_Constraint_Controller]): Controllers to update.
        line_max_changes (torch.Tensor): Shape (num_agents, T+H, num_lines); per-step max limit changes per agent.
        line_min_changes (torch.Tensor): Shape (num_agents, T+H, num_lines); per-step min limit changes per agent.

    Raises:
        ValueError: If controller_list length does not match the first dimension of line_max_changes.
    """
    num_agents = line_max_changes.shape[0]
    if not len(controller_list) == num_agents:
        raise ValueError(f"controller list has {len(controller_list)} controllers but line_max_vals has values for {num_agents} agents.")

    with torch.no_grad():
        for i in range(len(controller_list)):
            controller = controller_list[i]
            controller.line_max_change = nn.Parameter(line_max_changes[i].clone())
            controller.line_min_change = nn.Parameter(line_min_changes[i].clone())


def assign_line_limits(controller_list, line_max_vals, line_min_vals):
    """Assigns absolute line limits to each controller.

    Args:
        controller_list (list[Split_Constraint_Controller]): Controllers to update.
        line_max_vals (torch.Tensor): Shape (num_agents, T+H, num_lines); max limits per agent.
        line_min_vals (torch.Tensor): Shape (num_agents, T+H, num_lines); min limits per agent.

    Raises:
        ValueError: If controller_list length does not match the first dimension of line_max_vals.
    """
    num_agents = line_max_vals.shape[0]
    if not len(controller_list) == num_agents:
        raise ValueError(f"controller list has {len(controller_list)} controllers but line_max_vals has values for {num_agents} agents.")

    with torch.no_grad():
        for i in range(len(controller_list)):
            controller = controller_list[i]
            controller.line_max_change = nn.Parameter(line_max_vals.clone())
            controller.line_min_change = nn.Parameter(line_min_vals.clone())


def get_next_action(grid, controller_list, disturbances, t, dQPTH_layer, verbose=False, with_cvxpy=False, update_state=False):
    """Collects and aggregates actions from all split-constraint agents for the next control step.

    Args:
        grid (Grid): Full grid that the controllers act on.
        controller_list (list[Split_Constraint_Controller]): All controllers on the grid.
        disturbances (torch.Tensor): Shape (H, num_buses); power injection noise predictions per bus.
        t (int): Current timestep.
        dQPTH_layer: Differentiable QP solver layer.
        verbose (bool): Unused.
        with_cvxpy (bool): Unused.
        update_state (bool): If True, each controller updates its internal state after solving.

    Returns:
        tuple:
            - pred_actions_curt (torch.Tensor): Shape (H, num_curt); curtailment actions aggregated across agents.
            - pred_actions_batt (torch.Tensor): Shape (H, num_batt); battery actions aggregated across agents.
            - pred_line_max_slacks (torch.Tensor): Shape (num_agents, H, num_lines); line max slack per agent.
            - pred_line_min_slacks (torch.Tensor): Shape (num_agents, H, num_lines); line min slack per agent.
            - pred_bus_slacks (torch.Tensor): Shape (H, 2*(2*num_batt + num_curt)); aggregated bus slacks.
    """
    H = controller_list[0].H
    pred_actions_curt = torch.zeros(H, grid.num_curt)
    pred_actions_batt = torch.zeros(H, grid.num_batt)
    num_agents = len(controller_list)
    pred_bus_slacks = torch.zeros(H, 2*(2*grid.num_batt + grid.num_curt))
    pred_line_max_slacks = torch.zeros(num_agents, H, grid.num_lines)
    pred_line_min_slacks = torch.zeros(num_agents, H, grid.num_lines)

    Q = []
    q = []
    G = []
    non_sparse_G = []
    h = []
    A = []
    b = []
    scipy_Q = []
    scipy_G = []
    scipy_A = []
    
    for i in range(num_agents):
        controller = controller_list[i]
        controller_disturbances = disturbances[:, controller.buses_in_area]
        config = controller.get_matrices_dQPTH(controller_disturbances, t)
        Q.append(config["csc_scaled_Q"])
        q.append(config["q"])
        G.append(config["csc_scaled_G"])
        non_sparse_G.append(config["G"])
        h.append(config["h"])
        A.append(config["csc_scaled_A"])
        b.append(config["b"])
        scipy_Q.append(config["scipy_scaled_Q"])
        scipy_G.append(config["scipy_scaled_G"])
        scipy_A.append(config["scipy_scaled_A"])

    results = dQPTH_layer(Q, q, G, non_sparse_G, h, A, b, scipy_Q, scipy_G, scipy_A, t)
    try:
        for i in range(num_agents):
            controller = controller_list[i]
            z = results[i]
            
            bus_states, line_states, actions_curt, actions_batt, bus_slacks, line_max_slacks, line_min_slacks, objective = \
                controller.interpret_z(z.flatten(), update_state=update_state)

            pred_actions_curt[:,controller.curt_idx] = actions_curt
            pred_actions_batt[:,controller.batt_idx] = actions_batt
            pred_line_max_slacks[i,:,:] = line_max_slacks
            pred_line_min_slacks[i,:,:] = line_min_slacks
            # pred_bus_slacks[:,controller.bus_slack_idx] = bus_slacks
    except Exception as e:
        print(e)
    return pred_actions_curt, pred_actions_batt, pred_line_max_slacks, pred_line_min_slacks, pred_bus_slacks


def get_next_action_base(grid, controller_list, disturbances, t, dQPTH_layer, verbose=False, with_cvxpy=False, update_state=False):
    """Collects and aggregates actions from base controllers for the next control step.

    Same interface as get_next_action but uses Base_Controller objects. Slack outputs are not populated.

    Args:
        grid (Grid): Full grid that the controllers act on.
        controller_list (list[Base_Controller]): All controllers on the grid.
        disturbances (torch.Tensor): Shape (H, num_buses); power injection noise predictions per bus.
        t (int): Current timestep.
        dQPTH_layer: Differentiable QP solver layer.
        verbose (bool): Unused.
        with_cvxpy (bool): Unused.
        update_state (bool): If True, each controller updates its internal state after solving.

    Returns:
        tuple:
            - pred_actions_curt (torch.Tensor): Shape (H, num_curt); curtailment actions aggregated across agents.
            - pred_actions_batt (torch.Tensor): Shape (H, num_batt); battery actions aggregated across agents.
            - None, None, None: Slack outputs not populated for base controllers.
    """
    start_next_action = time.time()
    H = controller_list[0].H
    pred_actions_curt = torch.zeros(H, grid.num_curt)
    pred_actions_batt = torch.zeros(H, grid.num_batt)
    num_agents = len(controller_list)
    pred_bus_slacks = torch.zeros(H, 2*(2*grid.num_batt + grid.num_curt))
    pred_line_max_slacks = torch.zeros(num_agents, H, grid.num_lines)
    pred_line_min_slacks = torch.zeros(num_agents, H, grid.num_lines)

    Q = []
    q = []
    G = []
    non_sparse_G = []
    h = []
    A = []
    b = []
    scipy_Q = []
    scipy_G = []
    scipy_A = []
    
    for i in range(num_agents):
        controller = controller_list[i]
        controller_disturbances = disturbances[:, controller.buses_in_area]
        config = controller.get_matrices_dQPTH(controller_disturbances, t)
        Q.append(config["csc_scaled_Q"])
        q.append(config["q"])
        G.append(config["csc_scaled_G"])
        non_sparse_G.append(config["G"])
        h.append(config["h"])
        A.append(config["csc_scaled_A"])
        b.append(config["b"])
        scipy_Q.append(config["scipy_scaled_Q"])
        scipy_G.append(config["scipy_scaled_G"])
        scipy_A.append(config["scipy_scaled_A"])
    try:
        results = dQPTH_layer(Q, q, G, non_sparse_G, h, A, b, scipy_Q, scipy_G, scipy_A, t)
    except Exception as e:
        print("Following error occurred when calling dQPTH_layer:")
        print(e)
    try:
        for i in range(num_agents):
            controller = controller_list[i]
            z = results[i]
            
            bus_states, line_states, actions_curt, actions_batt, bus_slacks, line_max_slacks, line_min_slacks, objective = \
                controller.interpret_z(z.flatten(), update_state=update_state)

            pred_actions_curt[:,controller.curt_idx] = actions_curt
            pred_actions_batt[:,controller.batt_idx] = actions_batt
    except Exception as e:
        print(e)
    return pred_actions_curt, pred_actions_batt, None, None, None


def reset_states(grid, controller_list):
    """Resets the grid and all controllers to their initial states.

    Args:
        grid (Grid): Full grid that the controllers act on.
        controller_list (list): All controllers on the grid.
    """
    grid.reset_state()
    for controller in controller_list:
        controller.update_bus_state()
        controller.reset_line_state()


def update_controller_bus_states(controller_list):
    """Refreshes bus states for all controllers from the current grid state.

    Args:
        controller_list (list): All controllers on the grid.
    """
    for controller in controller_list:
        controller.update_bus_state()


def is_partition(partition, full_list):
    """Returns True if partition covers full_list exactly with no duplicates.

    Args:
        partition (list of lists): Candidate partition.
        full_list (list): The list being partitioned.

    Returns:
        bool: True if valid partition, False otherwise.
    """
    # Flatten the partition and check if it contains all elements of full_list
    flattened = [item for sublist in partition for item in sublist]
    return sorted(flattened) == sorted(full_list) and len(flattened) == len(set(flattened))


def find_cut_lines(grid, partition):
    """Identifies lines connecting buses in different areas of the partition.

    Args:
        grid (Grid): Grid object.
        partition (list of lists): Partition of buses into areas.

    Returns:
        tuple:
            - cut_lines (list[int]): Indices of cut lines.
            - cut_lines_data (list[tuple]): Each entry is (line_idx, node1, node2, area1_idx, area2_idx).
    """
    # Initialize an empty list to store the indices of lines connecting different areas
    cut_lines = []
    cut_lines_data = []

    # Iterate through each line in the "line_data" array
    for idx, line in enumerate(grid.line_data):
        node1, node2 = int(line[0]), int(line[1])  # Convert to 0-based indexing
        # Check if the nodes belong to different areas
        for area1_idx, area1 in enumerate(partition):
            if node1 in area1:
                for area2_idx, area2 in enumerate(partition):
                    if area1_idx != area2_idx and node2 in area2:
                        cut_lines.append(idx)
                        cut_lines_data.append((idx, node1, node2, area1_idx, area2_idx))
                        break

    return cut_lines, cut_lines_data


def explain_infeasibility(grid, infeasible_idx, violations):
    """Returns a human-readable message describing which constraints were violated.

    Args:
        grid (Grid): Grid object.
        infeasible_idx (list[int]): Indices into the constraint violation vector that were violated.
        violations (list[float]): Violation magnitudes corresponding to each index.

    Returns:
        str: Description of each violated constraint (buses are 0-indexed).
    """
    full_grid_params = grid.get_grid_data()
    full_line_data = full_grid_params["line_data"]
    bus_with_batt = full_grid_params["bus_with_batt"]
    bus_with_gen  = full_grid_params["bus_with_gen"]
    message = "BUSES ARE 0-INDEXED HERE:\n"

    for i in range(len(infeasible_idx)):
        idx_message = ""
        idx = infeasible_idx[i]
        if idx < grid.num_lines:
            line_num = idx
            idx_message += f"Line {line_num} from {full_line_data[line_num, 0]} to {full_line_data[line_num, 1]} "\
                           f"with limit of {grid.line_limits[line_num]} had min violated by {violations[i]}"
        elif idx >= grid.num_lines and idx < 2*grid.num_lines:
            line_num = idx - grid.num_lines
            idx_message += f"Line {line_num} from {full_line_data[line_num, 0]} to {full_line_data[line_num, 1]} "\
                           f"with limit of {grid.line_limits[line_num]} had max violated by {violations[i]}"
        elif idx >= 2*grid.num_lines and idx < 2*grid.num_lines + grid.num_batt:
            batt_num = idx - 2*grid.num_lines
            idx_message += f"Battery {batt_num} at node {bus_with_batt[batt_num]} with min charge limit of "\
                           f"{grid.batt_charge_min_limits[batt_num]} had min violated by {violations[i]}"
        elif idx >= 2*grid.num_lines + grid.num_batt and idx < 2*(grid.num_lines + grid.num_batt):
            batt_num = idx - 2*grid.num_lines - grid.num_batt
            idx_message += f"Battery {batt_num} at node {bus_with_batt[batt_num]} with max charge limit of "\
                           f"{grid.batt_charge_max_limits[batt_num]} had max violated by {violations[i]}"
        elif idx >= 2*(grid.num_lines + grid.num_batt) and idx < 2*(grid.num_lines + grid.num_batt) + grid.num_gen:
            gen_num = idx - 2*(grid.num_lines + grid.num_batt)
            idx_message += f"Curtailable generation {gen_num} at node {bus_with_gen[gen_num]} with min limit of "\
                           f"{grid.curt_min_limits[gen_num]} had min violated by {violations[i]}"
        elif idx >= 2*(grid.num_lines + grid.num_batt) + grid.num_gen and idx < 2*(grid.num_lines + grid.num_batt + grid.num_gen):
            gen_num = idx - 2*(grid.num_lines + grid.num_batt) - grid.num_gen
            idx_message += f"Curtailable generation {gen_num} at node {bus_with_gen[gen_num]} with max limit of "\
                           f"{grid.curt_max_limits[gen_num]} had max violated by {violations[i]}"
        elif idx >= 2*(grid.num_lines + grid.num_batt + grid.num_gen) and idx < 2*(grid.num_lines + grid.num_batt + grid.num_gen) + grid.num_batt:
            batt_num = idx - 2*(grid.num_lines + grid.num_batt + grid.num_gen)
            idx_message += f"Battery {batt_num} at node {bus_with_batt[batt_num]} with min power injection limit of "\
                           f" {grid.batt_power_min_limits[batt_num]} had min violated by {violations[i]}"
        else:
            batt_num = idx - 2*(grid.num_lines + grid.num_batt + grid.num_gen) - grid.num_batt
            idx_message += f"Battery {batt_num} at node {bus_with_batt[batt_num]} with max power injection limit of "\
                           f" {grid.batt_power_max_limits[batt_num]} had max violated by {violations[i]}"

        if not i == len(infeasible_idx) - 1:
            idx_message += "\n"
        message += idx_message

    return message