import sys
sys.path.append("../../")

import numpy as np
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from tqdm import tqdm
import time

import grid_pkg
import controller_pkg
import controller_utils
import dQPTH

torch.set_default_dtype(torch.double)
torch.set_printoptions(threshold=10000)
NUMPY_SEED = 0
TORCH_SEED = 0
np.random.seed(NUMPY_SEED)
torch.manual_seed(TORCH_SEED)

def _make_optimizer(param_groups, optimizer_type, lr):
    if optimizer_type in ('clipped_gd', 'sgd'):
        return optim.SGD(param_groups, lr=lr)
    elif optimizer_type == 'adam':
        return optim.Adam(param_groups, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer_type: {optimizer_type!r}")


def _make_scheduler(optimizer, lr_schedule, schedule_kwargs, total_steps):
    if lr_schedule is None:
        return None
    kw = schedule_kwargs or {}
    if lr_schedule == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, **kw)
    elif lr_schedule == 'step_decay':
        return optim.lr_scheduler.StepLR(optimizer, step_size=kw.get('step_size', 30), gamma=kw.get('gamma', 0.5))
    elif lr_schedule == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                    mode=kw.get('mode', 'min'),
                                                    factor=kw.get('factor', 0.5),
                                                    patience=kw.get('patience', 10))
    else:
        raise ValueError(f"Unknown lr_schedule: {lr_schedule!r}")


def _step_scheduler(scheduler, loss):
    """Step scheduler, passing loss for ReduceLROnPlateau."""
    if scheduler is None:
        return
    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(loss)
    else:
        scheduler.step()


def _project_params(controller_dqp_list, max_target_sum=0, min_target_sum=0):
    """Project line_max/min_change so their sum across agents equals the target."""
    n = len(controller_dqp_list)
    with torch.no_grad():
        max_params = torch.stack([controller.line_max_change.data for controller in controller_dqp_list])
        min_params = torch.stack([controller.line_min_change.data for controller in controller_dqp_list])
        max_correction = (max_params.sum(dim=0) - max_target_sum) / n
        min_correction = (min_params.sum(dim=0) - min_target_sum) / n
        for controller in controller_dqp_list:
            controller.line_max_change.data -= max_correction
            controller.line_min_change.data -= min_correction


# @profile
def get_central_full_traj(grid, T, H, disturbances, batt_cost, curt_change_cost, curt_net_cost, bus_slack_cost, line_slack_cost):
    """Simulates a centralized planner with full look-ahead over T steps.

    Solves a single open-loop planning problem using all T future disturbances at once,
    equivalent to a perfect-information benchmark.

    Args:
        grid (Grid): Grid that the control problem is set up on.
        T (int): Trajectory length (number of control steps).
        H (int): Unused; included for interface consistency.
        disturbances (torch.Tensor): Shape (T+H, num_buses); future power injection noise.
        batt_cost (float): Cost on battery charge deviation from target.
        curt_change_cost (float): Cost on changes in curtailed power per step.
        curt_net_cost (float): Cost on net curtailed power per step.
        bus_slack_cost (float): Penalty on bus constraint violations.
        line_slack_cost (float): Penalty on line flow constraint violations.

    Returns:
        dict: Run results with keys:
            - actions_curt (torch.Tensor): Shape (T, num_curt); curtailment actions.
            - actions_batt (torch.Tensor): Shape (T, num_batt); battery actions.
            - viol_per_step (torch.Tensor): Shape (T,); constraint violation cost at each step.
            - curt_cost_per_step (torch.Tensor): Shape (T,); curtailment cost at each step.
            - batt_cost_per_step (torch.Tensor): Shape (T,); battery cost at each step.
            - total_losses (torch.Tensor): [total, curt, violation] summed over trajectory.
    """
    partition = [[i for i in range(grid.num_buses)]]
    num_agents = 1
    controller_dqp_list = controller_utils.create_split_constraint_controllers(grid, num_agents, partition, T, T, batt_cost, curt_change_cost,
                                                            curt_net_cost, bus_slack_cost, line_slack_cost)
    pool = mp.Pool(processes=num_agents)
    settings = dQPTH.build_settings(solve_type="sparse", qp_solver="gurobi", lin_solver="qdldl", warm_start_from_previous=True)
    dQPTH_layer = dQPTH.dQPTH_layer(settings=settings, pool=pool)

    grid.reset_state()
    for controller in controller_dqp_list:
        controller.zero_line_state()
        controller.update_bus_state()
        controller.reset_params()

    noise = disturbances[0:T]
    t = 0
    start_time = time.time()
    actions_curt, actions_batt, pred_line_max_slacks, pred_line_min_slacks, pred_bus_slacks = \
        controller_utils.get_next_action(grid, controller_dqp_list, noise, t, dQPTH_layer, verbose=False, update_state=False)
    grid.reset_state()
    for controller in controller_dqp_list:
        controller.zero_line_state()
        controller.update_bus_state()
        controller.reset_params()

    viol_per_step = torch.zeros(T)
    curt_cost_per_step = torch.zeros(T)
    batt_cost_per_step = torch.zeros(T)

    for t in range(T):
        action_curt = actions_curt[t]
        action_batt = actions_batt[t]
        grid.update_state(action_curt, action_batt, noise[t,:])

        net_curt = grid.state[grid.state_curt_idx]
        bus_violations = torch.relu(grid.H_x @ grid.state - grid.H_limit)[2*grid.num_lines:]
        batt_charge_violations = bus_violations[:2*grid.num_batt]
        batt_power_violations = bus_violations[2*grid.num_batt:4*grid.num_batt]
        curt_violations = bus_violations[4*grid.num_batt:]
        scaled_batt_charge_violations = batt_charge_violations / torch.concat([grid.target_batt_charges] * 2)
        scaled_batt_power_violations = batt_power_violations / torch.concat([grid.batt_power_max_limits] * 2)
        scaled_curt_violations = curt_violations / torch.concat([grid.curt_max_limits] * 2)
        bus_violations_loss = bus_slack_cost * (torch.sum(scaled_batt_charge_violations ** 2)
                                                + torch.sum(scaled_batt_power_violations ** 2)
                                                + torch.sum(scaled_curt_violations ** 2))
        line_flows = grid.state[grid.state_line_flow_idx]
        line_limits = grid.line_data[:,5]
        line_violations = torch.relu(torch.abs(line_flows) - line_limits)
        scaled_line_violations = line_violations / grid.line_data[:,5]
        line_violations_loss = line_slack_cost * torch.sum(line_violations ** 2)
        batt_charge_deviation = grid.state[grid.state_batt_charge_idx] - grid.target_batt_charges
        batt_loss = batt_cost * torch.sum(batt_charge_deviation ** 2)
        curt_loss = curt_change_cost * torch.sum(action_curt ** 2) + curt_net_cost * torch.sum(net_curt ** 2)
        viol_per_step[t] = bus_violations_loss + line_violations_loss
        curt_cost_per_step[t] = curt_loss
        batt_cost_per_step[t] = batt_loss

    curt_cost_per_step = curt_cost_per_step.detach()
    batt_cost_per_step = batt_cost_per_step.detach()
    viol_per_step = viol_per_step.detach()
    total_loss = torch.sum(curt_cost_per_step) + torch.sum(batt_cost_per_step) + torch.sum(viol_per_step)
    total_losses = torch.tensor([total_loss, torch.sum(curt_cost_per_step), torch.sum(viol_per_step)])
    ckpt = {
        "actions_curt": actions_curt.detach(),
        "actions_batt": actions_batt.detach(),
        "viol_per_step": viol_per_step,
        "curt_cost_per_step": curt_cost_per_step,
        "batt_cost_per_step": batt_cost_per_step,
        "total_losses": total_losses
    }

    pool.close()

    return ckpt


def evaluate_base_on_traj(grid, base_controller_list, dQPTH_layer, test_traj, T, H, batt_cost, curt_change_cost, curt_net_cost, bus_slack_cost, line_slack_cost):
    """Evaluates base controllers on a test trajectory using closed-loop MPC rollout.

    Args:
        grid (Grid): Grid to evaluate on.
        base_controller_list (list[Base_Controller]): Controllers to run.
        dQPTH_layer: Differentiable QP solver layer.
        test_traj (torch.Tensor): Shape (T+H, num_buses); disturbance trajectory.
        T (int): Number of control steps.
        H (int): MPC prediction horizon.
        batt_cost (float): Cost on battery charge deviation from target.
        curt_change_cost (float): Cost on changes in curtailed power per step.
        curt_net_cost (float): Cost on net curtailed power per step.
        bus_slack_cost (float): Penalty on bus constraint violations.
        line_slack_cost (float): Penalty on line flow constraint violations.

    Returns:
        dict: Evaluation results with keys:
            - actions_curt (torch.Tensor): Shape (T, num_curt); curtailment actions taken.
            - actions_batt (torch.Tensor): Shape (T, num_batt); battery actions taken.
            - viol_per_step (torch.Tensor): Shape (T,); line violation cost at each step.
            - curt_cost_per_step (torch.Tensor): Shape (T,); curtailment cost at each step.
            - batt_cost_per_step (torch.Tensor): Shape (T,); battery cost at each step.
            - total_losses (torch.Tensor): [total, economic, violation] summed over trajectory.
    """
    grid.reset_state()
    for controller in base_controller_list:
        controller.update_line_state()
        controller.update_bus_state()
    actions_curt = torch.zeros(T, grid.num_curt)
    actions_batt = torch.zeros(T, grid.num_batt)
    viol_per_step = torch.zeros(T)
    curt_cost_per_step = torch.zeros(T)
    batt_cost_per_step = torch.zeros(T)
    total_loss = 0
    total_econ_loss = 0
    total_viol_loss = 0
    total_train_loss = 0

    for t in range(T):
        for controller in base_controller_list:
            controller.update_line_state()
            controller.update_bus_state()
        noise = test_traj[t:t+H,:]
        pred_actions_curt, pred_actions_batt, pred_line_max_slacks, pred_line_min_slacks, pred_bus_slacks = \
            controller_utils.get_next_action_base(grid, base_controller_list, noise, t, dQPTH_layer, verbose=False, update_state=True)

        action_curt = pred_actions_curt[0]
        action_batt = pred_actions_batt[0]
        grid.update_state(action_curt, action_batt, noise[0,:])

        net_curt = grid.state[grid.state_curt_idx]
        bus_violations = torch.relu(grid.H_x @ grid.state - grid.H_limit)[2*grid.num_lines:]
        line_flows = grid.state[grid.state_line_flow_idx]
        line_limits = grid.line_data[:,5]
        line_violations = torch.relu(torch.abs(line_flows) - line_limits)
        line_violations_loss = line_slack_cost * torch.sum(line_violations ** 2)

        batt_charge_deviation = grid.state[grid.state_batt_charge_idx] - grid.target_batt_charges
        batt_loss = batt_cost * torch.sum(batt_charge_deviation ** 2)
        curt_loss = curt_change_cost * torch.sum(action_curt ** 2) + curt_net_cost * torch.sum(net_curt ** 2)
        total_loss += batt_loss + curt_loss + line_violations_loss
        total_train_loss += batt_loss + curt_loss + line_slack_cost * torch.sum(line_violations ** 2)
        total_econ_loss += batt_loss + curt_loss
        total_viol_loss += line_violations_loss
        viol_per_step[t] = line_violations_loss
        curt_cost_per_step[t] = curt_loss
        batt_cost_per_step[t] = batt_loss

        actions_curt[t] = action_curt
        actions_batt[t] = action_batt

    total_losses = torch.tensor([total_loss.detach(), total_econ_loss.detach(), total_viol_loss.detach()])

    ckpt = {
        "actions_curt": actions_curt.detach(),
        "actions_batt": actions_batt.detach(),
        "viol_per_step": viol_per_step.detach(),
        "curt_cost_per_step": curt_cost_per_step.detach(),
        "batt_cost_per_step": batt_cost_per_step.detach(),
        "total_losses": total_losses
    }

    return ckpt

# @profile
def evaluate_on_traj(grid, controller_dqp_list, dQPTH_layer, test_traj, T, H, batt_cost, curt_change_cost, curt_net_cost, bus_slack_cost, line_slack_cost):
    """Evaluates split-constraint controllers on a test trajectory using closed-loop MPC rollout.

    Args:
        grid (Grid): Grid to evaluate on.
        controller_dqp_list (list[Split_Constraint_Controller]): Controllers to run.
        dQPTH_layer: Differentiable QP solver layer.
        test_traj (torch.Tensor): Shape (T+H, num_buses); disturbance trajectory.
        T (int): Number of control steps.
        H (int): MPC prediction horizon.
        batt_cost (float): Cost on battery charge deviation from target.
        curt_change_cost (float): Cost on changes in curtailed power per step.
        curt_net_cost (float): Cost on net curtailed power per step.
        bus_slack_cost (float): Penalty on bus constraint violations.
        line_slack_cost (float): Penalty on line flow constraint violations.

    Returns:
        dict: Evaluation results with keys:
            - actions_curt (torch.Tensor): Shape (T, num_curt); curtailment actions taken.
            - actions_batt (torch.Tensor): Shape (T, num_batt); battery actions taken.
            - viol_per_step (torch.Tensor): Shape (T,); line violation cost at each step.
            - curt_cost_per_step (torch.Tensor): Shape (T,); curtailment cost at each step.
            - batt_cost_per_step (torch.Tensor): Shape (T,); battery cost at each step.
            - total_losses (torch.Tensor): [total, economic, violation] summed over trajectory.
    """
    grid.reset_state()
    for controller in controller_dqp_list:
        controller.zero_line_state()
        controller.update_bus_state()

    test_total_loss = torch.zeros(3)
    dQPTH_layer.reset_cache()

    actions_curt = torch.zeros(T, grid.num_curt)
    actions_batt = torch.zeros(T, grid.num_batt)
    viol_per_step = torch.zeros(T)
    curt_cost_per_step = torch.zeros(T)
    batt_cost_per_step = torch.zeros(T)
    total_loss = 0
    total_econ_loss = 0
    total_viol_loss = 0
    total_train_loss = 0

    for t in range(T):
        noise = test_traj[t:t+H,:]
        pred_actions_curt, pred_actions_batt, pred_line_max_slacks, pred_line_min_slacks, pred_bus_slacks = \
            controller_utils.get_next_action(grid, controller_dqp_list, noise, t, dQPTH_layer, verbose=False, update_state=True)

        action_curt = pred_actions_curt[0]
        action_batt = pred_actions_batt[0]
        grid.update_state(action_curt, action_batt, noise[0,:])

        net_curt = grid.state[grid.state_curt_idx]
        bus_violations = torch.relu(grid.H_x @ grid.state - grid.H_limit)[2*grid.num_lines:]
        batt_charge_violations = bus_violations[:2*grid.num_batt]
        batt_power_violations = bus_violations[2*grid.num_batt:4*grid.num_batt]
        curt_violations = bus_violations[4*grid.num_batt:]
        scaled_batt_charge_violations = batt_charge_violations / torch.concat([grid.target_batt_charges] * 2)
        scaled_batt_power_violations = batt_power_violations / torch.concat([grid.batt_power_max_limits] * 2)
        scaled_curt_violations = curt_violations / torch.concat([grid.curt_max_limits] * 2)
        bus_violations_loss = bus_slack_cost * (torch.sum(scaled_batt_charge_violations ** 2)
                                                + torch.sum(scaled_batt_power_violations ** 2)
                                                + torch.sum(scaled_curt_violations ** 2))
        line_flows = grid.state[grid.state_line_flow_idx]
        line_limits = grid.line_data[:,5]
        line_violations = torch.relu(torch.abs(line_flows) - line_limits)
        line_violations_loss = line_slack_cost * torch.sum(line_violations ** 2)
        batt_charge_deviation = grid.state[grid.state_batt_charge_idx] - grid.target_batt_charges
        batt_loss = batt_cost * torch.sum(batt_charge_deviation ** 2)
        curt_loss = curt_change_cost * torch.sum(action_curt ** 2) + curt_net_cost * torch.sum(net_curt ** 2)

        # bus violations no longer included after I changed code to not allow slacks to be set on bus constraints
        total_loss += batt_loss + curt_loss + line_violations_loss
        total_train_loss += batt_loss + curt_loss + line_slack_cost * torch.sum(line_violations ** 2)
        total_econ_loss += batt_loss + curt_loss
        total_viol_loss += line_violations_loss
        viol_per_step[t] = line_violations_loss
        curt_cost_per_step[t] = curt_loss
        batt_cost_per_step[t] = batt_loss

    total_losses = torch.tensor([total_loss.detach(), total_econ_loss.detach(), total_viol_loss.detach()])

    ckpt = {
        "actions_curt": actions_curt.detach(),
        "actions_batt": actions_batt.detach(),
        "viol_per_step": viol_per_step.detach(),
        "curt_cost_per_step": curt_cost_per_step.detach(),
        "batt_cost_per_step": batt_cost_per_step.detach(),
        "total_losses": total_losses
    }

    return ckpt


def train_with_proxy(grid, T, H, all_train_traj, batt_cost, curt_change_cost, curt_net_cost, bus_slack_cost, line_slack_cost, epochs, lr,
                     optimizer_type='clipped_gd', max_grad_norm=30, lr_schedule=None, schedule_kwargs=None, batch_size=None):
    """Learns line limit allocations using a proxy (open-loop surrogate) training approach.

    Differentiates through a single T-step open-loop solve rather than T sequential MPC steps,
    providing approximate gradients for the limit allocation parameters.

    Args:
        grid (Grid): Grid to train on.
        T (int): Trajectory length.
        H (int): MPC prediction horizon.
        all_train_traj (torch.Tensor): Shape (num_traj, T+H, num_buses); training disturbance trajectories.
        batt_cost (float): Cost on battery charge deviation from target.
        curt_change_cost (float): Cost on changes in curtailed power per step.
        curt_net_cost (float): Cost on net curtailed power per step.
        bus_slack_cost (float): Penalty on bus constraint violations.
        line_slack_cost (float): Penalty on line flow constraint violations.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        optimizer_type (str): One of 'clipped_gd', 'sgd', or 'adam'.
        max_grad_norm (float): Gradient clipping norm (only applied when optimizer_type is 'clipped_gd').
        lr_schedule (str or None): One of None, 'cosine', 'step_decay', or 'plateau'.
        schedule_kwargs (dict or None): Additional kwargs passed to the lr scheduler.
        batch_size (int or None): Trajectories per gradient step; defaults to all trajectories.

    Returns:
        dict: Training results with keys:
            - line_max_changes (torch.Tensor): Shape (num_agents, T+H, num_lines); learned max limit changes.
            - line_min_changes (torch.Tensor): Shape (num_agents, T+H, num_lines); learned min limit changes.
            - total_losses_per_iter (torch.Tensor): Total loss per epoch.
            - total_econ_losses_per_iter (torch.Tensor): Economic loss per epoch.
            - total_viol_losses_per_iter (torch.Tensor): Violation loss per epoch.
            - total_train_losses_per_iter (torch.Tensor): Training loss per epoch.
            - lr_reduction_epochs (list): (epoch, new_lr) entries whenever learning rate was reduced.
            - run_failed (bool): True if training encountered a failure.
            - fail_message (str): Error message if run_failed is True.
            - all_train_traj (torch.Tensor): Copy of the input training trajectories.
    """
    num_agents = 3
    nodes_1 = [i for i in range(0, 41)] + [42, 112, 113, 114, 116]
    nodes_2 = [i for i in range(43, 69)] + [41, 72]
    nodes_3 = [i for i in range(73, 112)] + [69, 70, 71, 115, 117]
    partition = [nodes_1, nodes_2, nodes_3]

    # surrogate solves all T steps altogether.
    # We set T=H, H=T since this gets us the size-T prediction window while also setting us up to learn limits for T+H steps.
    # TODO: do this in a way that is less hacky, also think about the fact that size-T prediction window uses only T limits, while
    # MPC rollout uses T+H
    controller_dqp_list = controller_utils.create_split_constraint_controllers(grid, num_agents, partition, H, T, batt_cost, curt_change_cost,
                                                            curt_net_cost, bus_slack_cost, line_slack_cost)
    pool = mp.Pool(processes=num_agents)
    settings = dQPTH.build_settings(solve_type="sparse", qp_solver="gurobi", lin_solver="qdldl", warm_start_from_previous=True)
    dQPTH_layer = dQPTH.dQPTH_layer(settings=settings, pool=pool)

    num_traj = all_train_traj.shape[0]
    if batch_size is None:
        batch_size = num_traj
    steps_per_epoch = (num_traj + batch_size - 1) // batch_size

    grid.reset_state()
    for controller in controller_dqp_list:
        controller.zero_line_state()
        controller.update_bus_state()
        controller.reset_params()

    param_groups = [{'params': [controller.line_max_change, controller.line_min_change]} for controller in controller_dqp_list]
    optimizer = _make_optimizer(param_groups, optimizer_type, lr)
    scheduler = _make_scheduler(optimizer, lr_schedule, schedule_kwargs, epochs * steps_per_epoch)

    total_max_margin = torch.zeros(T+H, grid.num_lines)
    total_min_margin = torch.zeros(T+H, grid.num_lines)
    total_max_margin[0] =  grid.line_data[:,5] - grid.init_state[0:grid.num_lines]
    total_min_margin[0] = -grid.line_data[:,5] - grid.init_state[0:grid.num_lines]

    run_failed = False
    fail_message = ""

    total_losses = torch.zeros(epochs)
    total_econ_losses = torch.zeros(epochs)
    total_viol_losses = torch.zeros(epochs)
    total_train_losses = torch.zeros(epochs)
    lr_reduction_epochs = []  # (epoch, batch_start, new_lr) each time plateau fires

    for epoch in tqdm(range(epochs)):
        epoch_total_loss = 0
        epoch_econ_loss = 0
        epoch_viol_loss = 0
        epoch_train_loss = 0
        perm = torch.randperm(num_traj)

        for batch_start in tqdm(range(0, num_traj, batch_size)):
            batch_idx = perm[batch_start:batch_start + batch_size]
            actual_batch_size = len(batch_idx)
            optimizer.zero_grad()
            batch_total_loss = 0

            for traj in batch_idx:
                train_disturbances = all_train_traj[traj]

                total_loss = 0
                total_econ_loss = 0
                total_viol_loss = 0
                total_train_loss = 0
                grid.reset_state()
                for controller in controller_dqp_list:
                    controller.zero_line_state()
                    controller.update_bus_state()

                start_time = time.time()
                pred_actions_curt, pred_actions_batt, pred_line_max_slacks, pred_line_min_slacks, pred_bus_slacks = \
                        controller_utils.get_next_action(grid, controller_dqp_list, train_disturbances[0:T,:], 0, dQPTH_layer, verbose=False, update_state=True)
                for t in range(T):
                    noise = train_disturbances[t]
                    action_curt = pred_actions_curt[t]
                    action_batt = pred_actions_batt[t]
                    grid.update_state(action_curt, action_batt, noise)

                    net_curt = grid.state[grid.state_curt_idx]
                    bus_violations = torch.relu(grid.H_x @ grid.state - grid.H_limit)[2*grid.num_lines:]
                    batt_charge_violations = bus_violations[:2*grid.num_batt]
                    batt_power_violations = bus_violations[2*grid.num_batt:4*grid.num_batt]
                    curt_violations = bus_violations[4*grid.num_batt:]
                    scaled_batt_charge_violations = batt_charge_violations / torch.concat([grid.target_batt_charges] * 2)
                    scaled_batt_power_violations = batt_power_violations / torch.concat([grid.batt_power_max_limits] * 2)
                    scaled_curt_violations = curt_violations / torch.concat([grid.curt_max_limits] * 2)
                    bus_violations_loss = bus_slack_cost * (torch.sum(scaled_batt_charge_violations ** 2)
                                                        + torch.sum(scaled_batt_power_violations ** 2)
                                                        + torch.sum(scaled_curt_violations ** 2))
                    line_flows = grid.state[grid.state_line_flow_idx]
                    line_limits = grid.line_data[:,5]
                    line_violations = torch.relu(torch.abs(line_flows) - line_limits)
                    line_violations_loss = line_slack_cost * torch.sum(line_violations ** 2)
                    batt_charge_deviation = grid.state[grid.state_batt_charge_idx] - grid.target_batt_charges
                    batt_loss = batt_cost * torch.sum(batt_charge_deviation ** 2)
                    curt_loss = curt_change_cost * torch.sum(action_curt ** 2) + curt_net_cost * torch.sum(net_curt ** 2)
                    total_loss += batt_loss + curt_loss + bus_violations_loss + line_violations_loss
                    total_train_loss += batt_loss + curt_loss + bus_violations_loss + line_slack_cost * torch.sum(line_violations ** 2)
                    total_econ_loss += batt_loss + curt_loss
                    total_viol_loss += bus_violations_loss + line_violations_loss
                start_time = time.time()
                (total_loss / actual_batch_size).backward()

                batch_total_loss += total_loss.detach()
                epoch_total_loss += total_loss.detach()
                epoch_econ_loss += total_econ_loss.detach()
                epoch_viol_loss += total_viol_loss.detach()
                epoch_train_loss += total_train_loss.detach()

            if optimizer_type == 'clipped_gd':
                all_params = [controller.line_max_change for controller in controller_dqp_list] + \
                             [controller.line_min_change for controller in controller_dqp_list]
                torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)

            optimizer.step()
            _project_params(controller_dqp_list, max_target_sum=total_max_margin, min_target_sum=total_min_margin)

        prev_lr = optimizer.param_groups[0]['lr']
        _step_scheduler(scheduler, epoch_total_loss / num_traj)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < prev_lr:
            lr_reduction_epochs.append((epoch, new_lr))

        print(f"Epoch {epoch+1}/{epochs}, loss over all trajs={epoch_total_loss.detach():.4f}, viol={epoch_viol_loss.detach():.4f}", flush=True)
        total_losses[epoch] = epoch_total_loss
        total_econ_losses[epoch] = epoch_econ_loss
        total_viol_losses[epoch] = epoch_viol_loss
        total_train_losses[epoch] = epoch_train_loss

    line_max_changes = torch.zeros(num_agents, T+H, grid.num_lines)
    line_min_changes = torch.zeros(num_agents, T+H, grid.num_lines)
    for i in range(num_agents):
        line_max_changes[i] = controller_dqp_list[i].line_max_change
        line_min_changes[i] = controller_dqp_list[i].line_min_change

    ckpt = dict()
    ckpt["line_max_changes"] = line_max_changes
    ckpt["line_min_changes"] = line_min_changes
    ckpt["total_losses_per_iter"] = total_losses
    ckpt["total_econ_losses_per_iter"] = total_econ_losses
    ckpt["total_viol_losses_per_iter"] = total_viol_losses
    ckpt["total_train_losses_per_iter"] = total_train_losses
    ckpt["lr_reduction_epochs"] = lr_reduction_epochs
    ckpt["run_failed"] = run_failed
    ckpt["fail_message"] = fail_message
    ckpt["all_train_traj"] = all_train_traj
    pool.close()
    return ckpt


def train_with_rollout(grid, T, H, all_train_traj, batt_cost, curt_change_cost, curt_net_cost, bus_slack_cost, line_slack_cost, epochs, lr,
                      optimizer_type='clipped_gd', max_grad_norm=30, lr_schedule=None, schedule_kwargs=None, batch_size=None):
    """Learns line limit allocations using closed-loop MPC rollout training.

    Differentiates through T sequential MPC solves per trajectory, providing true
    rollout gradients for the limit allocation parameters.

    Args:
        grid (Grid): Grid to train on.
        T (int): Trajectory length.
        H (int): MPC prediction horizon.
        all_train_traj (torch.Tensor): Shape (num_traj, T+H, num_buses); training disturbance trajectories.
        batt_cost (float): Cost on battery charge deviation from target.
        curt_change_cost (float): Cost on changes in curtailed power per step.
        curt_net_cost (float): Cost on net curtailed power per step.
        bus_slack_cost (float): Penalty on bus constraint violations.
        line_slack_cost (float): Penalty on line flow constraint violations.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        optimizer_type (str): One of 'clipped_gd', 'sgd', or 'adam'.
        max_grad_norm (float): Gradient clipping norm (only applied when optimizer_type is 'clipped_gd').
        lr_schedule (str or None): One of None, 'cosine', 'step_decay', or 'plateau'.
        schedule_kwargs (dict or None): Additional kwargs passed to the lr scheduler.
        batch_size (int or None): Trajectories per gradient step; defaults to all trajectories.

    Returns:
        dict: Training results with keys:
            - line_max_changes (torch.Tensor): Shape (num_agents, T+H, num_lines); learned max limit changes.
            - line_min_changes (torch.Tensor): Shape (num_agents, T+H, num_lines); learned min limit changes.
            - total_losses_per_iter (torch.Tensor): Total loss per epoch.
            - total_econ_losses_per_iter (torch.Tensor): Economic loss per epoch.
            - total_viol_losses_per_iter (torch.Tensor): Violation loss per epoch.
            - total_train_losses_per_iter (torch.Tensor): Training loss per epoch.
            - lr_reduction_epochs (list): (epoch, new_lr) entries whenever learning rate was reduced.
            - run_failed (bool): True if training encountered a failure.
            - fail_message (str): Error message if run_failed is True.
            - all_train_traj (torch.Tensor): Copy of the input training trajectories.
    """
    num_agents = 3
    nodes_1 = [i for i in range(0, 41)] + [42, 112, 113, 114, 116]
    nodes_2 = [i for i in range(43, 69)] + [41, 72]
    nodes_3 = [i for i in range(73, 112)] + [69, 70, 71, 115, 117]
    partition = [nodes_1, nodes_2, nodes_3]

    controller_dqp_list = controller_utils.create_split_constraint_controllers(grid, num_agents, partition, T, H, batt_cost, curt_change_cost,
                                                            curt_net_cost, bus_slack_cost, line_slack_cost)
    pool = mp.Pool(processes=num_agents)
    settings = dQPTH.build_settings(solve_type="sparse", qp_solver="gurobi", lin_solver="qdldl", warm_start_from_previous=True)
    dQPTH_layer = dQPTH.dQPTH_layer(settings=settings, pool=pool)

    num_traj = all_train_traj.shape[0]
    if batch_size is None:
        batch_size = num_traj
    steps_per_epoch = (num_traj + batch_size - 1) // batch_size

    grid.reset_state()
    for controller in controller_dqp_list:
        controller.zero_line_state()
        controller.update_bus_state()
        controller.reset_params()

    param_groups = [{'params': [controller.line_max_change, controller.line_min_change]} for controller in controller_dqp_list]
    optimizer = _make_optimizer(param_groups, optimizer_type, lr)
    scheduler = _make_scheduler(optimizer, lr_schedule, schedule_kwargs, epochs * steps_per_epoch)

    total_max_margin = torch.zeros(T+H, grid.num_lines)
    total_min_margin = torch.zeros(T+H, grid.num_lines)
    total_max_margin[0] =  grid.line_data[:,5] - grid.init_state[0:grid.num_lines]
    total_min_margin[0] = -grid.line_data[:,5] - grid.init_state[0:grid.num_lines]

    run_failed = False
    fail_message = ""

    total_losses = torch.zeros(epochs)
    total_econ_losses = torch.zeros(epochs)
    total_viol_losses = torch.zeros(epochs)
    total_train_losses = torch.zeros(epochs)
    lr_reduction_epochs = []

    for epoch in tqdm(range(epochs)):
        epoch_total_loss = 0
        epoch_econ_loss = 0
        epoch_viol_loss = 0
        epoch_train_loss = 0
        perm = torch.randperm(num_traj)

        for batch_start in tqdm(range(0, num_traj, batch_size)):
            batch_idx = perm[batch_start:batch_start + batch_size]
            actual_batch_size = len(batch_idx)
            optimizer.zero_grad()
            batch_total_loss = 0

            for traj in batch_idx:
                train_disturbances = all_train_traj[traj]

                total_loss = 0
                total_econ_loss = 0
                total_viol_loss = 0
                total_train_loss = 0
                grid.reset_state()
                for controller in controller_dqp_list:
                    controller.zero_line_state()
                    controller.update_bus_state()

                for t in range(T):
                    noise = train_disturbances[t:t+H,:]
                    pred_actions_curt, pred_actions_batt, pred_line_max_slacks, pred_line_min_slacks, pred_bus_slacks = \
                        controller_utils.get_next_action(grid, controller_dqp_list, noise, t, dQPTH_layer, verbose=False, update_state=True)
                    action_curt = pred_actions_curt[0]
                    action_batt = pred_actions_batt[0]
                    grid.update_state(action_curt, action_batt, noise[0])

                    net_curt = grid.state[grid.state_curt_idx]
                    bus_violations = torch.relu(grid.H_x @ grid.state - grid.H_limit)[2*grid.num_lines:]
                    batt_charge_violations = bus_violations[:2*grid.num_batt]
                    batt_power_violations = bus_violations[2*grid.num_batt:4*grid.num_batt]
                    curt_violations = bus_violations[4*grid.num_batt:]
                    scaled_batt_charge_violations = batt_charge_violations / torch.concat([grid.target_batt_charges] * 2)
                    scaled_batt_power_violations = batt_power_violations / torch.concat([grid.batt_power_max_limits] * 2)
                    scaled_curt_violations = curt_violations / torch.concat([grid.curt_max_limits] * 2)
                    bus_violations_loss = bus_slack_cost * (torch.sum(scaled_batt_charge_violations ** 2)
                                                        + torch.sum(scaled_batt_power_violations ** 2)
                                                        + torch.sum(scaled_curt_violations ** 2))
                    line_flows = grid.state[grid.state_line_flow_idx]
                    line_limits = grid.line_data[:,5]
                    line_violations = torch.relu(torch.abs(line_flows) - line_limits)
                    line_violations_loss = line_slack_cost * torch.sum(line_violations ** 2)
                    batt_charge_deviation = grid.state[grid.state_batt_charge_idx] - grid.target_batt_charges
                    batt_loss = batt_cost * torch.sum(batt_charge_deviation ** 2)
                    curt_loss = curt_change_cost * torch.sum(action_curt ** 2) + curt_net_cost * torch.sum(net_curt ** 2)
                    total_loss += batt_loss + curt_loss + bus_violations_loss + line_violations_loss
                    total_train_loss += batt_loss + curt_loss + bus_violations_loss + line_slack_cost * torch.sum(line_violations ** 2)
                    total_econ_loss += batt_loss + curt_loss
                    total_viol_loss += bus_violations_loss + line_violations_loss

                (total_loss / actual_batch_size).backward()

                batch_total_loss += total_loss.detach()
                epoch_total_loss += total_loss.detach()
                epoch_econ_loss += total_econ_loss.detach()
                epoch_viol_loss += total_viol_loss.detach()
                epoch_train_loss += total_train_loss.detach()

            if optimizer_type == 'clipped_gd':
                all_params = [controller.line_max_change for controller in controller_dqp_list] + \
                             [controller.line_min_change for controller in controller_dqp_list]
                torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)

            optimizer.step()
            _project_params(controller_dqp_list, max_target_sum=total_max_margin, min_target_sum=total_min_margin)

        prev_lr = optimizer.param_groups[0]['lr']
        _step_scheduler(scheduler, epoch_total_loss / num_traj)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < prev_lr:
            lr_reduction_epochs.append((epoch, new_lr))

        print(f"Epoch {epoch+1}/{epochs}, loss over all trajs={epoch_total_loss.detach():.4f}, viol={epoch_viol_loss.detach():.4f}", flush=True)
        total_losses[epoch] = epoch_total_loss
        total_econ_losses[epoch] = epoch_econ_loss
        total_viol_losses[epoch] = epoch_viol_loss
        total_train_losses[epoch] = epoch_train_loss

    line_max_changes = torch.zeros(num_agents, T+H, grid.num_lines)
    line_min_changes = torch.zeros(num_agents, T+H, grid.num_lines)
    for i in range(num_agents):
        line_max_changes[i] = controller_dqp_list[i].line_max_change
        line_min_changes[i] = controller_dqp_list[i].line_min_change

    ckpt = dict()
    ckpt["line_max_changes"] = line_max_changes
    ckpt["line_min_changes"] = line_min_changes
    ckpt["total_losses_per_iter"] = total_losses
    ckpt["total_econ_losses_per_iter"] = total_econ_losses
    ckpt["total_viol_losses_per_iter"] = total_viol_losses
    ckpt["total_train_losses_per_iter"] = total_train_losses
    ckpt["lr_reduction_epochs"] = lr_reduction_epochs
    ckpt["run_failed"] = run_failed
    ckpt["fail_message"] = fail_message
    ckpt["all_train_traj"] = all_train_traj
    pool.close()
    return ckpt