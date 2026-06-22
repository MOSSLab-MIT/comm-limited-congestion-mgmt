import numpy as np
import torch
import torch.nn as nn

import qpth
import time

torch.set_default_dtype(torch.double)
np.random.seed(0)
torch.manual_seed(0)

class Base_Controller(nn.Module):
    """Controller for a designated grid area; considers only buses and lines within that area.

    Attributes:
        grid (Grid): Full grid that the agent runs on.
        buses_in_area (list): Bus numbers in this controller's area.
        lines_in_area (list): Line indices in this controller's area.
        bus_with_curt (torch.Tensor): Bus numbers with curtailable generation.
        bus_with_batt (torch.Tensor): Bus numbers with a battery.
        num_buses (int): Number of buses in area.
        num_lines (int): Number of lines in area.
        num_curt (int): Number of curtailable generators in area.
        num_batt (int): Number of batteries in area.
        curt_idx (torch.Tensor): Indices for curtailment buses within full grid arrays.
        batt_idx (torch.Tensor): Indices for battery buses within full grid arrays.
        delta_t (float): Time between grid updates.
        state_batt_charge_idx (torch.Tensor): Indices for battery charges in state vector.
        state_curt_idx (torch.Tensor): Indices for curtailment in state vector.
        H_x (torch.Tensor): Matrix for bus state inequality constraints, H_x @ x <= H_limit.
        H_limit (torch.Tensor): RHS vector for bus state inequality constraints.
        batt_cost (float): Weight on battery charge deviation from target.
        curt_net_cost (float): Weight on net curtailment.
        curt_change_cost (float): Weight on curtailment changes.
        bus_slack_cost (float): Penalty on bus slack variables.
        line_slack_cost (float): Penalty on line slack variables.
    """

    def __init__(self, grid, H, buses_in_area, batt_cost, curt_change_cost, curt_net_cost, bus_slack_cost, line_slack_cost,
                 eps=1e-8):
        """
        Args:
            grid (Grid): Full grid that the agent runs on.
            H (int): MPC prediction horizon length.
            buses_in_area (torch.Tensor): Bus numbers in this controller's designated area.
            batt_cost (float): Cost on battery charge deviation from target. Must be non-negative.
            curt_change_cost (float): Cost on changes in curtailed power. Must be non-negative.
            curt_net_cost (float): Cost on net curtailed power. Must be non-negative.
            bus_slack_cost (float): Bus slack penalty. Must be non-negative.
            line_slack_cost (float): Line slack penalty. Must be non-negative.
            eps (float): Regularization coefficient ensuring the QP is positive definite.
        """
        super().__init__()
        self.grid = grid
        self.H = H
        self.buses_in_area = buses_in_area
        self.eps = eps

        if batt_cost < 0:
            raise ValueError(f"batt_cost {batt_cost} is negative.")
        if curt_change_cost < 0:
            raise ValueError(f"curt_change_cost {curt_change_cost} is negative.")
        if curt_net_cost < 0:
            raise ValueError(f"curt_net_cost {curt_net_cost} is negative.")
        if bus_slack_cost < 0:
            raise ValueError(f"bus_slack_cost {bus_slack_cost} is negative.")
        if line_slack_cost < 0:
            raise ValueError(f"line_slack_cost {line_slack_cost} is negative.")

        self.batt_cost = batt_cost
        self.curt_change_cost = curt_change_cost
        self.curt_net_cost = curt_net_cost
        self.bus_slack_cost = bus_slack_cost
        self.line_slack_cost = line_slack_cost

        # ---------------- Determine which grid components belong in this controller's region ----------------
        lines_in_area = list()
        for i in range(self.grid.num_lines):
            if torch.round(self.grid.line_data[i,0]) in self.buses_in_area: # or torch.round(self.grid.line_data[i,1]) in self.buses_in_area:
                lines_in_area.append(i)
        self.num_lines = len(lines_in_area)
        self.lines_in_area = torch.tensor(lines_in_area, dtype=torch.int)
        self.line_state = self.grid.state[self.lines_in_area]

        # we need to know the particular indices to correctly construct array slices
        curt_idx = list()
        bus_with_curt = list()
        for i in range(self.grid.num_curt):
            if self.grid.bus_with_curt[i] in self.buses_in_area:
                curt_idx.append(i)
                bus_with_curt.append(self.grid.bus_with_curt[i])
        self.curt_idx = torch.tensor(curt_idx, dtype=torch.int)
        self.bus_with_curt = torch.tensor(bus_with_curt, dtype=torch.int)

        batt_idx = list()
        bus_with_batt = list()
        for i in range(self.grid.num_batt):
            if self.grid.bus_with_batt[i] in self.buses_in_area:
                batt_idx.append(i)
                bus_with_batt.append(self.grid.bus_with_batt[i])
        self.batt_idx = torch.tensor(batt_idx, dtype=torch.int)
        self.bus_with_batt = torch.tensor(bus_with_batt, dtype=torch.int)

        self.num_buses = len(buses_in_area)
        self.num_lines = len(lines_in_area)
        self.num_curt = len(bus_with_curt)
        self.num_batt = len(bus_with_batt)
        self.state_dim = self.num_lines + self.num_curt + 2 * self.num_batt
        self.action_dim = self.num_curt + self.num_batt
        self.bus_state_dim = self.num_curt + 2 * self.num_batt
        self.bus_ineq_dim = 2 * self.bus_state_dim
        self.line_state_dim = self.num_lines
        self.line_max_slack_dim = self.line_state_dim
        self.line_min_slack_dim = self.line_state_dim
        self.target_charges = self.grid.target_batt_charges[self.batt_idx]

        self.grid_state_full_state_idx = torch.tensor(lines_in_area + [x + self.grid.num_lines for x in batt_idx]\
                                    + [x + self.grid.num_lines + self.grid.num_batt for x in batt_idx]\
                                    + [x + self.grid.num_lines + 2*self.grid.num_batt for x in curt_idx], dtype=torch.int)
        self.grid_state_bus_state_idx = torch.tensor([x + self.grid.num_lines for x in batt_idx]\
                                                    + [x + self.grid.num_lines + self.grid.num_batt for x in batt_idx]\
                                                    + [x + self.grid.num_lines + 2*self.grid.num_batt for x in curt_idx], dtype=torch.int)
        self.grid_state_line_state_idx = self.lines_in_area
        self.bus_state_idx = torch.tensor([x for x in batt_idx]\
                                        + [x + self.grid.num_batt for x in batt_idx]\
                                        + [x + 2*self.grid.num_batt for x in curt_idx], dtype=torch.int)
        self.bus_state_batt_charge_idx = torch.tensor([x for x in range(self.num_batt)], dtype=torch.int)
        self.bus_state_curt_idx = torch.tensor([x for x in range(2*self.num_batt, 2*self.num_batt + self.num_curt)], dtype=torch.int)
        # controller's state is whatever the grid's current state is
        self.state = self.grid.state[self.grid_state_full_state_idx]
        self.bus_state = self.grid.state[self.grid_state_bus_state_idx]

        self.bus_slack_idx = torch.tensor([2*x+i for x in batt_idx for i in (0,1)] + \
                                        [2*self.grid.num_batt + 2*x+i for x in batt_idx for i in (0,1)] + \
                                        [4*self.grid.num_batt + 2*x+i for x in curt_idx for i in (0,1)], dtype=torch.int)

        self.batt_charge_min_limits = self.grid.batt_charge_min_limits[self.batt_idx]
        self.batt_charge_max_limits = self.grid.batt_charge_max_limits[self.batt_idx]
        self.batt_power_min_limits = self.grid.batt_power_min_limits[self.batt_idx]
        self.batt_power_max_limits = self.grid.batt_power_max_limits[self.batt_idx]
        self.curt_min_limits = self.grid.curt_min_limits[self.curt_idx]
        self.curt_max_limits = self.grid.curt_max_limits[self.curt_idx]
        # ------------------------------------------------------------------------------------------------

        # ------------------------ Construct dynamics and constraints tensors ------------------------
        self.delta_t = self.grid.delta_t

        self.bus_A = self.grid.A.index_select(0, self.grid_state_bus_state_idx).index_select(1, self.grid_state_bus_state_idx)
        self.bus_B_curt = self.grid.B_curt.index_select(0, self.grid_state_bus_state_idx).index_select(1, self.curt_idx)
        self.bus_B_batt = self.grid.B_batt.index_select(0, self.grid_state_bus_state_idx).index_select(1, self.batt_idx)
        # self.bus_B_noise = self.grid.B_noise.index_select(0, self.grid_state_bus_state_idx).index_select(1, self.buses_in_area)

        # line flow constraints are enforced by split constraints, so we don't include them here
        nB = self.num_batt
        nC = self.num_curt
        rows = 4*nB + 2*nC
        cols = 2*nB + nC
        self.H_x = torch.zeros((rows, cols))
        self.H_x[0:nB,              0:nB]         = -torch.eye(nB)
        self.H_x[nB:2*nB,           0:nB]         =  torch.eye(nB)
        self.H_x[2*nB:3*nB,         nB:2*nB]      = -torch.eye(nB)
        self.H_x[3*nB:4*nB,         nB:2*nB]      =  torch.eye(nB)
        self.H_x[4*nB:4*nB+nC,      2*nB:2*nB+nC] = -torch.eye(nC)
        self.H_x[4*nB+nC:4*nB+2*nC, 2*nB:2*nB+nC] =  torch.eye(nC)
        
        self.H_limit = torch.cat([-self.batt_charge_min_limits, self.batt_charge_max_limits, -self.batt_power_min_limits,
                                   self.batt_power_max_limits, -self.curt_min_limits, self.curt_max_limits])
        self.line_B_curt = self.grid.B_curt[self.lines_in_area.unsqueeze(1),self.curt_idx.unsqueeze(0)]
        self.line_B_batt = self.grid.B_batt[self.lines_in_area.unsqueeze(1),self.batt_idx.unsqueeze(0)]
        self.line_B_noise = self.grid.B_noise[self.lines_in_area.unsqueeze(1),self.buses_in_area.unsqueeze(0)]
        self.line_upper_limits = self.grid.line_data[self.lines_in_area,5]
        self.line_lower_limits = -self.grid.line_data[self.lines_in_area,5]


    def init_matrices(self):
        # ----------------------------------------------------------------------------------------
        # aggregate vector is bus state, bus slack, line state, line max slack, line min slack, batt action, curt action
        self.z_dim = self.H * (self.bus_state_dim + self.line_state_dim + self.line_max_slack_dim \
                            + self.line_min_slack_dim + self.num_batt + self.num_curt)
        self.eq_dim = self.H * (self.bus_state_dim + self.line_state_dim)
        self.ineq_dim = self.H * (self.bus_ineq_dim + self.line_max_slack_dim + self.line_min_slack_dim)

        true_cost_vec = torch.zeros(self.z_dim)
        target_vec = torch.zeros(self.z_dim)

        self.line_state_start_idx = self.H * self.bus_state_dim
        self.line_max_slack_start_idx = self.line_state_start_idx + self.H * self.line_state_dim
        self.line_min_slack_start_idx = self.line_max_slack_start_idx + self.H * self.line_state_dim

        self.action_batt_start_idx = self.line_min_slack_start_idx + self.H * self.line_state_dim
        self.action_curt_start_idx = self.action_batt_start_idx + self.H * self.num_batt

        # -------- Create Q and q matrices --------
        for i in range(self.H):
            # battery charge targets and costs
            idx = i * self.bus_state_dim
            true_cost_vec[idx:idx+self.num_batt] = self.batt_cost * torch.ones(self.num_batt)
            target_vec[idx:idx+self.num_batt] = self.grid.target_batt_charges[self.batt_idx]

            # net curtailment costs
            idx = i * self.bus_state_dim + 2 * self.num_batt
            true_cost_vec[idx:idx+self.num_curt] = self.curt_net_cost * torch.ones(self.num_curt)

        # curt action costs
        idx = self.action_curt_start_idx
        true_cost_vec[idx:idx+self.H*self.num_curt] = self.curt_change_cost * torch.ones(self.H*self.num_curt)

        # line max slack costs
        idx = self.line_max_slack_start_idx
        true_cost_vec[idx:idx+self.H*self.line_max_slack_dim] = self.line_slack_cost * torch.ones(self.H*self.line_max_slack_dim)

        # line min slack costs
        idx = self.line_min_slack_start_idx
        true_cost_vec[idx:idx+self.H*self.line_min_slack_dim] = self.line_slack_cost * torch.ones(self.H*self.line_min_slack_dim)

        self.true_cost_vec = true_cost_vec

        # ----------------------------------------

        # -------- Create A matrix --------
        self.A = torch.zeros(self.eq_dim, self.z_dim)
        # evolution of bus states
        for i in range(self.H):
            block_row = torch.zeros(self.bus_state_dim, self.z_dim)
            action_batt_idx = i * self.num_batt + self.action_batt_start_idx
            action_curt_idx = i * self.num_curt + self.action_curt_start_idx

            if not i == 0:
                block_row[:,(i-1)*self.bus_state_dim:i*self.bus_state_dim] = -self.bus_A
            block_row[:,i*self.bus_state_dim:(i+1)*self.bus_state_dim] = torch.eye(self.bus_state_dim)
            block_row[:,action_batt_idx:action_batt_idx+self.num_batt] = -self.bus_B_batt
            block_row[:,action_curt_idx:action_curt_idx+self.num_curt] = -self.bus_B_curt

            row_idx = i * self.bus_state_dim
            self.A[row_idx:row_idx+self.bus_state_dim] = block_row

        # evolution of line states
        for i in range(self.H):
            action_batt_idx = i * self.num_batt + self.action_batt_start_idx
            action_curt_idx = i * self.num_curt + self.action_curt_start_idx
            block_row = torch.zeros(self.line_state_dim, self.z_dim)

            idx = i * self.line_state_dim + self.line_state_start_idx
            if not i == 0:
                block_row[:,idx-self.line_state_dim:idx] = -torch.eye(self.line_state_dim)
            block_row[:,idx:idx+self.line_state_dim] = torch.eye(self.line_state_dim)
            block_row[:,action_batt_idx:action_batt_idx+self.num_batt] = -self.line_B_batt
            block_row[:,action_curt_idx:action_curt_idx+self.num_curt] = -self.line_B_curt

            row_idx = i * self.line_state_dim + self.H * self.bus_state_dim
            self.A[row_idx:row_idx+self.line_state_dim] = block_row
        # --------------------------------

        # -------- Create G matrix --------
        self.G = torch.zeros(self.ineq_dim, self.z_dim)
        for i in range(self.H):
            block_row = torch.zeros(self.bus_ineq_dim, self.z_dim)
            block_row[:,i*self.bus_state_dim:(i+1)*self.bus_state_dim] = self.H_x

            row_idx = i * self.bus_ineq_dim
            self.G[row_idx:row_idx+self.bus_ineq_dim] = block_row

        for i in range(self.H):
            block_row = torch.zeros(self.line_state_dim, self.z_dim)
            line_state_idx = i * self.line_state_dim + self.line_state_start_idx
            line_max_slack_idx = i * self.line_max_slack_dim + self.line_max_slack_start_idx

            block_row[:,line_state_idx:line_state_idx+self.line_state_dim] = torch.eye(self.line_state_dim)
            block_row[:,line_max_slack_idx:line_max_slack_idx+self.line_state_dim] = -torch.eye(self.line_max_slack_dim)

            row_idx = i * self.line_state_dim + self.H * self.bus_ineq_dim
            self.G[row_idx:row_idx+self.line_state_dim] = block_row

        for i in range(self.H):
            block_row = torch.zeros(self.line_state_dim, self.z_dim)
            line_state_idx = i * self.line_state_dim + self.line_state_start_idx
            line_min_slack_idx = i * self.line_min_slack_dim + self.line_min_slack_start_idx

            block_row[:,line_state_idx:line_state_idx+self.line_state_dim] = -torch.eye(self.line_state_dim)
            block_row[:,line_min_slack_idx:line_min_slack_idx+self.line_state_dim] = -torch.eye(self.line_min_slack_dim)

            row_idx = i * self.line_state_dim + self.H * (self.bus_ineq_dim + self.line_state_dim)
            self.G[row_idx:row_idx+self.line_state_dim] = block_row
        # ----------------------------------------
        # self.nBatch = 1
        # Q_, _ = qpth.util.expandParam(self.Q, self.nBatch, 3)
        # G_, _ = qpth.util.expandParam(self.G, self.nBatch, 3)
        # A_, _ = qpth.util.expandParam(self.A, self.nBatch, 3)
        # ------------ Construct scaling matrices ------------
        scale_vec = torch.zeros(self.z_dim)
        # bus states
        for i in range(self.H):
            # battery charge 
            idx = i * self.bus_state_dim
            scale_vec[idx:idx+self.num_batt] = self.grid.target_batt_charges[self.batt_idx]

            # battery power injection
            idx = i * self.bus_state_dim + self.num_batt
            scale_vec[idx:idx+self.num_batt] = self.grid.batt_power_max_limits[self.batt_idx]

            # curtailment
            idx = i * self.bus_state_dim + 2 * self.num_batt
            scale_vec[idx:idx+self.num_curt] = self.grid.curt_max_limits[self.curt_idx]

        # line state
        for i in range(self.H):
            idx = self.line_state_start_idx + i * self.line_state_dim
            scale_vec[idx:idx+self.line_state_dim] = self.grid.line_data[self.lines_in_area,5]

        # line max slack
        for i in range(self.H):
            idx = self.line_max_slack_start_idx + i * self.line_max_slack_dim
            scale_vec[idx:idx+self.line_max_slack_dim] = self.grid.line_data[self.lines_in_area,5]
        self.line_max_slack_scale_vec = self.grid.line_data[:,5]

        # line min slack
        for i in range(self.H):
            idx = self.line_min_slack_start_idx + i * self.line_min_slack_dim
            scale_vec[idx:idx+self.line_min_slack_dim] = self.grid.line_data[self.lines_in_area,5]
        self.line_min_slack_scale_vec = self.grid.line_data[:,5]

        # batt action
        for i in range(self.H):
            idx = self.action_batt_start_idx + i * self.num_batt
            scale_vec[idx:idx+self.num_batt] = self.grid.batt_power_max_limits[self.batt_idx]

        # curt action
        for i in range(self.H):
            idx = self.action_curt_start_idx + i * self.num_curt
            scale_vec[idx:idx+self.num_curt] = self.grid.curt_max_limits[self.curt_idx]

        self.scale_vec = scale_vec

        # D = diag(scale_vec)
        AD = torch.mul(self.A, scale_vec)
        E_vec = torch.zeros(self.eq_dim)
        eps = 1e-6
        for i in range(self.eq_dim):
            E_vec[i] = 1 / (torch.linalg.norm(AD[i]) + eps)
        
        GD = torch.mul(self.G, scale_vec)
        F_vec = torch.zeros(self.ineq_dim)
        eps = 1e-6
        for i in range(self.ineq_dim):
            F_vec[i] = 1 / (torch.linalg.norm(GD[i]) + eps)
        
        self.E_vec = E_vec
        self.F_vec = F_vec
        raw_q = 2 * scale_vec * true_cost_vec * scale_vec

        cost_scale = torch.max(torch.abs(raw_q))
        
        ridge_eps = 1e-8 * cost_scale
        
        q_vec = ((2 * scale_vec * true_cost_vec * scale_vec) + ridge_eps * torch.ones(self.z_dim)) / cost_scale
        self.q = -2 * torch.diag(true_cost_vec) @ target_vec / cost_scale   # slack and ridge cost have linear term 0
        self.const = target_vec @ torch.diag(true_cost_vec) @ target_vec / cost_scale
        
        self.q_vec = q_vec
        self.scaled_Q = torch.diag(q_vec)
        
        self.scaled_q = scale_vec * self.q 
    
        self.scaled_A = E_vec.unsqueeze(-1) * AD
        self.scaled_G = F_vec.unsqueeze(-1) * GD
        self.nBatch = 1
        Q_, _ = qpth.util.expandParam(self.scaled_Q, self.nBatch, 3)
        G_, _ = qpth.util.expandParam(self.scaled_G, self.nBatch, 3)
        A_, _ = qpth.util.expandParam(self.scaled_A, self.nBatch, 3)

        self.csc_scaled_Q = self.scaled_Q.to_sparse_csc()
        self.csc_scaled_G = self.scaled_G.to_sparse_csc()
        self.csc_scaled_A = self.scaled_A.to_sparse_csc()
        self.scipy_scaled_Q = csc_torch_to_scipy(self.csc_scaled_Q)
        self.scipy_scaled_G = csc_torch_to_scipy(self.csc_scaled_G)
        self.scipy_scaled_A = csc_torch_to_scipy(self.csc_scaled_A)


    def update_line_state(self):
        """Fetches line state from the overall Grid object."""
        self.line_state = self.grid.state[self.grid_state_line_state_idx]


    def update_bus_state(self):
        """Fetches bus state from the overall Grid object."""
        self.bus_state = self.grid.state[self.grid_state_bus_state_idx]


    def get_matrices_dQPTH(self, disturbances, t):
        """Returns the QP matrices for this controller given current disturbance predictions.

        Args:
            disturbances (torch.Tensor): Shape (H, num_buses); power injection noise predictions per bus.
            t (int): Current timestep.

        Returns:
            dict: QP matrices with keys: csc_scaled_Q, csc_scaled_G, csc_scaled_A, scipy_scaled_Q,
                scipy_scaled_G, scipy_scaled_A, Q, q, A, b, G, h, const.
        """
        b = torch.zeros(self.eq_dim)
        b[0:self.bus_state_dim] = self.bus_A @ self.bus_state

        for i in range(self.H):
            row_idx = i * self.line_state_dim + self.H * self.bus_state_dim
            b[row_idx:row_idx+self.line_state_dim] = self.line_B_noise @ disturbances[i]

        row_idx = self.H * self.bus_state_dim
        b[row_idx:row_idx+self.line_state_dim] = self.line_state

        h = torch.zeros(self.ineq_dim)
        for i in range(self.H):
            row_idx = i * self.bus_ineq_dim
            h[row_idx:row_idx+self.bus_ineq_dim] = self.H_limit

            row_idx = i * self.line_state_dim + self.H * self.bus_ineq_dim
            h[row_idx:row_idx+self.line_state_dim] = self.line_upper_limits

            row_idx = i * self.line_state_dim + self.H * (self.bus_ineq_dim + self.line_state_dim)
            h[row_idx:row_idx+self.line_state_dim] = -self.line_lower_limits

        dQPTH_config = {
            "csc_scaled_Q": self.csc_scaled_Q,
            "csc_scaled_G": self.csc_scaled_G,
            "csc_scaled_A": self.csc_scaled_A,
            "scipy_scaled_Q": self.scipy_scaled_Q,
            "scipy_scaled_G": self.scipy_scaled_G,
            "scipy_scaled_A": self.scipy_scaled_A,
            "Q": self.scaled_Q,
            "q": self.scaled_q,
            "A": self.scaled_A,
            "b": self.E_vec * b,
            "G": self.scaled_G,
            "h": self.F_vec * h,
            "const": self.const,
        }

        return dQPTH_config
    

    def interpret_z(self, z, update_state=False):
        norm_squared = torch.sum(z**2)
        z *= self.scale_vec
        bus_states = torch.zeros(self.H, self.bus_state_dim)
        for i in range(self.H):
            bus_states[i] = z[i*self.bus_state_dim:(i+1)*self.bus_state_dim]

        line_states = torch.zeros(self.H, self.line_state_dim)
        for i in range(self.H):
            line_states[i] = z[i*self.line_state_dim+self.line_state_start_idx:(i+1)*self.line_state_dim+self.line_state_start_idx]

        actions_curt = torch.zeros(self.H, self.num_curt)
        for i in range(self.H):
            actions_curt[i] = z[i*self.num_curt+self.action_curt_start_idx:(i+1)*self.num_curt+self.action_curt_start_idx]

        actions_batt = torch.zeros(self.H, self.num_batt)
        for i in range(self.H):
            actions_batt[i] = z[i*self.num_batt+self.action_batt_start_idx:(i+1)*self.num_batt+self.action_batt_start_idx]

        line_max_slacks = torch.zeros(self.H, self.line_max_slack_dim)
        for i in range(self.H):
            line_max_slacks[i] = z[i*self.line_max_slack_dim+self.line_max_slack_start_idx:(i+1)*self.line_max_slack_dim+self.line_max_slack_start_idx]

        line_min_slacks = torch.zeros(self.H, self.line_min_slack_dim)
        for i in range(self.H):
            line_min_slacks[i] = z[i*self.line_min_slack_dim+self.line_min_slack_start_idx:(i+1)*self.line_min_slack_dim+self.line_min_slack_start_idx]

        self.target_charges = self.grid.target_batt_charges[self.batt_idx]
        objective = 0
        batt_obj = 0
        curt_change_obj = 0
        curt_net_obj = 0
        line_slack_obj = 0
        with torch.no_grad():
            for i in range(self.H):
                # print(actions_curt[i])
                objective += self.batt_cost * torch.sum((bus_states[i][self.bus_state_batt_charge_idx] - self.target_charges)**2)
                batt_obj += self.batt_cost * torch.sum((bus_states[i][self.bus_state_batt_charge_idx] - self.target_charges)**2)
                # print(f"i = {i}, {(bus_states[i][self.bus_state_batt_charge_idx] - self.target_charges) / self.target_charges}")
                objective += self.curt_change_cost * torch.sum(actions_curt[i]**2)
                curt_change_obj += self.curt_change_cost * torch.sum(actions_curt[i]**2)
                objective += self.curt_net_cost * torch.sum(bus_states[i][self.bus_state_curt_idx]**2)
                curt_net_obj += self.curt_net_cost * torch.sum(bus_states[i][self.bus_state_curt_idx]**2)
                objective += self.line_slack_cost * torch.sum(line_max_slacks[i]**2)
                objective += self.line_slack_cost * torch.sum(line_min_slacks[i]**2)
            objective += self.eps * torch.sum(norm_squared ** 2)

        if update_state:
            self.bus_state = bus_states[0]
            self.line_state = line_states[0]

        return bus_states, line_states, actions_curt, actions_batt, None, line_max_slacks, line_min_slacks, objective

from enum import Enum

class QPSolvers(Enum):
    PDIPM_BATCHED = 1
    CVXPY = 2

from scipy.sparse import csc_matrix,csr_matrix,coo_matrix
def csc_torch_to_scipy(A):
    return csc_matrix((A.values().detach().numpy(),A.row_indices().numpy(),A.ccol_indices().numpy()),shape=A.size())

class Split_Constraint_Controller(Base_Controller):
    """Controller using the split-constraint framework; considers all grid lines, not just those in its area.

    Attributes:
        grid (Grid): Full grid that the agent runs on.
        buses_in_area (list): Bus numbers in this controller's area.
        lines_in_area (list): Line indices in this controller's area.
        bus_with_curt (torch.Tensor): Bus numbers with curtailable generation.
        bus_with_batt (torch.Tensor): Bus numbers with a battery.
        num_buses (int): Number of buses in area.
        num_lines (int): Total number of lines in the full grid (split constraints cover all lines).
        num_curt (int): Number of curtailable generators in area.
        num_batt (int): Number of batteries in area.
        curt_idx (torch.Tensor): Indices for curtailment buses within full grid arrays.
        batt_idx (torch.Tensor): Indices for battery buses within full grid arrays.
        delta_t (float): Time between grid updates.
        line_max_change (nn.Parameter): Learnable per-step max line limit changes over the trajectory.
        line_min_change (nn.Parameter): Learnable per-step min line limit changes over the trajectory.
        H_x (torch.Tensor): Matrix for bus state inequality constraints, H_x @ x <= H_limit.
        H_limit (torch.Tensor): RHS vector for bus state inequality constraints.
        batt_cost (float): Weight on battery charge deviation from target.
        curt_net_cost (float): Weight on net curtailment.
        curt_change_cost (float): Weight on curtailment changes.
        bus_slack_cost (float): Penalty on bus slack variables.
        line_slack_cost (float): Penalty on line slack variables.
    """

    def __init__(self, grid, H, buses_in_area, init_line_max, init_line_min, batt_cost, curt_change_cost,
                 curt_net_cost, bus_slack_cost, line_slack_cost, eps=1e-8):
        """
        Args:
            grid (Grid): Full grid that the agent runs on.
            H (int): MPC prediction horizon length.
            buses_in_area (torch.Tensor): Bus numbers in this controller's designated area.
            init_line_max (torch.Tensor): Shape (T, num_lines); initial max line limit changes per step.
            init_line_min (torch.Tensor): Shape (T, num_lines); initial min line limit changes per step.
            batt_cost (float): Cost on battery charge deviation from target. Must be non-negative.
            curt_change_cost (float): Cost on changes in curtailed power. Must be non-negative.
            curt_net_cost (float): Cost on net curtailed power. Must be non-negative.
            bus_slack_cost (float): Bus slack penalty. Must be non-negative.
            line_slack_cost (float): Line slack penalty. Must be non-negative.
            eps (float): Regularization coefficient ensuring the QP is positive definite.
        """
        super().__init__(grid, H, buses_in_area, batt_cost, curt_change_cost, curt_net_cost, bus_slack_cost, line_slack_cost,
                         eps=eps)
        self.register_buffer("init_line_max_change_buf", init_line_max.clone())
        self.register_buffer("init_line_min_change_buf", init_line_min.clone())
        self.line_max_change = nn.Parameter(self.init_line_max_change_buf.clone())
        self.line_min_change = nn.Parameter(self.init_line_min_change_buf.clone())
        self.line_state_dim = self.grid.num_lines
        self.line_state = torch.zeros(self.line_state_dim)
        self.bus_ineq_dim = 2 * self.bus_state_dim
        self.line_max_slack_dim = self.line_state_dim
        self.line_min_slack_dim = self.line_state_dim

        self.line_B_curt = self.grid.B_curt[0:self.grid.num_lines,self.curt_idx]
        self.line_B_batt = self.grid.B_batt[0:self.grid.num_lines,self.batt_idx]
        self.line_B_noise = self.grid.B_noise[0:self.grid.num_lines,self.buses_in_area]
        # ----------------------------------------------------------------------------------------
        # aggregate vector is bus state, bus slack, line state, line max slack, line min slack, batt action, curt action
        self.z_dim = H * (self.bus_state_dim + self.line_state_dim + self.line_max_slack_dim \
                            + self.line_min_slack_dim + self.num_batt + self.num_curt)
        self.eq_dim = H * (self.bus_state_dim + self.line_state_dim)
        self.ineq_dim = H * (self.bus_ineq_dim + self.line_max_slack_dim + self.line_min_slack_dim)

        true_cost_vec = torch.zeros(self.z_dim)
        target_vec = torch.zeros(self.z_dim)

        self.line_state_start_idx = H * self.bus_state_dim
        self.line_max_slack_start_idx = self.line_state_start_idx + H * self.line_state_dim
        self.line_min_slack_start_idx = self.line_max_slack_start_idx + H * self.line_state_dim

        self.action_batt_start_idx = self.line_min_slack_start_idx + H * self.line_state_dim
        self.action_curt_start_idx = self.action_batt_start_idx + H * self.num_batt

        # -------- Create Q and q matrices --------
        for i in range(self.H):
            # battery charge targets and costs
            idx = i * self.bus_state_dim
            true_cost_vec[idx:idx+self.num_batt] = batt_cost * torch.ones(self.num_batt)
            target_vec[idx:idx+self.num_batt] = self.grid.target_batt_charges[self.batt_idx]

            # net curtailment costs
            idx = i * self.bus_state_dim + 2 * self.num_batt
            true_cost_vec[idx:idx+self.num_curt] = curt_net_cost * torch.ones(self.num_curt)

        # curt action costs
        idx = self.action_curt_start_idx
        true_cost_vec[idx:idx+H*self.num_curt] = curt_change_cost * torch.ones(H*self.num_curt)

        # line max slack costs
        idx = self.line_max_slack_start_idx
        true_cost_vec[idx:idx+H*self.line_max_slack_dim] = line_slack_cost * torch.ones(H*self.line_max_slack_dim)

        # line min slack costs
        idx = self.line_min_slack_start_idx
        true_cost_vec[idx:idx+H*self.line_min_slack_dim] = line_slack_cost * torch.ones(H*self.line_min_slack_dim)

        self.true_cost_vec = true_cost_vec

        # ----------------------------------------

        # -------- Create A matrix --------
        self.A = torch.zeros(self.eq_dim, self.z_dim)
        # evolution of bus states
        for i in range(H):
            block_row = torch.zeros(self.bus_state_dim, self.z_dim)
            action_batt_idx = i * self.num_batt + self.action_batt_start_idx
            action_curt_idx = i * self.num_curt + self.action_curt_start_idx

            if not i == 0:
                block_row[:,(i-1)*self.bus_state_dim:i*self.bus_state_dim] = -self.bus_A
            block_row[:,i*self.bus_state_dim:(i+1)*self.bus_state_dim] = torch.eye(self.bus_state_dim)
            block_row[:,action_batt_idx:action_batt_idx+self.num_batt] = -self.bus_B_batt
            block_row[:,action_curt_idx:action_curt_idx+self.num_curt] = -self.bus_B_curt

            row_idx = i * self.bus_state_dim
            self.A[row_idx:row_idx+self.bus_state_dim] = block_row

        # evolution of line states
        for i in range(H):
            action_batt_idx = i * self.num_batt + self.action_batt_start_idx
            action_curt_idx = i * self.num_curt + self.action_curt_start_idx
            block_row = torch.zeros(self.line_state_dim, self.z_dim)

            idx = i * self.line_state_dim + self.line_state_start_idx
            if not i == 0:
                block_row[:,idx-self.line_state_dim:idx] = -torch.eye(self.line_state_dim)
            block_row[:,idx:idx+self.line_state_dim] = torch.eye(self.line_state_dim)
            block_row[:,action_batt_idx:action_batt_idx+self.num_batt] = -self.line_B_batt
            block_row[:,action_curt_idx:action_curt_idx+self.num_curt] = -self.line_B_curt

            row_idx = i * self.line_state_dim + H * self.bus_state_dim
            self.A[row_idx:row_idx+self.line_state_dim] = block_row
        # --------------------------------

        # -------- Create G matrix --------
        self.G = torch.zeros(self.ineq_dim, self.z_dim)
        for i in range(H):
            block_row = torch.zeros(self.bus_ineq_dim, self.z_dim)
            block_row[:,i*self.bus_state_dim:(i+1)*self.bus_state_dim] = self.H_x

            row_idx = i * self.bus_ineq_dim
            self.G[row_idx:row_idx+self.bus_ineq_dim] = block_row

        for i in range(H):
            block_row = torch.zeros(self.line_state_dim, self.z_dim)
            line_state_idx = i * self.line_state_dim + self.line_state_start_idx
            line_max_slack_idx = i * self.line_max_slack_dim + self.line_max_slack_start_idx

            block_row[:,line_state_idx:line_state_idx+self.line_state_dim] = torch.eye(self.line_state_dim)
            block_row[:,line_max_slack_idx:line_max_slack_idx+self.line_state_dim] = -torch.eye(self.line_max_slack_dim)

            row_idx = i * self.line_state_dim + H * self.bus_ineq_dim
            self.G[row_idx:row_idx+self.line_state_dim] = block_row

        for i in range(H):
            block_row = torch.zeros(self.line_state_dim, self.z_dim)
            line_state_idx = i * self.line_state_dim + self.line_state_start_idx
            line_min_slack_idx = i * self.line_min_slack_dim + self.line_min_slack_start_idx

            block_row[:,line_state_idx:line_state_idx+self.line_state_dim] = -torch.eye(self.line_state_dim)
            block_row[:,line_min_slack_idx:line_min_slack_idx+self.line_state_dim] = -torch.eye(self.line_min_slack_dim)

            row_idx = i * self.line_state_dim + H * (self.bus_ineq_dim + self.line_state_dim)
            self.G[row_idx:row_idx+self.line_state_dim] = block_row
        # ----------------------------------------
        # self.nBatch = 1
        # Q_, _ = qpth.util.expandParam(self.Q, self.nBatch, 3)
        # G_, _ = qpth.util.expandParam(self.G, self.nBatch, 3)
        # A_, _ = qpth.util.expandParam(self.A, self.nBatch, 3)
        # ------------ Construct scaling matrices ------------
        scale_vec = torch.zeros(self.z_dim)
        # bus states
        for i in range(H):
            # battery charge 
            idx = i * self.bus_state_dim
            scale_vec[idx:idx+self.num_batt] = self.grid.target_batt_charges[self.batt_idx]

            # battery power injection
            idx = i * self.bus_state_dim + self.num_batt
            scale_vec[idx:idx+self.num_batt] = self.grid.batt_power_max_limits[self.batt_idx]

            # curtailment
            idx = i * self.bus_state_dim + 2 * self.num_batt
            scale_vec[idx:idx+self.num_curt] = self.grid.curt_max_limits[self.curt_idx]

        # line state
        for i in range(H):
            idx = self.line_state_start_idx + i * self.line_state_dim
            scale_vec[idx:idx+self.line_state_dim] = self.grid.line_data[:,5]

        # line max slack
        for i in range(H):
            idx = self.line_max_slack_start_idx + i * self.line_max_slack_dim
            scale_vec[idx:idx+self.line_max_slack_dim] = self.grid.line_data[:,5]
        self.line_max_slack_scale_vec = self.grid.line_data[:,5]

        # line min slack
        for i in range(H):
            idx = self.line_min_slack_start_idx + i * self.line_min_slack_dim
            scale_vec[idx:idx+self.line_min_slack_dim] = self.grid.line_data[:,5]
        self.line_min_slack_scale_vec = self.grid.line_data[:,5]

        # batt action
        for i in range(H):
            idx = self.action_batt_start_idx + i * self.num_batt
            scale_vec[idx:idx+self.num_batt] = self.grid.batt_power_max_limits[self.batt_idx]

        # curt action
        for i in range(H):
            idx = self.action_curt_start_idx + i * self.num_curt
            scale_vec[idx:idx+self.num_curt] = self.grid.curt_max_limits[self.curt_idx]

        self.scale_vec = scale_vec

        # D = diag(scale_vec)
        AD = torch.mul(self.A, scale_vec)
        E_vec = torch.zeros(self.eq_dim)
        eps = 1e-6
        for i in range(self.eq_dim):
            E_vec[i] = 1 / (torch.linalg.norm(AD[i]) + eps)
        
        GD = torch.mul(self.G, scale_vec)
        F_vec = torch.zeros(self.ineq_dim)
        eps = 1e-6
        for i in range(self.ineq_dim):
            F_vec[i] = 1 / (torch.linalg.norm(GD[i]) + eps)
        
        self.E_vec = E_vec
        self.F_vec = F_vec
        raw_q = 2 * scale_vec * true_cost_vec * scale_vec

        cost_scale = torch.max(torch.abs(raw_q))
        ridge_eps = 1e-8 * cost_scale
        
        q_vec = ((2 * scale_vec * true_cost_vec * scale_vec) + ridge_eps * torch.ones(self.z_dim)) / cost_scale
        # print(torch.min(q_vec), torch.max(q_vec))
        self.q = -2 * true_cost_vec * target_vec / cost_scale   # slack and ridge cost have linear term 0
        self.const = target_vec @ (true_cost_vec * target_vec) / cost_scale
        
        self.q_vec = q_vec
        self.scaled_Q = torch.diag(q_vec)
        
        self.scaled_q = scale_vec * self.q 
    
        self.scaled_A = E_vec.unsqueeze(-1) * AD
        self.scaled_G = F_vec.unsqueeze(-1) * GD
        self.nBatch = 1
        Q_, _ = qpth.util.expandParam(self.scaled_Q, self.nBatch, 3)
        G_, _ = qpth.util.expandParam(self.scaled_G, self.nBatch, 3)
        A_, _ = qpth.util.expandParam(self.scaled_A, self.nBatch, 3)
        self.csc_scaled_Q = self.scaled_Q.to_sparse_csc()
        self.csc_scaled_G = self.scaled_G.to_sparse_csc()
        self.csc_scaled_A = self.scaled_A.to_sparse_csc()
        self.scipy_scaled_Q = csc_torch_to_scipy(self.csc_scaled_Q)
        self.scipy_scaled_G = csc_torch_to_scipy(self.csc_scaled_G)
        self.scipy_scaled_A = csc_torch_to_scipy(self.csc_scaled_A)


    def zero_line_state(self):
        """Sets line states to all zeros."""
        self.line_state = torch.zeros_like(self.line_state)


    def reset_params(self):
        """Resets split constraint parameters to their initial values."""
        self.line_max_change = nn.Parameter(self.init_line_max_change_buf.clone())
        self.line_min_change = nn.Parameter(self.init_line_min_change_buf.clone())


    def interpret_z(self, z, update_state=False):
        norm_squared = torch.sum(z**2)
        z *= self.scale_vec
        bus_states = torch.zeros(self.H, self.bus_state_dim)
        for i in range(self.H):
            bus_states[i] = z[i*self.bus_state_dim:(i+1)*self.bus_state_dim]

        line_states = torch.zeros(self.H, self.line_state_dim)
        for i in range(self.H):
            line_states[i] = z[i*self.line_state_dim+self.line_state_start_idx:(i+1)*self.line_state_dim+self.line_state_start_idx]

        actions_curt = torch.zeros(self.H, self.num_curt)
        for i in range(self.H):
            actions_curt[i] = z[i*self.num_curt+self.action_curt_start_idx:(i+1)*self.num_curt+self.action_curt_start_idx]

        actions_batt = torch.zeros(self.H, self.num_batt)
        for i in range(self.H):
            actions_batt[i] = z[i*self.num_batt+self.action_batt_start_idx:(i+1)*self.num_batt+self.action_batt_start_idx]

        line_max_slacks = torch.zeros(self.H, self.line_max_slack_dim)
        for i in range(self.H):
            line_max_slacks[i] = z[i*self.line_max_slack_dim+self.line_max_slack_start_idx:(i+1)*self.line_max_slack_dim+self.line_max_slack_start_idx]

        line_min_slacks = torch.zeros(self.H, self.line_min_slack_dim)
        for i in range(self.H):
            line_min_slacks[i] = z[i*self.line_min_slack_dim+self.line_min_slack_start_idx:(i+1)*self.line_min_slack_dim+self.line_min_slack_start_idx]

        self.target_charges = self.grid.target_batt_charges[self.batt_idx]
        objective = 0
        batt_obj = 0
        curt_change_obj = 0
        curt_net_obj = 0
        with torch.no_grad():
            for i in range(self.H):
                # print(actions_curt[i])
                objective += self.batt_cost * torch.sum((bus_states[i][self.bus_state_batt_charge_idx] - self.target_charges)**2)
                batt_obj += self.batt_cost * torch.sum((bus_states[i][self.bus_state_batt_charge_idx] - self.target_charges)**2)
                objective += self.curt_change_cost * torch.sum(actions_curt[i]**2)
                curt_change_obj += self.curt_change_cost * torch.sum(actions_curt[i]**2)
                objective += self.curt_net_cost * torch.sum(bus_states[i][self.bus_state_curt_idx]**2)
                curt_net_obj += self.curt_net_cost * torch.sum(bus_states[i][self.bus_state_curt_idx]**2)
                objective += self.line_slack_cost * torch.sum(line_max_slacks[i]**2)
                objective += self.line_slack_cost * torch.sum(line_min_slacks[i]**2)
            objective += self.eps * torch.sum(norm_squared ** 2)

        if update_state:
            self.bus_state = bus_states[0]
            self.line_state = line_states[0]

        return bus_states, line_states, actions_curt, actions_batt, None, line_max_slacks, line_min_slacks, objective

    
    def get_matrices_dQPTH(self, disturbances, t):
        """Returns the QP matrices for this controller given current disturbance predictions.

        Args:
            disturbances (torch.Tensor): Shape (H, num_buses); power injection noise predictions per bus.
            t (int): Current timestep.

        Returns:
            dict: QP matrices with keys: csc_scaled_Q, csc_scaled_G, csc_scaled_A, scipy_scaled_Q,
                scipy_scaled_G, scipy_scaled_A, Q, q, A, b, G, h, const.
        """
        b = torch.zeros(self.eq_dim)
        b[0:self.bus_state_dim] = self.bus_A @ self.bus_state

        for i in range(self.H):
            row_idx = i * self.line_state_dim + self.H * self.bus_state_dim
            b[row_idx:row_idx+self.line_state_dim] = self.line_B_noise @ disturbances[i]

        row_idx = self.H * self.bus_state_dim
        b[row_idx:row_idx+self.line_state_dim] += self.line_state

        h = torch.zeros(self.ineq_dim)
        for i in range(self.H):
            row_idx = i * self.bus_ineq_dim
            h[row_idx:row_idx+self.bus_ineq_dim] = self.H_limit

            row_idx = i * self.line_state_dim + self.H * self.bus_ineq_dim
            line_max = torch.sum(self.line_max_change[0:t+i+1], dim=0)
            h[row_idx:row_idx+self.line_state_dim] = line_max

            row_idx = i * self.line_state_dim + self.H * (self.bus_ineq_dim + self.line_state_dim)
            line_min = torch.sum(self.line_min_change[0:t+i+1], dim=0)
            h[row_idx:row_idx+self.line_state_dim] = -line_min

        dQPTH_config = {
            "csc_scaled_Q": self.csc_scaled_Q,
            "csc_scaled_G": self.csc_scaled_G,
            "csc_scaled_A": self.csc_scaled_A,
            "scipy_scaled_Q": self.scipy_scaled_Q,
            "scipy_scaled_G": self.scipy_scaled_G,
            "scipy_scaled_A": self.scipy_scaled_A,
            "Q": self.scaled_Q,
            "q": self.scaled_q,
            "A": self.scaled_A,
            "b": self.E_vec * b,
            "G": self.scaled_G,
            "h": self.F_vec * h,
            "const": self.const,
        }

        return dQPTH_config
        
