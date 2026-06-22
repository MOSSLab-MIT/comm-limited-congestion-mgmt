import torch

class Grid:
    """Tracks and updates grid state with linearized dynamics.

    Agnostic to control strategy or communication structure; only models grid physics (line flows,
    batteries, and curtailable generation).

    Attributes:
        state_dim (int): Dimension of the state vector.
        num_buses (int): Number of buses on the grid.
        num_lines (int): Number of lines on the grid.
        num_curt (int): Number of buses with curtailable generation.
        num_batt (int): Number of batteries on the grid.
        state (torch.Tensor): Shape (state_dim,); current state ordered as
            [line_flows, batt_charge, batt_power_injection, curtailed_power].
        init_state (torch.Tensor): Shape (state_dim,); initial state vector.
        bus_with_curt (torch.Tensor): Shape (num_curt,); bus indices with curtailable generation.
        bus_with_batt (torch.Tensor): Shape (num_batt,); bus indices with a battery.
        delta_t (float): Time between grid updates (seconds).
        line_data (torch.Tensor): Raw branch data from MATPOWER test case.
        bus_data (torch.Tensor): Raw bus data from MATPOWER test case.
        gen_data (torch.Tensor): Raw generator data from MATPOWER test case.
        M (torch.Tensor): Shape (num_lines, num_buses); PTDF sensitivity matrix.
        A (torch.Tensor): Shape (state_dim, state_dim); state transition matrix (no actions or noise).
        B_curt (torch.Tensor): Shape (state_dim, num_curt); impact of curtailment on state.
        B_batt (torch.Tensor): Shape (state_dim, num_batt); impact of battery actions on state.
        B_noise (torch.Tensor): Shape (state_dim, num_buses); impact of bus noise on state.
        H_x (torch.Tensor): Constraint matrix; H_x @ state <= H_limit.
        H_limit (torch.Tensor): Constraint RHS vector.
        line_limits (torch.Tensor): Shape (num_lines,); max power flow magnitude per line.
        batt_charge_max_limits (torch.Tensor): Shape (num_batt,); max battery charge.
        batt_charge_min_limits (torch.Tensor): Shape (num_batt,); min battery charge.
        batt_power_max_limits (torch.Tensor): Shape (num_batt,); max battery power injection.
        batt_power_min_limits (torch.Tensor): Shape (num_batt,); min battery power injection.
        curt_max_limits (torch.Tensor): Shape (num_curt,); max curtailable power per generator.
        curt_min_limits (torch.Tensor): Shape (num_curt,); min curtailable power per generator.
        target_batt_charges (torch.Tensor): Shape (num_batt,); target charge level per battery.
        state_line_flow_idx (slice): Index slice for line flows in state vector.
        state_batt_charge_idx (slice): Index slice for battery charges in state vector.
        state_curt_idx (slice): Index slice for curtailment in state vector.
    """

    def __init__(self, bus_with_curt, bus_with_batt, delta_t, line_data_loc, bus_data_loc, gen_data_loc, ptdf_data_loc):
        """
        Args:
            bus_with_curt (torch.Tensor): Shape (num_curt,); bus indices with curtailable generation.
            bus_with_batt (torch.Tensor): Shape (num_batt,); bus indices with a battery.
            delta_t (float): Time between grid updates (seconds).
            line_data_loc (str): Path to branch data file (.npy, MATPOWER format).
            bus_data_loc (str): Path to bus data file (.npy, MATPOWER format).
            gen_data_loc (str): Path to generator data file (.npy, MATPOWER format).
            ptdf_data_loc (str): Path to PTDF matrix file used as sensitivity matrix M.

        Notes:
            Buses and lines are 0-indexed. Data files use the MATPOWER case format stored as .npy arrays.
        """
        self.bus_with_curt = bus_with_curt
        self.bus_with_batt = bus_with_batt
        self.delta_t = float(delta_t)
        self.line_data = torch.load(line_data_loc, weights_only=True)
        self.bus_data = torch.load(bus_data_loc, weights_only=True)
        self.gen_data = torch.load(gen_data_loc, weights_only=True)
        self.M = torch.load(ptdf_data_loc, weights_only=True)

        self.num_buses = self.bus_data.shape[0]
        self.num_lines = self.line_data.shape[0]
        self.num_curt = bus_with_curt.shape[0]
        self.num_batt = bus_with_batt.shape[0]
        self.state_dim = self.num_lines + self.num_curt + 2*self.num_batt
        self.bus_state_dim = self.num_curt + 2*self.num_batt
        self.bus_slack_dim = 2 * self.num_curt + 4 * self.num_batt

        init_line_flows = self.line_data[:, 13]
        init_batt_charges, init_batt_powers = self.init_batteries()
        init_curt_values = self.init_curt()
        self.set_matrices()

        self.init_state = torch.cat([init_line_flows, init_batt_charges, init_batt_powers, init_curt_values])
        self.state = self.init_state

        self.state_line_flow_idx = slice(0, self.num_lines)
        self.state_batt_charge_idx = slice(self.num_lines, self.num_lines + self.num_batt)
        self.state_batt_power_idx = slice(self.num_lines + self.num_batt, self.num_lines + 2*self.num_batt)
        self.state_curt_idx = slice(self.num_lines + 2*self.num_batt, self.num_lines + 2*self.num_batt + self.num_curt)
        self.state_bus_state_idx = slice(self.num_lines, self.num_lines + 2*self.num_batt + self.num_curt)


    def check_state_feasible(self, tol=1e-3):
        """Checks if the current grid state is feasible.

        Args:
            tol (float): Maximum acceptable constraint violation.

        Returns:
            tuple:
                - bool: True if feasible.
                - torch.Tensor: Indices of infeasible constraints.
                - torch.Tensor: Violation amounts at those indices.
        """
        gap = self.H_x @ self.state - self.H_limit
        infeasible_idx = torch.where(gap > tol)[0]
        return torch.all(gap <= tol), infeasible_idx, gap[infeasible_idx]


    def add_agent(self, agent):
        """Adds an agent to the grid's agent list.

        Args:
            agent: Agent to add.
        """
        self.agents.append(agent)


    def set_matrices(self):
        """Constructs matrices for grid dynamics and constraints."""
        self.M_curt = self.M[:, self.bus_with_curt]
        self.M_batt = self.M[:, self.bus_with_batt]
        self.M_noise = self.M
        nL = self.num_lines
        nB = self.num_batt
        nC = self.num_curt
        N = nL + 2*nB + nC
        self.A = torch.zeros((N, N))
        self.A[0:nL,          0:nL]          = torch.eye(nL)
        self.A[nL:nL+nB,      nL:nL+nB]      = torch.eye(nB)
        self.A[nL+nB:nL+2*nB, nL+nB:nL+2*nB] = torch.eye(nB)
        self.A[nL+2*nB:,      nL+2*nB:]      = torch.eye(nC)
        self.A[nL:nL+nB, nL+nB:nL+2*nB]      = -self.delta_t * torch.eye(nB) / 3600

        self.B_curt = torch.cat([self.M_curt,         torch.zeros(nB, nC),
                                 torch.zeros(nB, nC), torch.eye(nC)])
        self.B_batt = torch.cat([self.M_batt,  -self.delta_t * torch.eye(nB) / 3600,
                                 torch.eye(nB), torch.zeros(nC, nB)])
        self.B_noise = torch.cat([self.M_noise, torch.zeros(self.state_dim - nL, self.num_buses)])

        self.H_x = torch.zeros((2*nL+4*nB+2*nC,nL+2*nB+nC))
        self.H_x[0:nL, 0:nL] = -torch.eye(nL)

        rows = 2*nL + 4*nB + 2*nC
        cols = nL + 2*nB + nC
        self.H_x = torch.zeros((rows, cols))
        self.H_x[0:nL,                        0:nL]               = -torch.eye(nL)
        self.H_x[nL:2*nL,                     0:nL]               =  torch.eye(nL)
        self.H_x[2*nL:2*nL+nB,                nL:nL+nB]           = -torch.eye(nB)
        self.H_x[2*nL+nB:2*nL+2*nB,           nL:nL+nB]           =  torch.eye(nB)
        self.H_x[2*nL+2*nB:2*nL+3*nB,         nL+nB:nL+2*nB]      = -torch.eye(nB)
        self.H_x[2*nL+3*nB:2*nL+4*nB,         nL+nB:nL+2*nB]      =  torch.eye(nB)
        self.H_x[2*nL+4*nB:2*nL+4*nB+nC,      nL+2*nB:nL+2*nB+nC] = -torch.eye(nC)
        self.H_x[2*nL+4*nB+nC:2*nL+4*nB+2*nC, nL+2*nB:nL+2*nB+nC] =  torch.eye(nC)

        self.line_limits = self.line_data[:,5]
        self.H_limit = torch.cat([self.line_limits,           self.line_limits,           -self.batt_charge_min_limits, self.batt_charge_max_limits,
                                 -self.batt_power_min_limits, self.batt_power_max_limits, -self.curt_min_limits,        self.curt_max_limits])


    def init_batteries(self):
        """Initializes battery limits and starting charge/power values.

        Returns:
            tuple:
                - init_batt_charges (torch.Tensor): Shape (num_batt,); initial battery charges.
                - init_batt_powers (torch.Tensor): Shape (num_batt,); initial battery power injections.
        """
        # assuming that batteries can collectively meet 5% of total average instantaneous demand
        total_demand = torch.sum(self.bus_data[:,2])
        max_batt_supply = total_demand / 20
        self.batt_power_max_limits = torch.ones(self.num_batt) * max_batt_supply / self.num_batt
        self.batt_power_min_limits = -self.batt_power_max_limits
        # self.batt_power_min_limits = -0 * torch.ones(self.num_batt) * (total_demand / 4) / self.num_batt

        # assuming that total battery power storage capacity can sustain max injection for 4 hours
        total_max = max_batt_supply * 4 * 60 * 60

        # assuming that all batteries have the same power storage capacity
        self.batt_charge_max_limits = torch.ones(self.num_batt) * (total_max / self.num_batt) / 3600    # conver to MWh
        self.batt_charge_min_limits = torch.zeros(self.num_batt)

        # init_batt_charges = self.batt_charge_min_limits\
        #       + np.random.rand(self.num_batt) * (self.batt_charge_max_limits - self.batt_charge_min_limits)
        init_batt_charges = 0.5 * (self.batt_charge_max_limits + self.batt_charge_min_limits)
        init_batt_powers = 0.5 * (self.batt_power_max_limits + self.batt_power_min_limits)

        # self.target_batt_charges = torch.rand(self.num_batt) * (self.batt_charge_max_limits - self.batt_charge_min_limits) \
        #                             + self.batt_charge_min_limits
        self.target_batt_charges = init_batt_charges

        return init_batt_charges, init_batt_powers


    def init_curt(self):
        """Initializes curtailment limits and starting values.

        Returns:
            torch.Tensor: Shape (num_curt,); initial curtailed power per generator.
        """
        self.curt_max_limits = self.gen_data[:,8]
        self.curt_min_limits = torch.zeros(self.num_curt)
        init_curt_values = 0.2 * self.curt_max_limits

        return init_curt_values


    def reset_state(self):
        """Resets grid state to its initial value."""
        self.state = self.init_state


    def update_state(self, action_curt, action_batt, noise):
        """Advances the grid state by one timestep given actions and noise.

        Args:
            action_curt (torch.Tensor): Shape (num_curt,); curtailment at each generator (same order as bus_with_curt).
            action_batt (torch.Tensor): Shape (num_batt,); battery power injection (same order as bus_with_batt).
            noise (torch.Tensor): Shape (num_buses,); power injection noise at each bus.
        """
        self.state = self.A @ self.state + self.B_curt @ action_curt + self.B_batt @ action_batt + self.B_noise @ noise


    def get_grid_data(self):
        """Returns a dict of grid parameters needed by agents.

        Returns:
            dict: Contains num_lines, line_data, num_curt, bus_with_curt, num_batt, bus_with_batt,
                ptdf, delta_t, and all battery/curtailment limit arrays.
        """
        grid_data = dict()
        grid_data["num_lines"] = self.num_lines
        grid_data["line_data"] = self.line_data
        grid_data["num_curt"] = self.num_curt
        grid_data["bus_with_curt"] = self.bus_with_curt
        grid_data["num_batt"] = self.num_batt
        grid_data["bus_with_batt"] = self.bus_with_batt
        grid_data["ptdf"] = self.M
        grid_data["delta_t"] = self.delta_t
        grid_data["batt_charge_min_limits"] = self.batt_charge_min_limits
        grid_data["batt_charge_max_limits"] = self.batt_charge_max_limits
        grid_data["batt_power_min_limits"] = self.batt_power_min_limits
        grid_data["batt_power_max_limits"] = self.batt_power_max_limits
        grid_data["curt_min_limits"] = self.curt_min_limits
        grid_data["curt_max_limits"] = self.curt_max_limits

        return grid_data


    def get_lines_from_buses(self, buses):
        """Returns line indices and endpoint buses for given bus pairs.

        Args:
            buses (list of tuples): Each tuple is a pair of bus numbers to look up.

        Returns:
            torch.Tensor: Rows of (line_idx, from_bus, to_bus); entries are None if no line connects the pair.
        """
        result = []
        for bus_pair in buses:
            bus1, bus2 = bus_pair
            match = torch.where(
                ((self.line_data[:, 0] == bus1) & (self.line_data[:, 1] == bus2)) |
                ((self.line_data[:, 0] == bus2) & (self.line_data[:, 1] == bus1))
            )[0]
            if match.size > 0:
                line_idx = match[0]
                from_bus = int(self.line_data[line_idx, 0])
                to_bus = int(self.line_data[line_idx, 1])
                result.append([line_idx, from_bus, to_bus])
            else:
                result.append([None, None, None])
        return torch.tensor(result) 
