import sys, os, argparse
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
import time

import grid_pkg
import controller_pkg
import controller_utils
import dQPTH
import evals

torch.set_default_dtype(torch.double)
torch.set_printoptions(threshold=10000)
NUMPY_SEED = 0
TORCH_SEED = 0
np.random.seed(NUMPY_SEED)
torch.manual_seed(TORCH_SEED)

def str2bool(s):
    return s.lower() in {"1","true","t","yes","y"}

def build_tag(parser, args, *, sep="_", bool_as_int=False, only_nondefaults=False):
    """
    Create a tag from all argparse args without hard-coding keys.

    - sep: separator between parts
    - bool_as_int: if True -> booleans encoded as 1/0; else True/False
    - only_nondefaults: if True -> include only args whose value != default
    """
    def fmt(v):
        if isinstance(v, bool):
            return ("1" if v else "0") if bool_as_int else str(v)
        if isinstance(v, float):
            s = f"{v:g}"
            return s if "." in s else s + ".0"
        return str(v)

    # Build a dict of defaults to optionally filter unchanged values
    defaults = {}
    for action in parser._actions:
        if not hasattr(action, "dest") or action.dest in (None, "help"):
            continue
        defaults[action.dest] = action.default

    parts = []
    # Use parser._actions to preserve declaration order
    for action in parser._actions:
        dest = getattr(action, "dest", None)
        if not dest or dest == "help":
            continue

        val = getattr(args, dest)
        if only_nondefaults and dest in defaults and val == defaults[dest]:
            continue

        parts.append(f"{dest}{fmt(val)}")

    return sep.join(parts)

# @profile
def main(args):
    torch.set_default_dtype(torch.double)
    torch.set_printoptions(threshold=10000)

    # -------- sweep params --------
    noise_mag = args.noise_mag
    offset = args.offset
    epochs = args.epochs
    lr = args.lr
    forecast_seed = args.forecast_seed
    radius = args.radius
    optimizer_type = args.optimizer_type
    lr_schedule = args.lr_schedule
    patience = args.patience
    lr_decay_step = args.lr_decay_step
    batch_size = args.batch_size
    max_grad_norm = args.max_grad_norm
    # ------------------------------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(BASE_DIR)
    folder = os.path.join(ROOT_DIR, f"results/ours/")
    tag = build_tag(parser, args, bool_as_int=False, only_nondefaults=False)
    print(f"Results for this run will be saved at {folder}ours_{tag}.pt")
    os.makedirs(folder, exist_ok=True)

    # line limits come from NREL-118 dataset, all other information from MATPOWER case 118
    line_data_loc = os.path.join(ROOT_DIR, 'data/case118_line_data.pt')
    bus_data_loc = os.path.join(ROOT_DIR, 'data/case118_bus_data.pt')
    gen_data_loc = os.path.join(ROOT_DIR, 'data/case118_gen_data.pt')
    ptdf_data_loc = os.path.join(ROOT_DIR, 'data/case118_ptdf_data.pt')

    bus_with_curt = torch.load(gen_data_loc, weights_only=True)[:,0].int()
    num_curt = bus_with_curt.shape[0]
    bus_with_batt = torch.tensor([10*i+2 for i in range(12)], dtype=torch.int)
    num_batt = bus_with_batt.shape[0]
    delta_t = 15

    grid = grid_pkg.Grid(bus_with_curt, bus_with_batt, delta_t, line_data_loc, bus_data_loc, gen_data_loc, ptdf_data_loc)
    T = 20
    H = 5

    forecast_traj_loc = os.path.join(ROOT_DIR, f'data/scenario_generation/forecast_trajs_rad{radius}.pt')
    forecast_trajs = torch.load(forecast_traj_loc)

    batt_cost = 100
    curt_change_cost = 0.01
    curt_net_cost = 1
    bus_slack_cost = 1e8
    line_slack_cost = 1e2

    num_agents = 3
    nodes_1 = [i for i in range(0, 42)] + [112, 113, 114, 116]
    nodes_2 = [i for i in range(42, 69)] + [115]
    nodes_3 = [i for i in range(69, 112)] + [117]
    partition = [nodes_1, nodes_2, nodes_3]
    controller_list = controller_utils.create_split_constraint_controllers(grid, num_agents, partition, T, H, batt_cost,
                                                                           curt_change_cost, curt_net_cost, bus_slack_cost,
                                                                           line_slack_cost)

    ckpt = dict()
    for k, v in vars(args).items():
        ckpt[k] = v

    all_forecast_traj = forecast_trajs[(noise_mag, offset, forecast_seed)]

    train_start_time = time.time()
    schedule_kwargs = {
        "step_size": lr_decay_step,
        "patience": patience
    }
    train_result = evals.train_with_rollout(grid, T, H, all_forecast_traj, batt_cost, curt_change_cost, curt_net_cost,
                                            bus_slack_cost, line_slack_cost, epochs, lr, optimizer_type=optimizer_type,
                                            max_grad_norm=max_grad_norm, lr_schedule=lr_schedule,
                                            schedule_kwargs=schedule_kwargs, batch_size=batch_size)
    train_time = time.time() - train_start_time
    ckpt["train_time"] = train_time
    ckpt = ckpt | train_result
    print("save after training")
    torch.save(ckpt, f"{folder}checkpoint_{tag}.pt")
    run_failed = train_result["run_failed"]
    line_max_changes = train_result["line_max_changes"].detach()
    line_min_changes = train_result["line_min_changes"].detach()
    fail_message = train_result["fail_message"]
    pool = mp.Pool(processes=num_agents)

    settings = dQPTH.build_settings(solve_type="sparse", qp_solver="gurobi", lin_solver="qdldl", warm_start_from_previous=True)
    dQPTH_layer = dQPTH.dQPTH_layer(settings=settings, pool=pool)
    failed_tests = dict()

    test_skew_mags = [10, 20, 30, 40]
    test_trajs_no_mismatch_loc = os.path.join(ROOT_DIR, f'data/scenario_generation/test_trajs_no_mismatch_rad{radius}.pt')
    test_trajs_no_mismatch = torch.load(test_trajs_no_mismatch_loc)
    if not run_failed:
        controller_utils.assign_line_limit_changes(controller_list, line_max_changes, line_min_changes)
        for test_seed in range(3):
            for test_skew_mag in test_skew_mags:
                test_traj_mismatch_loc = os.path.join(ROOT_DIR, f'data/scenario_generation/test_trajs_with_mismatch_rad{radius}/test_trajs_with_mismatch_noise_mag{noise_mag}_offset{offset}_tsm{test_skew_mag}.pt')
                test_trajs_with_mismatch = torch.load(test_traj_mismatch_loc)
                num_test_traj = test_trajs_with_mismatch["num_test_traj"]
                all_test_trajs = test_trajs_with_mismatch[(noise_mag, offset, test_skew_mag, test_seed)]
                our_losses = torch.zeros(num_test_traj,3)

                for j in tqdm(range(10)):
                    test_traj = all_test_trajs[j]
                    try:
                        ckpt_ours = evals.evaluate_on_traj(grid, controller_list, dQPTH_layer, test_traj, T, H, batt_cost, curt_change_cost, curt_net_cost, bus_slack_cost, line_slack_cost)
                        our_losses[j] = ckpt_ours["total_losses"]
                        print(f"test loss was {ckpt_ours['total_losses']}")
                    except Exception as e:
                        print(f"Evaluate on test traj failed for test_seed {test_seed}, j {j}, exception {e}")
                        failed_tests[(i,j)] = test_traj
                        our_losses[j] = -1
                ckpt[(test_skew_mag, test_seed)] = our_losses
            print(f"saving forecast seed {forecast_seed} test_seed {test_seed}", flush=True)
            torch.save(ckpt, f"{folder}checkpoint_{tag}.pt")

        test_skew_mag = 0
        all_test_trajs = test_trajs_no_mismatch[(noise_mag, offset)]
        our_losses = torch.zeros(num_test_traj, 3)
        for i in range(num_test_traj):
            test_traj = all_test_trajs[i]
            ckpt_ours = evals.evaluate_on_traj(grid, controller_list, dQPTH_layer, test_traj, T, H, batt_cost, curt_change_cost, curt_net_cost, bus_slack_cost, line_slack_cost)
            our_losses[i] = ckpt_ours["total_losses"]
        ckpt[test_skew_mag] = our_losses
        test_skew_mag = 0

    pool.close()
    ckpt["failed_tests"] = failed_tests
    print("final save")
    torch.save(ckpt, f"{folder}checkpoint_{tag}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # decides which set of forecast/test scenarios
    parser.add_argument("--noise_mag", type=float, default=4.0)
    parser.add_argument("--offset", type=int, default=180)
    parser.add_argument("--radius", type=float, default=0.2)
    parser.add_argument("--forecast_seed", type=int, default=0)

    # decides optimization parameters
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer_type", type=str, default="clipped_gd")
    parser.add_argument("--lr_schedule", type=str, default="plateau")
    parser.add_argument("--lr_decay_step", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_grad_norm", type=float, default=3e4)
    args = parser.parse_args()
    main(args)