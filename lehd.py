import os
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.cvrp_helpers import (
    create_spatial_support_adjacency_matrix,
    create_spatial_support_clustering_matrix,
    remove_cvrp_solution_padding,
    create_route_prototypes_matrix,
    tour_to_vehicle_routes,
    create_seed_labels,
)


def parse_cvrp_line(line):
    # Split the string into a list of items
    tokens = line.strip().split(",")

    # Find the indices of our key separator words
    idx_depot = tokens.index("depot")
    idx_customer = tokens.index("customer")
    idx_capacity = tokens.index("capacity")
    idx_demand = tokens.index("demand")
    idx_cost = tokens.index("cost")
    idx_node_flag = tokens.index("node_flag")

    # 1. Parse Coordinates (Depot + Customers)
    depot_coords = list(map(float, tokens[idx_depot + 1 : idx_customer]))
    customer_coords = list(map(float, tokens[idx_customer + 1 : idx_capacity]))

    # Combine them and reshape into an [N, 2] array
    coords = np.array(depot_coords + customer_coords).reshape(-1, 2)

    # 2. Parse Capacity (just an int, but float parsing is safer)
    capacity = int(float(tokens[idx_capacity + 1]))

    # 3. Parse Demand (numpy array, safer to parse as float then cast to int)
    demand = np.array(tokens[idx_demand + 1 : idx_cost], dtype=float).astype(int)
    
    # Safely pad the depot demand (0) if the dataset excludes it
    if len(demand) == len(coords) - 1:
        demand = np.insert(demand, 0, 0)

    # Parse Cost (optional, but good to have based on the string)
    cost = float(tokens[idx_cost + 1])

    # 4. Parse Node Flag and reshape to [?, 2]
    node_flag = (
        np.array(tokens[idx_node_flag + 1 :], dtype=int).reshape(2, -1).transpose()
    )

    return {
        "coords": coords,
        "capacity": capacity,
        "demand": demand,
        "cost": cost,
        "node_flag": node_flag,
    }


class LEHDDataset(Dataset):
    """
    Dataset for LEHD format CVRP problems.
    """

    def __init__(self, config: dict, shard_file: str):
        """
        Initializes the dataset.

        Args:
            config: Configuration dictionary.
            shard_file: Path to the text file dataset.
        """
        self.config = config
        self.shard_file = shard_file

        num_samples = self.config["training"]["num_samples"]

        # Build line offsets for random access without loading the whole file
        offsets_cache_file = self.shard_file + ".offsets.npy"
        if os.path.exists(offsets_cache_file):
            self.line_offsets = np.load(offsets_cache_file).tolist()
        else:
            self.line_offsets = []
            with open(self.shard_file, "rb") as f:
                while True:
                    offset = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    if line.strip():  # ignore empty lines
                        self.line_offsets.append(offset)

            # Save to cache for future runs
            try:
                np.save(offsets_cache_file, np.array(self.line_offsets))
            except Exception as e:
                print(
                    f"Warning: could not save offsets cache to {offsets_cache_file}: {e}"
                )

        self.num_problems = len(self.line_offsets)
        if num_samples is None:
            self.num_samples = self.num_problems
        else:
            self.num_samples = min(num_samples, self.num_problems)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Open the file on-the-fly to be safe across worker processes
        with open(self.shard_file, "rb") as f:
            f.seek(self.line_offsets[idx])
            line = f.readline().decode("utf-8")

        parsed = parse_cvrp_line(line)
        coords = parsed["coords"]
        capacity = parsed["capacity"]
        demand = parsed["demand"]
        node_flag = parsed["node_flag"]
        cost = parsed["cost"]

        # Build the solution array based on node_flag
        # 1 marks the start of the solution (route), and it ends before the next 1
        sol = [0]
        for node, is_start in node_flag:
            if is_start == 1 and len(sol) > 1 and sol[-1] != 0:
                sol.append(0)
            sol.append(node)
        if sol[-1] != 0:
            sol.append(0)

        unpadded_solution = np.array(sol, dtype=np.int32)

        # In LEHD, we just use a consecutive range for the node indices (problem)
        problem = np.arange(len(coords), dtype=np.int32)

        demand = np.array(demand, dtype=np.float32)
        demand = demand / capacity

        adj_matrix = create_spatial_support_adjacency_matrix(problem, unpadded_solution)
        clustering_matrix = create_spatial_support_clustering_matrix(
            problem, unpadded_solution
        )
        depot_coords = coords[0]
        customer_coords = coords[1:]

        problem = torch.tensor(problem).int()
        demand = torch.tensor(demand).float()
        capacity = torch.tensor(capacity).int()
        adj_matrix = torch.tensor(adj_matrix).int()
        clustering_matrix = torch.tensor(clustering_matrix).float()

        coords = torch.tensor(coords)
        distance_matrix = torch.cdist(coords.float(), coords.float(), p=2.0)

        # Determine exactly how many non-empty routes exist in the validation instance
        routes_list = tour_to_vehicle_routes(unpadded_solution.tolist())
        num_routes = sum(1 for r in routes_list if [n for n in r if n != 0])

        seed_labels, seed_mask = create_seed_labels(
            problem.numpy(), unpadded_solution, distance_matrix.numpy()
        )
        seed_labels = torch.tensor(seed_labels).float()
        seed_mask = torch.tensor(seed_mask).float()

        route_prototypes_matrix = create_route_prototypes_matrix(
            problem.numpy(), unpadded_solution, num_routes, seed_mask.numpy()
        )
        route_prototypes_matrix = torch.tensor(route_prototypes_matrix).float()
        depot_coords = torch.tensor(depot_coords).float()
        customer_coords = torch.tensor(customer_coords).float()
        gt_cost = torch.tensor(cost, dtype=torch.float32)

        return (
            problem,
            demand,
            capacity,
            adj_matrix,
            clustering_matrix,
            distance_matrix,
            route_prototypes_matrix,
            seed_labels,
            seed_mask,
            depot_coords,
            customer_coords,
            gt_cost,
        )
