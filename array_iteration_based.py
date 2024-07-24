# FT 23/7/24
# Array- and iteration-based kd-tree.

# Currently uses Nodes to create tree structure, then flattens into arrays.

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import time
import matplotlib.pyplot as plt
from functools import partial

# Used for tree creation
class Node:
    def __init__(self, state, parent, val, depth):
        self.state = state
        self.parent = parent
        self.val = val
        self.depth = depth

        self.left = None
        self.right = None

class KDTree:
    def __init__(self, init_coords, init_vals):
        self.num_pts, self.coord_dim = init_coords.shape
        
        self.coords = jnp.empty((self.num_pts, self.coord_dim), dtype=jnp.float64)
        self.vals = jnp.empty(self.num_pts, dtype=jnp.float64)
        self.depth = jnp.empty(self.num_pts, dtype=jnp.int32)
        self.left = jnp.empty(self.num_pts, dtype=jnp.int32)
        self.right = jnp.empty(self.num_pts, dtype=jnp.int32)

        # Construct tree arrays
        # Build kd-tree (recursively)
        root = self.build(init_coords, init_vals)
        self.number = 0
        print("Numbering nodes")
        self.number_nodes(root)
        # Populate tree arrays
        print("Populating tree arrays")
        self.construct_tree_arrays(root)
        
        self.root_ind = root.id
        self.max_depth = np.max(self.depth)

    def build(self, coords, vals):
        # Build tree or nodes recursively
        build_start = time.time()
        root = self._build_recursive(coords, vals, None, 0)
        print(f"Build time: {time.time() - build_start}s")

        return root
    
    @staticmethod
    def partition(coords, vals, sort_dim):
        # Sort in the dimension of interest
        #  -> Find the indices which yield a sorted coordinate array in the dimension of interest
        sort_arr = coords.T[sort_dim]
        sort_inds = jnp.lexsort((sort_arr,))

        # Sort the coordinate and value arrays
        coords_sorted = coords[sort_inds]
        vals_sorted = vals[sort_inds]

        # Find the median value
        # Using floor_divide so that the result is an integer and can be used for indexing.
        split_ind = jnp.floor_divide(len(coords_sorted), 2)
        # If there are repeated values of the median, find the rightmost occurance. This is required
        # since the tree search works on '<=': it's required that all left children of the split node
        # have a value (for the dimension) less than or equal to the split point, and all of the nodes
        # to the right must have a greater value.
        median = coords_sorted[split_ind, sort_dim]
        max_ind = np.max(jnp.where(coords_sorted[:, sort_dim] == median))

        # Split against the max_ind
        return (coords_sorted[:max_ind], vals_sorted[:max_ind]), \
            (coords_sorted[max_ind], vals_sorted[max_ind]), \
                (coords_sorted[max_ind + 1:], vals_sorted[max_ind + 1:])
    
    # Requires that the data is sorted in every axis.
    def _build_recursive(self, coords, vals, parent, depth):
        # Base case
        # If a leaf
        if len(coords) == 0:
            return None
        if len(coords) == 1:
            print(".", end="")
            return Node(coords[0], parent, vals[0], depth)

        # Find the middle point for the given axis. Create a node for this. [Set this as the root]
        lower, split, upper = self.partition(coords, vals, depth % self.coord_dim)

        # Create centre node
        state, val = split
        node = Node(state, parent, val, depth)
        print(".", end="")

        # Find the left-hand values (give it the left-hand list)
        node.left = self._build_recursive(*lower, node, depth + 1)
        
        # Find the right-hand values (give it the right-hand list)
        node.right = self._build_recursive(*upper, node, depth + 1)

        # The final returned node will be the root of the tree
        return node
    
    # If a leaf, add the number.
    # If not a leaf, go left.
    # Then go right.
    def number_nodes(self, current_node):
        if current_node.left is not None:
            self.number_nodes(current_node.left)
        if current_node.right is not None:
            self.number_nodes(current_node.right)
        
        current_node.id = self.number
        self.number += 1
    
    # Traverse tree and construct arrays
    def construct_tree_arrays(self, current_node):
        if current_node.left is not None:
            self.construct_tree_arrays(current_node.left)
        if current_node.right is not None:
            self.construct_tree_arrays(current_node.right)
        
        ind = current_node.id
        self.coords = self.coords.at[ind, :].set(current_node.state)
        self.vals = self.vals.at[ind].set(current_node.val)
        self.depth = self.depth.at[ind].set(current_node.depth)
        self.left = self.left.at[ind].set(current_node.left.id if current_node.left is not None else -1)
        self.right = self.right.at[ind].set(current_node.right.id if current_node.right is not None else -1)

        #if current_node.depth > self.max_depth:
        #    self.max_depth = current_node.depth
    
    @staticmethod
    @jax.jit
    def append(arr, ind, value):
        return arr.at[ind + 1].set(value), ind + 1
    
    @staticmethod
    @jax.jit
    def pop(arr, ind):
        return arr[ind], ind - 1

    def find_closest_iter(self, target, k=1):
        started = False
        returning = False

        prev_inds = jnp.empty(self.max_depth, dtype=jnp.int32)
        call_pts = jnp.empty(2*self.max_depth, dtype=jnp.int32)
        prev_dirs = jnp.empty(2*self.max_depth, dtype=jnp.int32)

        # List index pointers
        prev_inds_head = -1
        call_pts_head = -1
        prev_dirs_head = -1
        current_ind = self.root_ind

        min_dists = jnp.ones(k)*jnp.inf
        closest_coords = jnp.zeros((k, self.coord_dim), dtype=jnp.float64)

        while (prev_inds_head >= 0) or not ((prev_inds_head >= 0) or started):
            started = True

            if not returning:
                # If a leaf
                if (self.left[current_ind] == -1) and (self.right[current_ind] == -1):
                    current_coord = self.coords[current_ind]
                    dist = self.dist(target, current_coord)
                    min_dists, closest_coords = self.update_closest(min_dists, closest_coords, dist, current_coord)
                    
                    returning = True                # Flag to let the algorithm know that it's returning from a previous level.
                    continue
                else:
                    # If target is found to the left of the current node
                    current_coord = self.coords[current_ind]
                    if target[(prev_inds_head + 1) % self.coord_dim] <= current_coord[(prev_inds_head + 1) % self.coord_dim]:
                        prev_inds, prev_inds_head = self.append(prev_inds, prev_inds_head, current_ind)
                        call_pts, call_pts_head = self.append(call_pts, call_pts_head, 0)
                        prev_dirs, prev_dirs_head = self.append(prev_dirs, prev_dirs_head, 0)
                        
                        left_ind = self.left[current_ind]
                        if left_ind >= 0:
                            returning = False
                            current_ind = left_ind
                            continue
                        else:
                            # Skip to next part of condition
                            returning = True

                    else:   # Go right
                        prev_inds, prev_inds_head = self.append(prev_inds, prev_inds_head, current_ind)
                        call_pts, call_pts_head = self.append(call_pts, call_pts_head, 0)
                        prev_dirs, prev_dirs_head = self.append(prev_dirs, prev_dirs_head, 1)
                        
                        right_ind = self.right[current_ind]
                        if right_ind >= 0:
                            returning = False
                            current_ind = right_ind
                            continue
                        else:
                            # Skip to next part of condition
                            returning = True
            else:
                current_ind, prev_inds_head = self.pop(prev_inds, prev_inds_head)
                return_point, call_pts_head = self.pop(call_pts, call_pts_head)
                prev_dir, prev_dirs_head = self.pop(prev_dirs, prev_dirs_head)

                if return_point == 0:
                    # Check current node
                    current_coord = self.coords[current_ind]
                    dist = self.dist(target, current_coord)
                    min_dists, closest_coords = self.update_closest(min_dists, closest_coords, dist, current_coord)

                    # If the distance is closer, go down other branch.
                    if jnp.abs(target[(prev_inds_head + 1) % self.coord_dim] - current_coord[(prev_inds_head + 1) % self.coord_dim]) < min_dists[-1]:
                        if prev_dir == 0:
                            right_ind = self.right[current_ind]
                            if right_ind >= 0:
                                prev_inds, prev_inds_head = self.append(prev_inds, prev_inds_head, current_ind)
                                returning = False
                                call_pts, call_pts_head = self.append(call_pts, call_pts_head, 1)
                                prev_dirs, prev_dirs_head = self.append(prev_dirs, prev_dirs_head, prev_dir)
                                current_ind = right_ind
                                continue
                        else:
                            left_ind = self.left[current_ind]
                            if left_ind >= 0:
                                prev_inds, prev_inds_head = self.append(prev_inds, prev_inds_head, current_ind)
                                returning = False
                                call_pts, call_pts_head = self.append(call_pts, call_pts_head, 1)
                                prev_dirs, prev_dirs_head = self.append(prev_dirs, prev_dirs_head, prev_dir)
                                current_ind = left_ind
                                continue

                # if CallPoint.SECONDARY, don't do anything.
                returning = True                # Flag to let the algorithm know that it's returning from a previous level.
                
        return closest_coords, min_dists
    
    @staticmethod
    @jax.jit
    def dist(a, b):
        return jnp.linalg.norm(a - b)

    # For testing method
    @staticmethod
    def is_matched(closest, truth):
        if np.array_equal(closest, truth):
            return True, "matched"
        else:
            return False, "not matched"
    
    # Test with the current tree
    # plot_level: 0 = never, 1 = if mismatch, 2 = always
    # k: number of nearest neighbours to find and compare against
    def test_current(self, all_coords, target, k, plot_level=2):
        plt.close('all')

        print(f"Target point: {target}")
        
        print("Finding kd: new method")
        start_time = time.time()
        closest_states_kdt, min_dists_kdt = self.find_closest_iter(target, k)
        kd_time = time.time() - start_time

        # How to test?
        # Calculate the distance between the target and each point, and see if the selected point (which will be definitely be correct) is the same one
        # as found by the kd-tree.
        # Iterate through the points. Store in dictionary indexed by distance.
        print()
        print("Finding brute force")

        # Dictionaries can't be used by Jax, so developing an alternative.
        start_time = time.time()
        min_dists = jnp.ones(k)*jnp.inf
        min_coords = jnp.zeros((k, self.coord_dim))
        for pt in all_coords:
            dist = tree.dist(pt, target)
            insert_ind = jnp.searchsorted(min_dists, dist)
            if insert_ind < k:
                min_dists = jnp.insert(min_dists, insert_ind, dist)
                min_dists = min_dists[:k]
                min_coords = jnp.insert(min_coords, insert_ind, pt, axis=0)
                min_coords = min_coords[:k]
        brute_time = time.time() - start_time
        
        print("Distances to closest:")
        coordinate_comparison = list(zip(min_dists_kdt, min_dists))
        print("(kd-tree, brute force)")
        [print(comp) for comp in coordinate_comparison]

        # Print match results
        matches = []
        for closest_state, found_closest_pt in zip(closest_states_kdt, min_coords):
            match, match_str = self.is_matched(closest_state, found_closest_pt)
            matches.append(match)
            print(match_str)
        
        matched = not (False in matches)
        print("**matched**" if matched else "__not matched__")

        print(f"Elapsed times:")
        print(f"      kd-tree: {kd_time}")
        print(f"  Brute force: {brute_time}")
        
        if (plot_level == 2) or ((plot_level == 1) and not matched):
            # Create figure
            if self.coord_dim == 2:
                fig = plt.figure()
                ax = fig.add_subplot()
            elif self.coord_dim == 3:
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
            else:
                print("NOTE: Can only visualise if dim = 2 or 3")
            
            # Plot
            ax.scatter(*all_coords.T, alpha=0.3)
            ax.scatter(*target, marker='x', c='orange')
            ax.scatter(*closest_states_kdt.T, c='green', marker='^')
            # Plot found closest points
            ax.scatter(*min_coords.T, c='red', marker='*')
            ax.set_aspect('equal')
            plt.show(block=True)

        return matched

if __name__ == "__main__":
    intv = 5
    X, Y, Z = np.meshgrid(np.arange(-10, 10, intv), np.arange(-10, 10, intv), np.arange(-10, 10, intv))
    coords = jnp.asarray(np.stack((X.flatten(), Y.flatten(), Z.flatten())).T)
    num_pts = len(coords)

    vals = np.arange(num_pts)
    # Build tree (in a balanced way)
    tree = KDTree(coords, vals)

    print(f"\ncoords: {tree.coords}")
    print(f"\nvals: {tree.vals}")
    print(f"\nleft: {tree.left}")
    print(f"\nright: {tree.right}")

    k = 10
    if True:
        counter = 0
        max_tests = 20000
        while counter < max_tests:
            target = jnp.asarray((np.random.randn(1, 3)*10-5)[0], dtype=jnp.float64)
            if not tree.test_current(coords, target, k, plot_level=2):
                print("Mismatch detected")
                continue
            counter += 1
            print(f"Successfully matched: {counter}")

    print("Done")