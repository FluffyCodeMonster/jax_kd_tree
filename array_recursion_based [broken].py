# FT 23/7/24
# Array- and iteration-based kd-tree.

# Currently uses Nodes to create tree structure, then flattens into arrays.

# NOTE This doesn't work at the moment - it just recurses indefinitely during jitting.

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

    def find_closest(self, target_coords, k=1):
        min_dists = jnp.ones(k)*jnp.inf
        closest_coords = jnp.zeros((k, self.coord_dim), dtype=jnp.float64)

        min_dists, closest_coords = self._find_closest(self.root_ind, target_coords, min_dists, closest_coords)
        return closest_coords, min_dists
    
    # My implementation, based on Wikipedia algorithm description: https://en.wikipedia.org/wiki/K-d_tree
    @partial(jax.jit, static_argnums=(0,))
    def _find_closest(self, node_ind, target, min_dists, closest_coords):
        # Base case: if the node is a leaf node, stop. Calculate the distance and save it if it's the best so far.
        jax.lax.cond(jnp.logical_and((self.left[node_ind] == -1), (self.right[node_ind] == -1)),
                     lambda node_ind, target, min_dists, closest_coords: self.handle_leaf(node_ind, target, min_dists, closest_coords),
                     lambda node_ind, target, min_dists, closest_coords: jax.lax.cond(self.target_leq_at_depth(target, node_ind),
                                                                                      self.search(target, node_ind, self.left[node_ind], self.right[node_ind], min_dists, closest_coords),  # If the state should be inserted on the left..
                                                                                      self.search(target, node_ind, self.right[node_ind], self.left[node_ind], min_dists, closest_coords),  # Node should be inserted on the right..
                                                                                      node_ind, target, min_dists, closest_coords),
                     node_ind, target, min_dists, closest_coords)
        
    @partial(jax.jit, static_argnums=(0,))
    def target_leq_at_depth(self, target, node_ind):
        depth_ind = self.depth[node_ind] % self.coord_dim
        jax.lax.cond(target[depth_ind] <= self.coords[node_ind][depth_ind],
                     lambda: True,
                     lambda: False)
    
    @partial(jax.jit, static_argnums=(0,))
    def handle_leaf(self, node_ind, target, min_dists, closest_coords):
        # dist operates on arrays - e.g. it requires states rather than nodes.
        current_coord = self.coords[node_ind]
        dist = self.dist(target, current_coord)
        min_dists, closest_coords = self.update_closest(min_dists, closest_coords, dist, current_coord)
        return min_dists, closest_coords
    
    @staticmethod
    @jax.jit
    def update_closest(min_dists, closest_coords, dist, coord):
        return jax.lax.cond(dist < min_dists[-1], 
                     lambda min_dists, closest_coords, dist, coord: KDTree.insert_closer(min_dists, closest_coords, dist, coord),
                     lambda min_dists, closest_coords, dist, coord: (min_dists, closest_coords),
                     min_dists, closest_coords, dist, coord)
    
    @staticmethod
    @jax.jit
    def insert_closer(min_dists, closest_coords, dist, coord):
        insert_ind = jnp.searchsorted(min_dists, dist)
        # Insert distance
        min_dists = jnp.insert(min_dists[:-1], insert_ind, dist)

        # Insert node coordinates
        closest_coords = jnp.insert(closest_coords[:-1], insert_ind, coord, axis=0)

        return min_dists, closest_coords
    
    # e.g. if exploring left child, then right: first_branch_node will be current_node.left and second_branch_node will be current_node.right.
    @partial(jax.jit, static_argnums=(0,))
    def search(self, target, node_ind, first_branch_node_ind, second_branch_node_ind, min_dists, closest_coords):        
        min_dists, closest_coords = jax.lax.cond(first_branch_node_ind >= 0,
                                                 self._find_closest(first_branch_node_ind, target, min_dists, closest_coords),
                                                 lambda first_branch_node_ind, target, min_dists, closest_coords: (min_dists, closest_coords),
                                                 first_branch_node_ind, target, min_dists, closest_coords)

        # This has now returned to the level above
        # Check to see if the distance to this node is closer.
        current_coord = self.coords[node_ind]
        dist = self.dist(target, current_coord)
        min_dists, closest_coords = self.update_closest(min_dists, closest_coords, dist, current_coord)

        # Check the distance to the coordinate splitting plane. If it's less than current_min_dist, have to explore the other branch of the tree. Could be a closer point on the other side of the splitting plane.
        min_dists, closest_coords = jax.lax.cond(jnp.logical_and((min_dists[-1] > self._dist_to_splitting_plane(target, node_ind)), (second_branch_node_ind >= 0)),
            # Have to explore the other side
            lambda second_branch_node_ind, target, min_dists, closest_coords: self._find_closest(second_branch_node_ind, target, min_dists, closest_coords),
            lambda second_branch_node_ind, target, min_dists, closest_coords: (min_dists, closest_coords),
            second_branch_node_ind, target, min_dists, closest_coords)
        
        # This node is now fully explored, can return
        return min_dists, closest_coords
    
    # Want to find the distance between the target and the partition plane represented by the current node.
    @partial(jax.jit, static_argnums=(0,))
    def _dist_to_splitting_plane(self, target, node_ind):
        switch_var = self.depth[node_ind] % self.coord_dim
        current_coord = self.coords[node_ind]
        return jnp.abs(current_coord[switch_var] - target[switch_var])
    
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
        closest_states_kdt, min_dists_kdt = self.find_closest(target, k)
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