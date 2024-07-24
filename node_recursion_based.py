# FT 19/7/24
# Custom kd-tree implementation

import numpy as np
import matplotlib.pyplot as plt
import time     # For testing

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax.tree_util import register_pytree_node_class

# NOTE It looks like there's a bug in this code - sometimes it doesn't match.

# Kd-tree node which stores an additional numerical value.
@register_pytree_node_class
class Node:
    # This signature needs to be full (everything present) because of Jax.
    # Has to be in this order, because of how children and aux_data are treated
    # (see tree_unflatten method).
    def __init__(self, state, left, right, val, depth, state_dim, depth_wrapped=None):
        self.state = state
        self.left = left
        self.right = right
        
        self.val = val

        self.depth = depth
        self.state_dim = state_dim
        if depth_wrapped is not None:
            self.depth_wrapped = depth_wrapped
        else:
            self.depth_wrapped = depth % self.state_dim
    
    # For testing
    def create_left_child(self, state, val):
        self.left = Node(state, None, -1, -1, val)
        return self.left
    
    # For testing
    def create_right_child(self, state, val):
        self.right = Node(state, None, -1, -1, val)
        return self.right
    
    def get_depth_wrapped(self):
        return self.depth_wrapped
    
    def val_at_depth(self):
        return self.state[self.depth_wrapped]
    
    # Registering as a PyTree for Jax
    def tree_flatten(self):
        children = (self.state, self.left, self.right)      # arrays / dynamic values
        aux_data = {'val': self.val, 'depth': self.depth, 'state_dim': self.state_dim, 'depth_wrapped': self.depth_wrapped}     # static values
        return (children, aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

# n-dim KDTree with depth information in Node
# This is a generic kd-tree class and can be used in other problems
class KDTree:
    def __init__(self, coords, vals):
        # Create tree
        self.node_type = Node
        self.state_dim = coords.shape[1]
        self.root = self.build(coords, vals)
    
    def build(self, coords, vals):
        # Build tree or nodes recursively
        build_start = time.time()
        root = self._build_recursive(coords, vals, 0)
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
    
    def _build_recursive(self, coords, vals, depth):
        # Base case
        # If a leaf
        if len(coords) == 0:
            return None
        if len(coords) == 1:
            print(".", end="")
            return Node(coords[0], None, None, vals[0], depth, self.state_dim, None)
        
        # Find the middle point for the given axis. Create a node for this. [Set this as the root]
        lower, split, upper = self.partition(coords, vals, depth % self.state_dim)

        # Create centre node
        state, val = split
        node = Node(state, None, None, val, depth, self.state_dim, None)
        print(".", end="")

        # Find the left-hand values (give it the left-hand list)
        node.left = self._build_recursive(*lower, depth + 1)
        
        # Find the right-hand values (give it the right-hand list)
        node.right = self._build_recursive(*upper, depth + 1)

        # The final returned node will be the root of the tree
        return node

    def find_nearest_points(self, target_coords, k=1):
        min_dists = jnp.ones(k)*jnp.inf
        closest_coords = jnp.zeros((k, self.state_dim), dtype=jnp.float64)

        min_dists, closest_coords = self._find_nearest_points(self.root, target_coords, min_dists, closest_coords)
        return closest_coords, min_dists
    
        # My implementation, based on Wikipedia algorithm description: https://en.wikipedia.org/wiki/K-d_tree
    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def _find_nearest_points(current_node, target, min_dists, closest_coords):
        # Base case: if the node is a leaf node, stop. Calculate the distance and save it if it's the best so far.
        # Probably don't need the 'is None' here, but it's there for safety.
        if (current_node.left is None) and (current_node.right is None):    # Is a leaf
            # dist operates on arrays - e.g. it requires states rather than nodes.
            current_state = current_node.state
            current_node_dist = KDTree.dist(target, current_state)
            min_dists, closest_coords = KDTree.update_closest(min_dists, closest_coords, current_node_dist, current_state)
            return min_dists, closest_coords

        else:
            # If the state should be inserted on the left..
            if KDTree.leq_at_depth(target, current_node):
                return KDTree.search(target, current_node, current_node.left, current_node.right, min_dists, closest_coords)
            else:
                # Node should be inserted on the right
                return KDTree.search(target, current_node, current_node.right, current_node.left, min_dists, closest_coords)
            
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
    
    # Compares the two states with the appropriate variable for the depth. True if left (<= 0), False if right.
    # leq = less than or equal to
    # Right-hand argument must be a node, so that depth information is available.
    @staticmethod
    @jax.jit
    def leq_at_depth(left_state, right_node):
        depth_ind = right_node.get_depth_wrapped()
        jax.lax.cond(left_state[depth_ind] <= right_node.val_at_depth(), lambda: True, lambda: False)
    
    # e.g. if exploring left child, then right: first_branch_node will be current_node.left and second_branch_node will be current_node.right.
    @staticmethod
    @jax.jit
    def search(target, current_node, first_branch_node, second_branch_node, min_dists, closest_coords):
        # Node should be inserted on the right
        if first_branch_node is not None:
            min_dists, closest_coords = KDTree._find_nearest_points(first_branch_node, target, min_dists, closest_coords)
        
        # This has now returned to the level above
        # Check to see if the distance to this node is closer.
        current_state = current_node.state
        current_node_dist = KDTree.dist(target, current_state)
        min_dists, closest_coords = KDTree.update_closest(min_dists, closest_coords, current_node_dist, current_state)

        # Check the distance to the coordinate splitting plane. If it's less than current_min_dist, have to explore the other branch of the tree. Could be a closer point on the other side of the splitting plane.
        min_dists, closest_coords = jax.lax.cond(jnp.logical_and((min_dists[-1] > KDTree._dist_to_splitting_plane(target, current_node)), (second_branch_node is not None)),
            # Have to explore the other side
            lambda current_node, target, min_dists, closest_coords: KDTree._find_nearest_points(second_branch_node, target, min_dists, closest_coords),
            lambda current_node, target, min_dists, closest_coords: (min_dists, closest_coords),
            current_node, target, min_dists, closest_coords)
        
        # This node is now fully explored, can return
        return min_dists, closest_coords
    
    # Want to find the distance between the target and the partition plane represented by the current node.
    @staticmethod
    @partial(jax.jit, static_argnums=(1,))
    def _dist_to_splitting_plane(target, current_node):
        switch_var = current_node.get_depth_wrapped()
        current_node_state = current_node.state

        return jnp.abs(current_node_state[switch_var] - target[switch_var])
    
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
        closest_states_kdt, min_dists_kdt = tree.find_nearest_points(target, k)
        kd_time = time.time() - start_time
        
        # Calculate the distance between the target and each point, and see if the selected point (which will be definitely be correct) is the same one
        # as found by the kd-tree.
        # Iterate through the points. Store in dictionary indexed by distance.
        print()
        print("Finding brute force")

        # Dictionaries can't be used by Jax, so developing an alternative.
        start_time = time.time()
        min_dists = jnp.ones(k)*jnp.inf
        min_coords = jnp.zeros((k, self.state_dim))
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
            if self.state_dim == 2:
                fig = plt.figure()
                ax = fig.add_subplot()
            elif self.state_dim == 3:
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
    # Trying with a grid of regularly spaced points (to check that it doesn't cause a problem having so many identical coordinates).
    intv = 5
    X, Y, Z = np.meshgrid(np.arange(-10, 10, intv), np.arange(-10, 10, intv), np.arange(-10, 10, intv))
    coords = jnp.asarray(np.stack((X.flatten(), Y.flatten(), Z.flatten())).T)
    num_pts = len(coords)

    # Node data values - these don't matter for now. 
    vals = np.arange(num_pts)
    # Build tree (in a balanced way)
    tree = KDTree(coords, vals)

    plt.ion()

    k = 10
    if True:
        counter = 0
        max_tests = 20000
        while counter < max_tests:
            if not tree.test_current(coords, jnp.asarray((np.random.randn(1, 3)*10-5)[0], dtype=jnp.float64), k, plot_level=2):
                print("Mismatch detected")
                continue
            counter += 1
            print(f"Successfully matched: {counter}")

    print("Done")
