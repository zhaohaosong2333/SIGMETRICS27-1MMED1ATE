"""Random forest / GBDT model"""

from typing import Dict, List
from .decision_tree import DecisionTree


class RandomForest:
    """Random forest or GBDT model."""
    
    def __init__(self, name: str):
        self.name = name
        self.tree_num = 0
        self.branch_num = 0
        self.leaf_num = 0
        self.tree_dict: Dict[str, DecisionTree] = {}
        self.feats = []
        self.threshs = []
        self.branches = []
    
    def set_labels(self, labels):
        """Sets the label."""
        self.labels = labels
    
    def add_decision_tree_from_line(self, line: str):
        """Adds a decision tree from a line of text."""
        from ..compiler.extractor import create_node
        
        # Update tree index
        tree_id = self.tree_num + 1
        self.tree_num = tree_id
        
        # Create a decision tree object
        root = create_node(line)
        dt = DecisionTree(root, tree_id, self.branch_num, self.leaf_num)
        
        # Update forest
        self.tree_dict[f"T{tree_id}"] = dt
        self.branch_num += len(dt.get_branch_names())
        self.leaf_num += len(dt.get_leaf_names())
        self.branches.append(dt.get_branch_names())
        
        dt_feats, dt_threshs = dt.get_feats_and_threshs()
        self.feats.append(dt_feats)
        self.threshs.append(dt_threshs)
    
    def initialize(self):
        """Initializes the forest."""
        self.branches = _serialize(self.branches)
        self.feats = _serialize(self.feats)
        self.threshs = _serialize(self.threshs)
        self.feats_num = len(set(self.feats))
        self.max_level = max([tree.max_level for tree in self.tree_dict.values()])
    
    def get_forest_thresh_and_k(self, expected_n: int = -1):
        """Returns the forest thresholds and K value."""
        # Initialize data structures
        thresh = [[] for _ in range(self.feats_num)]
        node = [[] for _ in range(self.feats_num)]
        
        self.feats = _update_feats(self.feats)
        
        for i in range(len(self.feats)):
            current_feat = self.feats[i]
            thresh[current_feat].append(self.threshs[i])
            node[current_feat].append(self.branches[i])
        
        _padding(thresh)
        _padding(node, "S")
        
        self.thresh = _serialize(thresh)
        self.branch = _serialize(node)
        self.k = len(thresh[0])
        
        if expected_n > 0:
            expected_length = expected_n * self.k
            while len(self.thresh) < expected_length:
                self.thresh.append(0)
                self.branch.append("S")
    
    def get_tree_names(self) -> List[str]:
        """Returns the names of all trees."""
        tree_list = list(self.tree_dict.keys())
        return sorted(tree_list, key=lambda x: int(x[1:]))
    
    def get_forest_lvls_and_masks(self):
        """Returns the forest lvls and masks."""
        self.masks = [[] for _ in range(self.max_level)]
        self.lvl_decisions = [[] for _ in range(self.max_level)]
        
        tree_list = self.get_tree_names()
        
        for tree_name in tree_list:
            tree = self.tree_dict[tree_name]
            for level in range(tree.max_level):
                tree.get_lvl_decision_and_mask(level)
            
            _fill_level_processing_matrix(tree, self.max_level)
            _merge_mat(self.masks, tree.masks)
            _merge_mat(self.lvl_decisions, tree.lvl_decisions)
    
    def generate_copse_model(self):
        """generate COPSE model"""
        from ..models.copse import COPSEModel
        from ..utils.helpers import generate_reorder_matrix
        
        current_copse_model = COPSEModel()
        
        current_copse_model.thresh = self.thresh
        current_copse_model.q = len(current_copse_model.thresh)
        
        current_copse_model.masks = self.masks
        current_copse_model.d = len(current_copse_model.masks)
        
        current_copse_model.reshuf = generate_reorder_matrix(self.branch, self.branches)
        current_copse_model.b = len(current_copse_model.reshuf)
        
        current_copse_model.k = len(self.thresh) // len(set(self.feats))
        
        current_copse_model.lvls = [
            generate_reorder_matrix(self.branches, current_lvl_decision) 
            for current_lvl_decision in self.lvl_decisions
        ]
        
        self.copse_model = current_copse_model
        return current_copse_model


def _serialize(mat):
    """Serializes a matrix into a vector."""
    serialized_vec = []
    for vec in mat:
        serialized_vec.extend(vec)
    return serialized_vec


def _padding(mat, padding_element=0):
    """Pads the matrix so that all rows have equal length."""
    k = max([len(vec) for vec in mat])
    for vec in mat:
        while len(vec) < k:
            vec.append(padding_element)


def _update_feats(feats: List[int]) -> List[int]:
    """Updates feature indices and removes gaps."""
    largest_index = max(feats)
    unique_count = len(set(feats))
    
    if largest_index != unique_count - 1:
        absent_indices = [i for i in range(largest_index) if i not in feats]
        for i in range(len(feats)):
            feats[i] -= sum(1 for absent in absent_indices if feats[i] > absent)
    
    return feats


def _fill_level_processing_matrix(tree, forest_max_level):
    """Pads the level processing matrix."""
    if len(tree.masks) != len(tree.lvl_decisions):
        raise ValueError("Current Tree's Masks and LvlDecisions Do Not Have The Same Size")
    
    if len(tree.masks) < forest_max_level:
        current_level = len(tree.masks)
        for i in range(forest_max_level - current_level):
            tree.masks.append(tree.masks[current_level - 1])
            tree.lvl_decisions.append(tree.lvl_decisions[current_level - 1])


def _merge_mat(mat_a, mat_b):
    """Merges two matrices."""
    if len(mat_a) != len(mat_b):
        raise ValueError("Try to Merge Two Matrix of Different Size")
    for i in range(len(mat_a)):
        mat_a[i].extend(mat_b[i])
