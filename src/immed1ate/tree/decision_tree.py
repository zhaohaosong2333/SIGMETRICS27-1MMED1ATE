"""Implementation of a single decision tree."""

from typing import Dict, List, Tuple
from .nodes import Branch, Leaf


class DecisionTree:
    """Decision tree class supporting DFS traversal and hierarchical analysis."""
    
    def __init__(self, root_branch: Branch, tree_id: int, 
                 branch_id: int = 0, leaf_id: int = 0):
        self.tree_id = tree_id
        self.branch_dict: Dict[str, Branch] = {}
        self.leaf_dict: Dict[str, Leaf] = {}
        self.branch_id = branch_id
        self.leaf_id = leaf_id
        self.max_level = -999
        self.lvl_decisions = []
        self.masks = []
        self.inr = []  # In-order traversal
        
        self._dfs(root_branch)
        self.max_level += 1
        self._idfs(root_branch)
    
    def _name_branch(self) -> str:
        """Generates branch node names."""
        name = f"d{self.branch_id}"
        self.branch_id += 1
        return name
    
    def _name_leaf(self) -> str:
        """Generates leaf node names."""
        name = f"L{self.leaf_id}"
        self.leaf_id += 1
        return name
    
    def _add_node_to_tree(self, node):
        """Adds a node to the tree and assigns a name."""
        if node.is_leaf():
            node.name = self._name_leaf()
            self.leaf_dict[node.name] = node
        else:
            node.name = self._name_branch()
            self.branch_dict[node.name] = node
            self.max_level = max(node.level, self.max_level)
    
    def _dfs(self, node):
        """Depth-first search."""
        self._add_node_to_tree(node)
        if not node.is_leaf():
            self._dfs(node.left)
            self._dfs(node.right)
    
    def _idfs(self, node):
        """In-order traversal."""
        if node.is_leaf():
            return
        self._idfs(node.left)
        self.inr.append(node.name)
        self._idfs(node.right)
    
    def get_lvl_decision_and_mask(self, level: int):
        """Returns the decisions and mask for a specified level."""
        lvl_decision = []
        mask = []
        
        for branch in self.inr:
            branch_node = self.branch_dict[branch]
            
            if branch_node.level == level:
                controlled_by_left_num = _get_controlled_leaf_num(branch_node.left)
                controlled_by_right_num = _get_controlled_leaf_num(branch_node.right)
                lvl_decision.extend([branch] * (controlled_by_left_num + controlled_by_right_num))
                mask.extend([0] * controlled_by_left_num + [1] * controlled_by_right_num)
            
            if branch_node.level < level and branch_node.left.is_leaf():
                controlled_by_left_num = _get_controlled_leaf_num(branch_node.left)
                lvl_decision.extend([branch] * controlled_by_left_num)
                mask.extend([0] * controlled_by_left_num)
                
            if branch_node.level < level and branch_node.right.is_leaf():
                controlled_by_right_num = _get_controlled_leaf_num(branch_node.right)
                lvl_decision.extend([branch] * controlled_by_right_num)
                mask.extend([1] * controlled_by_right_num)
        
        self.lvl_decisions.append(lvl_decision)
        self.masks.append(mask)
    
    def get_feats_and_threshs(self) -> Tuple[List[int], List[int]]:
        """Returns all features and thresholds."""
        branch_list = sorted(self.branch_dict.keys(), key=lambda x: int(x[1:]))
        feats = []
        threshs = []
        for branch_name in branch_list:
            node = self.branch_dict[branch_name]
            feats.append(int(node.feature_id))
            threshs.append(int(node.thresh))
        return feats, threshs
    
    def display(self):
        """Displays tree structure information."""
        print("TREE ID : ", self.tree_id)
        node_list = list(self.branch_dict.keys())
        for nodename in node_list:
            node = self.branch_dict[nodename]
            print(f"--L{node.level} Branch {nodename}: F{node.feature_id} < {node.thresh}")
    
    def get_branch_names(self) -> List[str]:
        """Returns all branch node names."""
        branch_list = list(self.branch_dict.keys())
        return sorted(branch_list, key=lambda x: int(x[1:]))
    
    def get_leaf_names(self) -> List[str]:
        """Returns all leaf node names."""
        leaf_list = list(self.leaf_dict.keys())
        return sorted(leaf_list, key=lambda x: int(x[1:]))


def _get_controlled_leaf_num(node) -> int:
    """Returns the number of leaf nodes controlled by a node."""
    if node.is_leaf():
        return 1
    return _get_controlled_leaf_num(node.left) + _get_controlled_leaf_num(node.right)
