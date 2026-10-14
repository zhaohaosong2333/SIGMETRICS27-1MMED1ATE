"""Decision tree node definitions"""


class Branch:
    """Branch node (internal node)"""
    
    def __init__(self, feature_id: int, thresh: int, level: int = 0):
        self.feature_id = feature_id
        self.thresh = thresh
        self.level = level
        self.left = None
        self.right = None
        self.name = ""
    
    def set_left(self, node):
        """Set left child node"""
        self.left = node
        
    def set_right(self, node):
        """Set right child node"""
        self.right = node
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node"""
        return False


class Leaf:
    """Leaf node"""
    
    def __init__(self, label_id: int):
        self.label_id = label_id
        self.name = ""
        
    def is_leaf(self) -> bool:
        """Check if this is a leaf node"""
        return True
