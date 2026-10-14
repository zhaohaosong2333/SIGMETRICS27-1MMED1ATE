"""General helper utilities for 1MMEDIATE"""

import time
from pathlib import Path
from typing import List


class Timer:
    """Timer for measuring execution time"""
    
    def __init__(self):
        self.timestamps = []
    
    def timing(self):
        """Record timestamp"""
        self.timestamps.append(time.time() * 1000)  # milliseconds
    
    def get_duration(self) -> List[float]:
        """Get time intervals"""
        if len(self.timestamps) < 2:
            return []
        return [self.timestamps[i+1] - self.timestamps[i] 
                for i in range(len(self.timestamps)-1)]


def check_and_create_folder(folder_path: str):
    """Checks and creates a directory."""
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    
def generate_selection_vector(target_element, source_vector: List) -> List[int]:
    """Generate selection vector"""
    selection_vec = [0] * len(source_vector)
    target_index = source_vector.index(target_element)
    selection_vec[target_index] = 1
    return selection_vec


def generate_reorder_matrix(input_vec: List, output_vec: List) -> List[List[int]]:
    """Generate reorder matrix"""
    reorder_mat = []
    for target_output in output_vec:
        reorder_mat.append(generate_selection_vector(target_output, input_vec))
    return reorder_mat
