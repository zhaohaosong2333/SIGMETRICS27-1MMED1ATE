"""Model extractor: parses tree models from .mdl files."""

import re
from typing import Tuple

from ..tree.nodes import Branch, Leaf
from ..tree.random_forest import RandomForest
from ..models.copse import COPSEModel


# Global bracket list
LEFT_BRACKET_LIST = ["(", "[", "{"]
RIGHT_BRACKET_LIST = [")", "]", "}"]

# Pattern definition
FEATURE_PATTERN = r'\{feat = F (\d+), val = (\d+)\}'
LEAF_PATTERN = r'Label \(L (\d+)\)'
INDICATOR_LIST = ["Branch", "threshold = Threshold", "left =", "right ="]

TEST_INFO = False


def create_node(line: str, level: int = 0):
    """Creates a node from a text line."""
    if _check_branch(line):
        line = _get_full_branch_line(line)
        thresh_line, line = _split_thresh_line_from_line(line)
        left_line, right_line = _split_left_and_right_from_line(line)
        
        left_node = create_node(left_line, level + 1)
        right_node = create_node(right_line, level + 1)
        
        feature_id, thresh = _get_thresh(thresh_line)
        branch = Branch(feature_id, thresh, level)
        branch.set_left(left_node)
        branch.set_right(right_node)
        
        return branch
    
    if _check_leaf(line):
        label_id = _get_label(line)
        return Leaf(label_id)
    
    return None


def _check_branch(line: str) -> bool:
    """Checks whether the node is a branch node."""
    return line.find("Branch", 0, 8) >= 0


def _check_leaf(line: str) -> bool:
    """Checks whether the node is a leaf node."""
    return line.find("Label", 0, 6) >= 0


def _get_end_of_bracket(line: str) -> int:
    """Returns the closing bracket position."""
    line = line.strip()
    if line[0] not in LEFT_BRACKET_LIST:
        raise ValueError("Current line are not contained in brackets!")
    
    left_count = 0
    right_count = 0
    
    for i in range(len(line)):
        if line[i] in LEFT_BRACKET_LIST:
            left_count += 1
        if line[i] in RIGHT_BRACKET_LIST:
            right_count += 1
        if left_count == right_count:
            return i
    
    raise ValueError("Current line does not contained a complete bracket!")


def _get_thresh(line: str) -> Tuple[str, str]:
    """Returns the feature ID and threshold."""
    line = _remove_indicator(line)
    thresh_match = re.match(FEATURE_PATTERN, line)
    feature_id = thresh_match.group(1)
    thresh = thresh_match.group(2)
    return feature_id, thresh


def _get_label(line: str) -> str:
    """Returns the label ID."""
    label_match = re.match(LEAF_PATTERN, line)
    return label_match.group(1)


def _remove_indicator(line: str) -> str:
    """Removes the indicator."""
    for indicator in INDICATOR_LIST:
        len_to_check = len(indicator) + 2
        if line.find(indicator, 0, len_to_check) >= 0:
            end_index = line.find(indicator, 0, len_to_check) + len(indicator)
            return line[end_index:].strip()
    raise ValueError("Current line does not start with an indicator!")


def _get_full_branch_line(line: str) -> str:
    """Returns the complete branch information."""
    line = _remove_indicator(line)
    end_index = _get_end_of_bracket(line)
    return line[:end_index + 1].strip()


def _split_thresh_line_from_line(line: str) -> Tuple[str, str]:
    """Splits the threshold line."""
    index_bin = line.find(",") + 1
    index_bin += (line[index_bin:]).find(",")
    return line[:index_bin].strip(), line[index_bin + 1:].strip()


def _split_left_and_right_from_line(line: str) -> Tuple[str, str]:
    """Splits the left and right subtrees."""
    if line.find("left", 0, 6) < 0:
        raise ValueError("Current line does not start with left node definition!")
    
    right_index = line.find("left") + 4
    current_line = line[line.find("left") + 4:]
    
    left_count = 1
    right_count = 0
    
    while current_line:
        next_right = current_line.find("right")
        next_left = current_line.find("left")
        
        if (next_right >= 0) and ((next_left < 0) or (next_left > next_right)):
            right_index += (next_right + 5)
            current_line = current_line[next_right + 5:]
            right_count += 1
        
        if ((next_left < next_right) and (next_left >= 0)):
            right_index += (next_left + 4)
            current_line = current_line[next_left + 4:]
            left_count += 1
        
        if left_count == right_count:
            right_index -= 5
            left_line = _remove_indicator(line[:right_index].strip()).strip()
            right_line = _remove_indicator(line[right_index:].strip()).strip()
            return left_line, right_line
        
        if next_right < 0:
            raise ValueError("Current line remains no information about right node definition!")
    
    raise ValueError("Current line does not contained a right node definition!")


def read_forest_from_file(file_path: str) -> RandomForest:
    """Reads a random forest from a file."""
    f = open(file_path, "r")
    name = (file_path.split("/")[-1]).split(".")[0]
    rf = RandomForest(name)
    
    # Read labels
    line = f.readline()
    rf.set_labels([label[1:-1] for label in line[1:-2].split(",")])
    
    # Read decision trees line by line
    line = f.readline()
    while line:
        rf.add_decision_tree_from_line(line)
        line = f.readline()
    
    f.close()
    return rf


def get_copse_model_from_serial_file(file_path: str, expected_n: int = -1) -> COPSEModel:
    """Loads a COPSE model from a serialized file."""
    rf = read_forest_from_file(file_path)
    rf.initialize()
    rf.get_forest_thresh_and_k(expected_n)
    rf.get_forest_lvls_and_masks()
    rf.generate_copse_model()
    
    rf.copse_model.name = file_path.split("/")[-1][:-4]

    # Set precision from the data, not the filename. Per-feature quantization maps
    # each feature's largest split onto [0, 2^p - 1], so an 8-bit model tops out at
    # 255 and a 16-bit model has thresholds well beyond it. Inferring p from the max
    # threshold is robust to model names (e.g. iris/mnist) and fixes two latent
    # mislabels of the old name-based check: `p16` (matched only `prec16`) and the
    # `Iris` GBDT (16-bit thresholds but no name hit) were both being run at p=8.
    max_thresh = max(rf.copse_model.thresh) if rf.copse_model.thresh else 0
    rf.copse_model.p = 16 if max_thresh > 255 else 8

    return rf.copse_model


def unsecure_copse_model(input_copse_model: COPSEModel) -> COPSEModel:
    """Generates optimized COPSE model (removes zero columns)."""
    import numpy as np
    
    unsecure_thresh = [thresh for thresh in input_copse_model.thresh if thresh != 0]
    zero_column_list = [input_copse_model.thresh[id] == 0 for id in range(len(input_copse_model.thresh))]
    
    reshuf_matrix = np.array(input_copse_model.reshuf)
    unsecure_reshuf = np.delete(reshuf_matrix, zero_column_list, 1).tolist()
    
    unsecure_copse = input_copse_model
    unsecure_copse.thresh = unsecure_thresh
    unsecure_copse.reshuf = unsecure_reshuf
    unsecure_copse.q = len(unsecure_thresh) 
    unsecure_copse.k = 1
    
    return unsecure_copse


def get_unsecure_copse_model_from_serial_model(model_path: str, expected_n: int = -1, real_world : bool = False) -> COPSEModel:
    """Loads optimized COPSE model from a serialized model file."""
    
    current_copse_model = get_copse_model_from_serial_file(model_path, expected_n)
    current_unsecure_copse_model = unsecure_copse_model(current_copse_model)
    return current_unsecure_copse_model
