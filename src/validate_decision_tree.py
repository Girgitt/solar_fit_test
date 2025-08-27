def _validate_tree_structure(node: dict) -> None:
    if "value" in node:
        return  # it's a leaf, that's fine

    required_keys = ["feature", "threshold", "left", "right"]
    for key in required_keys:
        if key not in node:
            raise ValueError(f"Missing key '{key}' in internal node: {node}")

    # Recursively validate children
    _validate_tree_structure(node["left"])
    _validate_tree_structure(node["right"])

def _traverse_tree(node: dict, x_val: float) -> float:
    if "value" in node:
        return node["value"]

    threshold = node["threshold"]
    feature = node["feature"]  # always 0 in this case

    if x_val <= threshold:
        return _traverse_tree(node["left"], x_val)
    else:
        return _traverse_tree(node["right"], x_val)