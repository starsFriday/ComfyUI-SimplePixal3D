try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:
    import importlib.util
    import os
    import sys

    _node_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nodes.py")
    _spec = importlib.util.spec_from_file_location("simple_pixal3d_nodes", _node_file)
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _module
    _spec.loader.exec_module(_module)
    NODE_CLASS_MAPPINGS = _module.NODE_CLASS_MAPPINGS
    NODE_DISPLAY_NAME_MAPPINGS = _module.NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
