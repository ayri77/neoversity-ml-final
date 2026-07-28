"""Generic, declarative experiment control-panel support."""

from src.churn_ml.control_panel.registry import ControlPanelRegistry, load_registry

__all__ = ["ControlPanelRegistry", "load_registry"]
