"""
TorchInfo JSON export utilities.

This module provides helpers for converting ``torchinfo`` model summaries
into JSON-serializable structures for downstream analysis and reporting.

Classes
-------
TorchinfoJSONExporter
    Helper for exporting flat and hierarchical layer views from
    ``torchinfo.ModelStatistics`` instances.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch
from torch import nn
from torchinfo import ModelStatistics


def _to_jsonable(obj: Any) -> Any:
    """
    Convert TorchInfo / PyTorch objects to JSON-serializable values.

    Parameters
    ----------
    obj : Any
        Object to convert.

    Returns
    -------
    Any
        JSON-compatible representation of the input object.
    """

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, torch.Size):
        return list(obj)

    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]

    if isinstance(obj, Mapping):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return obj.item()
        return {"tensor_shape": list(obj.shape)}

    return repr(obj)


class TorchinfoJSONExporter:
    """
    Export TorchInfo summaries to JSON-style dictionaries.

    This class takes a :class:`torchinfo.ModelStatistics` instance together
    with a PyTorch ``nn.Module`` and builds both a flat per-layer view and
    a hierarchical representation of the model suitable for JSON export.

    Attributes
    ----------
    m_stats : ModelStatistics or None
        Model statistics object to export.
    m_named_modules : Mapping[int, str] or None
        Mapping from ``id(module)`` to fully-qualified module name.
    """

    def __init__(self) -> None:
        """Initialize an empty exporter instance."""

        self.m_stats: Optional[ModelStatistics] = None
        self.m_named_modules: Optional[Mapping[int, str]] = None

    @classmethod
    def from_model(
        cls,
        stats: ModelStatistics,
        model: nn.Module,
    ) -> "TorchinfoJSONExporter":
        """
        Create an exporter from model statistics and a PyTorch module.

        Parameters
        ----------
        stats : ModelStatistics
            TorchInfo statistics for the model.
        model : nn.Module
            The PyTorch module used to derive module names.

        Returns
        -------
        TorchinfoJSONExporter
            An initialized exporter instance.
        """

        mapping: Dict[int, str] = {id(mod): name for name, mod in model.named_modules()}
        instance = cls()
        instance.m_stats = stats
        instance.m_named_modules = mapping
        return instance

    def _ensure_initialized(self) -> None:
        """Validate that the exporter has been initialized."""

        if self.m_stats is None or self.m_named_modules is None:
            raise RuntimeError("TorchinfoJSONExporter is not initialized. Use from_model(...) first.")

    def _layer_base_dict(self, layer: Any, index: int | None) -> dict[str, Any]:
        """
        Extract flat JSON-serializable fields from a TorchInfo LayerInfo.

        Parameters
        ----------
        layer : Any
            TorchInfo layer object.
        index : int or None
            Optional index of this layer in the summary list.

        Returns
        -------
        dict
            Dictionary containing flat layer attributes.
        """

        self._ensure_initialized()
        module = getattr(layer, "module", None)
        module_name = self.m_named_modules.get(id(module)) if module is not None else None  # type: ignore[union-attr]

        return {
            "index": index,
            "layer_id": _to_jsonable(getattr(layer, "layer_id", None)),
            "var_name": getattr(layer, "var_name", None),
            "class_name": getattr(layer, "class_name", None),
            "module_name": module_name,
            "depth": int(getattr(layer, "depth", 0) or 0),
            "depth_index": int(getattr(layer, "depth_index", 0) or 0),
            "is_leaf_layer": bool(getattr(layer, "is_leaf_layer", False)),
            "is_recursive": bool(getattr(layer, "is_recursive", False)),
            "contains_lazy_param": bool(getattr(layer, "contains_lazy_param", False)),
            "executed": bool(getattr(layer, "executed", False)),
            "input_size": _to_jsonable(getattr(layer, "input_size", None)),
            "output_size": _to_jsonable(getattr(layer, "output_size", None)),
            "kernel_size": _to_jsonable(getattr(layer, "kernel_size", None)),
            "num_params": int(getattr(layer, "num_params", 0) or 0),
            "trainable_params": int(getattr(layer, "trainable_params", 0) or 0),
            "param_bytes": int(getattr(layer, "param_bytes", 0) or 0),
            "output_bytes": int(getattr(layer, "output_bytes", 0) or 0),
            "macs": int(getattr(layer, "macs", 0) or 0),
        }

    def layers_flat(self) -> list[dict[str, Any]]:
        """
        Build a flat list of per-layer dictionaries.

        Returns
        -------
        list of dict
            Flat representation of all layers in the summary.
        """

        self._ensure_initialized()
        assert self.m_stats is not None

        flat: list[dict[str, Any]] = []
        for idx, layer in enumerate(self.m_stats.summary_list):
            flat.append(self._layer_base_dict(layer, index=idx))
        return flat

    def hierarchy(self) -> list[dict[str, Any]]:
        """
        Build a hierarchical representation of the TorchInfo summary.

        Returns
        -------
        list of dict
            Root nodes of the hierarchical model representation, each
            containing nested ``children`` entries.
        """

        self._ensure_initialized()
        assert self.m_stats is not None

        index_by_layer: Dict[int, int] = {
            id(layer): idx for idx, layer in enumerate(self.m_stats.summary_list)
        }

        def build_node(layer: Any, seen: set[int]) -> dict[str, Any]:
            """
            Recursively build a node dictionary for a given layer.

            Parameters
            ----------
            layer : Any
                TorchInfo layer object.
            seen : set of int
                Set of visited layer IDs to guard against recursion.

            Returns
            -------
            dict
                Node dictionary with optional ``children``.
            """

            lid = id(layer)
            idx = index_by_layer.get(lid)

            data = self._layer_base_dict(layer, index=idx)

            if lid in seen or bool(getattr(layer, "is_recursive", False)):
                data["recursive_ref"] = True
                data["children"] = []
                return data

            seen.add(lid)
            children_payload: list[dict[str, Any]] = []

            for child in getattr(layer, "children", []) or []:
                children_payload.append(build_node(child, seen))

            data["recursive_ref"] = False
            data["children"] = children_payload
            return data

        roots: list[dict[str, Any]] = []
        for layer in self.m_stats.summary_list:
            depth = int(getattr(layer, "depth", 0) or 0)
            if depth == 0:
                roots.append(build_node(layer, set()))

        return roots
