"""Import-level tests.

These run first and ensure that every name advertised in the public API is
actually importable after a normal `pip install .`.  A failure here means the
package is broken at the import level and all downstream tests would fail for
the wrong reason.
"""


def test_import_tgraphx_top_level():
    import tgraphx  # noqa: F401


def test_version_attribute_is_non_empty_string():
    import tgraphx
    assert isinstance(tgraphx.__version__, str)
    assert tgraphx.__version__ != ""


# ------------------------------------------------------------------ #
# Top-level re-exports (from tgraphx import ...)                       #
# ------------------------------------------------------------------ #

def test_import_graph_and_graphbatch():
    from tgraphx import Graph, GraphBatch  # noqa: F401


def test_import_dataset_and_dataloader():
    from tgraphx import GraphDataset, GraphDataLoader  # noqa: F401


def test_import_layers_from_top():
    from tgraphx import (  # noqa: F401
        TensorMessagePassingLayer,
        LinearMessagePassing,
        ConvMessagePassing,
        AttentionMessagePassing,
        TensorGATLayer,
        TensorGraphSAGELayer,
        TensorGINLayer,
    )


def test_import_utils_from_top():
    from tgraphx import load_config, get_device  # noqa: F401


# ------------------------------------------------------------------ #
# Sub-package imports (from tgraphx.layers / .models / .core import …) #
# ------------------------------------------------------------------ #

def test_import_layers_subpackage():
    from tgraphx.layers import (  # noqa: F401
        TensorMessagePassingLayer,
        LinearMessagePassing,
        ConvMessagePassing,
        AttentionMessagePassing,
        TensorGATLayer,
        TensorGraphSAGELayer,
        TensorGINLayer,
        DeepCNNAggregator,
        SafeMaxPool2d,
    )


def test_import_core_subpackage():
    from tgraphx.core import (  # noqa: F401
        Graph,
        GraphBatch,
        GraphDataset,
        GraphDataLoader,
        load_config,
        get_device,
    )


def test_import_models_subpackage():
    from tgraphx.models import (  # noqa: F401
        CNNEncoder,
        CNN_GNN_Model,
        GraphClassifier,
        NodeClassifier,
        PreEncoder,
    )
