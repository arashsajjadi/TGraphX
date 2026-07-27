"""Compatibility helpers for configuration-default transitions.

TGraphX <= 1.4.2 shipped two low-level modules whose constructors silently
introduced a scientifically meaningful dropout probability of ``0.3``:

* :class:`tgraphx.models.CNNEncoder`  (``dropout_prob=0.3``)
* :class:`tgraphx.layers.aggregator.DeepCNNAggregator` (``dropout_prob=0.3``)

Because the value never appeared in ``repr()``, exported configs, or
checkpoints, models built through :func:`tgraphx.build_model`,
:class:`tgraphx.CNN_GNN_Model`, or :class:`tgraphx.GraphClassifier` trained
with hidden regularization that users never asked for.  Controlled
re-runs on PASTIS-R measured a cost of roughly 0.04-0.06 validation
macro-F1 from the hidden 0.3 alone.

Since v1.5.0 the documented default for these constructors is ``0.0``
(no dropout).  Construction sites that do not pass an explicit value
receive the new default **and** a :class:`DropoutDefaultChangeWarning`,
so the change is loud, not silent.  Legacy behaviour remains available
explicitly via ``dropout_prob=0.3`` or the ``.legacy(...)`` constructors.

Loading old checkpoints is unaffected: dropout modules hold no
parameters, so ``state_dict`` layouts are identical, and evaluation-mode
outputs never depended on the dropout probability.
"""
from __future__ import annotations

import warnings

__all__ = [
    "DropoutDefaultChangeWarning",
    "LEGACY_CNN_DROPOUT_PROB",
    "resolve_dropout_prob",
]

#: Dropout probability silently applied by TGraphX <= 1.4.2 in
#: ``CNNEncoder`` and ``DeepCNNAggregator`` when the caller did not pass
#: ``dropout_prob`` explicitly.
LEGACY_CNN_DROPOUT_PROB: float = 0.3


class DropoutDefaultChangeWarning(UserWarning):
    """Emitted when a module resolves an unspecified ``dropout_prob``.

    TGraphX <= 1.4.2 silently used ``dropout_prob=0.3`` in ``CNNEncoder``
    and ``DeepCNNAggregator``; since v1.5.0 the default is ``0.0``.
    Pass ``dropout_prob`` explicitly (any value, including ``0.0``) to
    silence this warning, or use the ``.legacy(...)`` constructors to
    reproduce the pre-1.5 behaviour intentionally.
    """


def resolve_dropout_prob(
    value: float | None,
    *,
    owner: str,
    new_default: float = 0.0,
    legacy_default: float = LEGACY_CNN_DROPOUT_PROB,
    legacy_hint: str | None = None,
    stacklevel: int = 3,
) -> float:
    """Resolve an optionally-unspecified ``dropout_prob`` constructor arg.

    Args:
        value: The value the caller passed, or ``None`` when unspecified.
        owner: Class name used in the warning message (e.g. ``"CNNEncoder"``).
        new_default: Documented default applied when ``value is None``.
        legacy_default: The pre-1.5 silent default, named in the warning.
        legacy_hint: How to reproduce the legacy behaviour intentionally;
            defaults to ``"<owner>.legacy(...)"``.
        stacklevel: Passed to :func:`warnings.warn` so the warning points
            at the user's construction site.

    Returns:
        The effective dropout probability as a ``float``.

    Raises:
        ValueError: If the resolved probability is outside ``[0, 1)``.
    """
    if value is None:
        if legacy_hint is None:
            legacy_hint = f"dropout_prob={legacy_default} / {owner}.legacy(...)"
        warnings.warn(
            f"{owner}: dropout_prob was not specified. Since TGraphX v1.5.0 "
            f"the default is {new_default} (no hidden dropout); TGraphX <= "
            f"1.4.2 silently used {legacy_default}. Pass dropout_prob "
            f"explicitly (e.g. dropout_prob={new_default}) to silence this "
            f"warning, or use {legacy_hint} to reproduce the old behaviour.",
            DropoutDefaultChangeWarning,
            stacklevel=stacklevel,
        )
        value = new_default
    value = float(value)
    if not 0.0 <= value < 1.0:
        raise ValueError(
            f"{owner}: dropout_prob must be in [0, 1); got {value}."
        )
    return value
