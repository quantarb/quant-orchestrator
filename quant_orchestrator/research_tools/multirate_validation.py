"""Fail-fast checks for the supervised multi-rate training contract."""

import math
from collections import Counter


def validate_supervision(targets, *, equity_symbols, option_symbols, required_tasks):
    """Require observed, finite supervision for each trading task and asset set."""
    counts = {}
    for asset, symbols in (("equity", set(equity_symbols)), ("option", set(option_symbols))):
        if not symbols:
            continue
        observed = Counter()
        for (symbol, _date), values in targets.items():
            if symbol in symbols:
                observed.update(name for name, value in values.items() if math.isfinite(float(value)))
        counts[asset] = {name: observed[name] for name in required_tasks}
        missing = [name for name in required_tasks if not observed[name]]
        if missing:
            raise ValueError(f"Missing supervised training labels for {asset}: {', '.join(missing)}")
    return counts
