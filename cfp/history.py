"""Bounded views of accepted observations; no labels or future observations."""


def anchor_recent_history(history, anchor, limit=8):
    """Keep the first accepted observation plus the latest observations.

    The predictor receives at most ``limit`` tokens. The underlying tracker
    retains its original recent buffer for exact disabled-mode compatibility.
    """
    if limit < 2:
        raise ValueError('Anchor/recent history requires at least two slots')
    history = list(history)[-limit:]
    if any(row[0] <= 0 for row in history) or any(
        history[i][0] >= history[i + 1][0] for i in range(len(history) - 1)
    ):
        raise ValueError('Accepted history timestamps must increase strictly')
    if anchor is None:
        return history
    if anchor[0] <= 0 or (history and anchor[0] > history[0][0]):
        raise ValueError('The initial anchor cannot follow an older observation')
    if not history:
        return [anchor]
    if anchor[0] == history[0][0]:
        return history
    return [anchor] + history[-(limit - 1):]
