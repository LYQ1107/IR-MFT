"""Offline physical-label audit of causal hypothesis traces; no online imports."""

def explanations(event, labels, frames):
    result = {}
    for branch in event['hypotheses']:
        rows = {}
        for track in branch['current']:
            proposal, birth = track['proposal'], tuple(track['birth'])
            if not 0 <= proposal < len(labels) or frames[proposal] != event['frame']:
                raise ValueError('Current hypothesis proposal timestamp drift')
            initial = birth[2]
            if initial is None:
                rows[birth] = dict(known=False, correct=False)
                continue
            if not 0 <= initial < len(labels) or frames[initial] != birth[1] or birth[1] > event['frame']:
                raise ValueError('Observed birth proposal timestamp drift')
            known = labels[initial] >= 0 and labels[proposal] >= 0
            rows[birth] = dict(known=bool(known), correct=bool(known and labels[proposal] == labels[initial]))
        result[branch['branch']] = rows
    if event['winner'] not in result:
        raise ValueError('Selected explanation absent from trace')
    return result
