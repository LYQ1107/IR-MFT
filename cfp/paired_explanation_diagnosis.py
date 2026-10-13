"""Retrospective comparison on the same observed-birth identity reference."""


def paired_cases(original, changed):
    cases = {}
    for birth in original.keys() & changed.keys():
        old, new = original[birth], changed[birth]
        if not old['known'] or not new['known']:
            cases[birth] = 'unknown'
        else:
            cases[birth] = 'N' + str(int(old['correct'])) + str(int(new['correct']))
    return cases, len(original.keys() - changed.keys()), len(changed.keys() - original.keys())
