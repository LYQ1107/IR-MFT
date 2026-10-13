from cfp.paired_explanation_diagnosis import paired_cases


def test_physical_comparison_requires_common_birth_and_two_known_labels():
    a, b, c, missing, newborn = [(1, 1, i) for i in range(5)]
    original = {a: dict(known=True, correct=False), b: dict(known=True, correct=True),
                c: dict(known=False, correct=False), missing: dict(known=True, correct=True)}
    changed = {a: dict(known=True, correct=True), b: dict(known=True, correct=False),
               c: dict(known=True, correct=True), newborn: dict(known=True, correct=True)}
    cases, disappeared, new = paired_cases(original, changed)
    assert cases == {a: 'N01', b: 'N10', c: 'unknown'}
    assert disappeared == 1 and new == 1
