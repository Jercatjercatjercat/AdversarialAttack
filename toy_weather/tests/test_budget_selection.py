"""A demonstration budget must be selected without consulting test errors."""
import pytest
from select_looser_budget import choose_budget


def test_smallest_budget_must_meet_both_objective_gaps():
    rows=[dict(split='validation',budget=2.,fixed_pgd_minus_fgsm_K=.02,rollout_pgd_minus_fgsm_K=.5),
          dict(split='validation',budget=.5,fixed_pgd_minus_fgsm_K=.001,rollout_pgd_minus_fgsm_K=.1),
          dict(split='validation',budget=1.,fixed_pgd_minus_fgsm_K=.006,rollout_pgd_minus_fgsm_K=.02)]
    selected=choose_budget(rows)
    assert selected['budget']==1. and selected['criterion_met']
    # A large test score must never influence a budget selection.
    with pytest.raises(ValueError,match='validation-only'):
        choose_budget(rows+[dict(rows[0],split='test')])


def test_failed_visibility_criterion_is_reported_instead_of_claimed():
    rows=[dict(split='validation',budget=b,fixed_pgd_minus_fgsm_K=.001,rollout_pgd_minus_fgsm_K=.002)
          for b in (.25,.5)]
    selected=choose_budget(rows)
    assert selected['budget']==.5
    assert not selected['criterion_met']
