from types import SimpleNamespace

import numpy as np
import pytest

from shading_aware_pv.models import Mesh
from shading_aware_pv.optimization import (
    PlacementCandidate, PlacementOptimization, PlacementSolution,
    optimize_relocation, retain_installed_layout,
)


def mixed_search():
    def rectangle(x):
        return np.array([[x, 1., 0.], [x + 1, 1., 0.], [x + 1, 2., 0.], [x, 2., 0.]])
    def cell(x):
        roof = rectangle(x)
        return SimpleNamespace(facet_id='facet_00', orientation='portrait',
                               roof_corners_xyz=roof, receiver_corners_xyz=roof + [0, 0, .2])
    layout = SimpleNamespace(modules=SimpleNamespace(cells=(cell(1), cell(4))), roof_sample_indices=np.array([0, 2]))
    candidates = tuple(PlacementCandidate(i, 'facet_00', 'portrait', rectangle(x),
                                         rectangle(x) + [0, 0, .2], np.array([sample]), score)
                       for i, (x, sample, score) in enumerate([(1.2, 1, 40.), (4.2, 3, 90.)]))
    maximum = PlacementSolution(2, np.array([0, 1]), np.array([1, 3]))
    original = PlacementOptimization(.05, .05, 0., .1, 1., 1., candidates,
                                     np.empty((0, 2), dtype=int), 2, maximum, maximum)
    mesh = Mesh(np.array([[0., 0., 0.], [10., 0., 0.], [10., 4., 0.], [0., 4., 0.]]),
                np.array([[0, 1, 2], [0, 2, 3]]))
    scene = SimpleNamespace(partition=SimpleNamespace(main_roof=mesh))
    resource = SimpleNamespace(
        roof_samples=SimpleNamespace(points=np.array([[1.05, 1.5, 0.], [2.15, 1.5, 0.], [4.05, 1.5, 0.], [5.15, 1.5, 0.]])),
        receivers=SimpleNamespace(areas=np.ones(4)),
        results={'full': SimpleNamespace(irradiation=np.array([100., 40., 10., 90.]))},
    )
    proposal, incumbent = optimize_relocation(scene, original, resource, layout)
    return original, proposal, incumbent, layout


def test_search_can_retain_one_panel_and_move_another():
    original, proposal, incumbent, layout = mixed_search()
    # Generated-only B+D scores130; inferred-only A+C scores110.
    # Only a true mixed search can find inferred A + generated D, scoring190.
    np.testing.assert_array_equal(proposal.relocated_layout.candidate_ids, [1, 2])
    np.testing.assert_array_equal(proposal.relocated_layout.roof_sample_indices, [0, 3])
    np.testing.assert_array_equal(incumbent.relocated_layout.candidate_ids, [2, 3])
    np.testing.assert_array_equal(incumbent.relocated_layout.roof_sample_indices, layout.roof_sample_indices)
    assert {tuple(pair) for pair in proposal.conflict_pairs} == {(0, 2), (1, 3)}
    assert proposal.maximum_layout is original.maximum_layout
    assert len(original.candidates) == 2
    assert len(proposal.candidates) == 4
    np.testing.assert_array_equal(proposal.candidates[2].receiver_corners_xyz, layout.modules.cells[0].receiver_corners_xyz)


@pytest.mark.parametrize('proposed_kwh,retained', [(199., True), (200., True), (201., False)])
def test_mixed_proposal_still_must_improve_modeled_ac(proposed_kwh, retained):
    original, proposal, incumbent, layout = mixed_search()
    selected = retain_installed_layout(proposal, incumbent, 200., proposed_kwh)
    assert (selected is incumbent) == retained
    assert selected.maximum_layout is original.maximum_layout
    if retained:
        for index, cell in zip(selected.relocated_layout.candidate_ids, layout.modules.cells, strict=True):
            np.testing.assert_array_equal(selected.candidates[index].receiver_corners_xyz, cell.receiver_corners_xyz)
