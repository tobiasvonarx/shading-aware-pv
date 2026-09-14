"""Independent placement-surface invariance and opt-in real/source regressions.

EXPOSED_REAL_RESULT: directory containing mesh.ply, roof_details.geojson,
scaffold.geojson. EXPOSED_ORIGINAL_ROOT: authorized original/reference checkout.
No weather, model downloads, live jobs, or input writes are performed.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import shapely
from shapely.geometry import Polygon, box, shape
from shapely.ops import unary_union


def api(package='shading_aware_pv'):
    return SimpleNamespace(**{name: importlib.import_module(f'{package}.{name}')
                              for name in ['geometry', 'models', 'modules', 'optimization', 'panels']})


def synthetic_mesh(module, *, joined=False, reverse=False):
    # The variant adds a roof triangulation vertex coincident with a dormer
    # bottom corner; it changes connectivity while preserving every surface.
    vertices = np.array([
        [0, 0, 0], [8, 0, 0], [8, 8, 0], [0, 8, 0],
        [0, 0, -3], [8, 0, -3], [8, 8, -3], [0, 8, -3],
        [2, 2, 0], [6, 2, 0], [6, 5, 0], [2, 5, 0],
        [2, 2, 2], [6, 2, 2], [6, 5, 2], [2, 5, 2],
    ], dtype=float)
    faces = [[0, 1, 8], [1, 2, 8], [2, 3, 8], [3, 0, 8]] if joined else [[0, 1, 2], [0, 2, 3]]
    faces += [[4, 6, 5], [4, 7, 6]]
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        faces += [[a, a + 4, b + 4], [a, b + 4, b]]
    labels = ['scaffold'] * len(faces)
    dormer = [[12, 13, 14], [12, 14, 15]]
    for a, b in [(8, 9), (9, 10), (10, 11), (11, 8)]:
        dormer += [[a, b, b + 4], [a, b + 4, a + 4]]
    faces += dormer
    labels += ['dormer'] * len(dormer)
    faces, labels = np.asarray(faces, dtype=np.int32), np.asarray(labels)
    if reverse:
        faces, labels = faces[::-1, ::-1], labels[::-1]
    return module.models.Mesh(vertices, faces), labels


def scene_and_candidates(module, mesh=None, labels=None, *, result=None, phase_step=.4):
    if result is None:
        partition = module.geometry.partition_mesh(mesh, face_labels=labels)
        samples = module.geometry.sample_roof(partition.main_roof, .2)
        scene = module.models.RoofScene(Path('synthetic.ply'), partition, samples,
                                       np.zeros(len(samples.points), dtype=bool))
    else:
        scene = module.geometry.load_roof_scene(
            result / 'mesh.ply', roof_details_path=result / 'roof_details.geojson',
            scaffold_path=result / 'scaffold.geojson', sample_spacing=.2,
            detail_clearance=.15)
    receivers = module.panels.roof_parallel_receivers(scene.samples, .2)
    count = len(scene.samples.points)
    resource = module.models.RoofResource(
        scene.samples, receivers, np.full(count, 'free_roof'),
        {'full': module.models.IrradianceResult(np.full(count, 1000.), np.zeros(count))})
    exclusions = scene.partition.non_pv_mesh
    candidates, _ = module.optimization.generate_candidates(
        scene, resource, exclusions, module_width_m=1., module_height_m=1.7,
        roof_setback_m=.2, obstruction_setback_m=.2, module_gap_m=0.,
        phase_step_m=phase_step, mounting_clearance_m=.2)
    return scene, candidates


def signature(candidates):
    return sorted(tuple(sorted(tuple(np.round(point, 7)) for point in candidate.receiver_corners_xyz))
                  for candidate in candidates)


def layers(mesh):
    result = {}
    for triangle in mesh.triangles:
        assert np.ptp(triangle[:, 2]) < 1e-9
        result.setdefault(round(float(triangle[0, 2]), 7), []).append(Polygon(triangle[:, :2]))
    return {height: unary_union(polygons) for height, polygons in result.items()}


def test_placement_independent_of_connected_dormer_and_face_order():
    module = api()
    reference = None
    component_counts = set()
    for joined, reverse in [(False, False), (True, False), (False, True), (True, True)]:
        mesh, labels = synthetic_mesh(module, joined=joined, reverse=reverse)
        scene, candidates = scene_and_candidates(module, mesh, labels)
        component_counts.add(scene.partition.component_count)
        surface = layers(scene.partition.main_roof)
        assert set(surface) == {0.}
        assert surface[0.].symmetric_difference(box(0, 0, 8, 8).difference(box(2, 2, 6, 5))).area < 1e-8
        assert candidates
        assert all(np.allclose(c.receiver_corners_xyz[:, 2], .2) for c in candidates)
        for candidate in candidates:
            if np.allclose(candidate.receiver_corners_xyz[:, 2], .2):
                assert Polygon(candidate.receiver_corners_xyz[:, :2]).intersection(box(2, 2, 6, 5)).area < 1e-8
        if reference is None:
            reference = signature(candidates)
        else:
            assert signature(candidates) == reference
    assert component_counts == {1, 2}, 'Fixture must actually change connectivity'


def original_api(monkeypatch):
    root = os.environ.get('EXPOSED_ORIGINAL_ROOT')
    if not root:
        pytest.skip('Set EXPOSED_ORIGINAL_ROOT for original/package parity')
    root = Path(root)
    monkeypatch.setattr(sys, 'dont_write_bytecode', True)
    for name, path in [('src', root / 'src'), ('src.emboss', root / 'src/emboss'),
                       ('_original_surface_solar', root / 'src/solar_poc')]:
        module = ModuleType(name)
        module.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, module)
    return api('_original_surface_solar')


def assert_numeric_equal(old, new):
    old_scene, old_candidates = old
    new_scene, new_candidates = new
    np.testing.assert_array_equal(old_scene.partition.main_roof.vertices, new_scene.partition.main_roof.vertices)
    np.testing.assert_array_equal(old_scene.partition.main_roof.faces, new_scene.partition.main_roof.faces)
    for field in ['points', 'normals', 'areas', 'face_ids']:
        np.testing.assert_array_equal(getattr(old_scene.samples, field), getattr(new_scene.samples, field))
    assert len(old_candidates) == len(new_candidates)
    for left, right in zip(old_candidates, new_candidates, strict=True):
        assert left.facet_id == right.facet_id
        assert left.orientation == right.orientation
        np.testing.assert_array_equal(left.receiver_corners_xyz, right.receiver_corners_xyz)
        np.testing.assert_array_equal(left.roof_corners_xyz, right.roof_corners_xyz)
        np.testing.assert_array_equal(left.roof_sample_indices, right.roof_sample_indices)
        assert left.annual_irradiation_kwh_m2 == right.annual_irradiation_kwh_m2


def test_original_packaged_synthetic_candidates_exactly_equal(monkeypatch):
    original, packaged = original_api(monkeypatch), api()
    assert_numeric_equal(scene_and_candidates(original, *synthetic_mesh(original, joined=True)),
                         scene_and_candidates(packaged, *synthetic_mesh(packaged, joined=True)))


@pytest.fixture(scope='module')
def real_pipeline():
    value = os.environ.get('EXPOSED_REAL_RESULT')
    if not value:
        pytest.skip('Set EXPOSED_REAL_RESULT for the saved building regression')
    path = Path(value)
    return path, scene_and_candidates(api(), result=path, phase_step=.1)


def test_real_candidates_have_no_under_or_on_dormer_placement(real_pipeline):
    path, (scene, candidates) = real_pipeline
    features = json.loads((path / 'roof_details.geojson').read_text())['features']
    top_count = 0
    for feature in features:
        properties = feature['properties']
        if properties.get('class_label') != 'dormer':
            continue
        footprint, plane = shape(feature['geometry']), np.asarray(properties['top_plane'])
        for candidate in candidates:
            xyz = candidate.receiver_corners_xyz
            overlap = Polygon(xyz[:, :2]).intersection(footprint)
            if overlap.area < 1e-6:
                continue
            origin = xyz[0]
            slope = np.linalg.solve(xyz[1:3, :2] - origin[:2], xyz[1:3, 2] - origin[2])
            xy = shapely.get_coordinates(overlap)
            delta = (xy - origin[:2]) @ slope + origin[2] - xy @ plane[:2] - plane[2]
            assert delta.min() >= -1e-6, (properties['solid_id'], candidate.candidate_id, delta.min())
            if properties['solid_id'] == 'solid_000' and np.allclose(delta, .2, atol=1e-6):
                top_count += 1
    assert top_count == 0, 'Dormer roofs must not support PV'
    source = api().geometry.read_ascii_ply(path / 'mesh.ply')
    np.testing.assert_array_equal(scene.partition.full_mesh.vertices, source.vertices)
    np.testing.assert_array_equal(scene.partition.full_mesh.faces, source.faces)


def test_original_packaged_real_candidates_exactly_equal(real_pipeline, monkeypatch):
    path, packaged = real_pipeline
    original = scene_and_candidates(original_api(monkeypatch), result=path, phase_step=.1)
    assert_numeric_equal(original, packaged)


@pytest.mark.parametrize('reverse', [False, True])
def test_flush_window_excluded_and_pv_removed_without_losing_roof(reverse):
    module = api()
    vertices = np.array([[0, 0, 0], [8, 0, 0], [8, 8, 0], [0, 8, 0],
                         [2, 2, 0], [4, 2, 0], [4, 4, 0], [2, 4, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
                      [4, 5, 6], [4, 6, 7]], dtype=np.int32)
    labels = np.array(['scaffold', 'scaffold', 'pvmodule', 'pvmodule', 'window', 'window'])
    if reverse:
        faces, labels = faces[::-1, ::-1], labels[::-1]
    partition = module.geometry.partition_mesh(module.models.Mesh(vertices, faces), face_labels=labels)
    footprint = module.geometry.mesh_footprint(partition.main_roof)
    assert footprint.symmetric_difference(box(0, 0, 8, 8).difference(box(2, 2, 4, 4))).area < 1e-8
    pv_only = labels != 'window'
    partition = module.geometry.partition_mesh(module.models.Mesh(vertices, faces[pv_only]), face_labels=labels[pv_only])
    assert module.geometry.mesh_footprint(partition.main_roof).symmetric_difference(box(0, 0, 8, 8)).area < 1e-8


def test_source_millimeter_rounding_does_not_create_internal_roof_setbacks():
    """A nearly planar survey roof remains one placement facet, as before clipping."""
    module = api()
    vertices = np.array([[0., 0., 0.], [8., 0., .001],
                         [8., 8., 2.4], [0., 8., 2.4]])
    mesh = module.models.Mesh(vertices, np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32))
    facets = module.modules.roof_facets(mesh)
    assert len(facets) == 1, 'Millimeter survey rounding must not become an internal setback boundary'
    assert facets[0].polygon_xy.area == pytest.approx(64.)
    samples = module.geometry.sample_roof(mesh, .2)
    resource = module.models.RoofResource(
        samples, module.panels.roof_parallel_receivers(samples, .2),
        np.full(len(samples.points), 'free_roof'), {})
    selected, _ = module.optimization._facet_sample_coordinates(facets[0], resource)
    assert len(selected) == len(samples.points), 'Survey rounding must not discard valid samples'
