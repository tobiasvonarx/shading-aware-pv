"""An inferred installed rectangle needs full support, not partial mask overlap."""
import numpy as np
import rasterio
import shapely
from rasterio.transform import from_origin
from shapely.geometry import Polygon, box

from shading_aware_pv.models import Mesh
from shading_aware_pv.modules import (
    INSTALLED_SUPPORT_TOLERANCE_M, _Observation, _fit_phase, _roof_frame,
    infer_module_layout, roof_facets,
)


def mesh_from_polygon(polygon, z=0):
    triangles = [np.column_stack((np.asarray(t.exterior.coords)[:3], np.full(3, z)))
                 for t in shapely.constrained_delaunay_triangles(polygon).geoms]
    return Mesh(np.asarray(triangles).reshape(-1, 3), np.arange(len(triangles) * 3).reshape(-1, 3))


def test_installed_cells_fit_exposed_support_and_clear_full_mesh(tmp_path):
    support = box(0, 0, 8, 8).difference(box(3, 3, 5, 5))
    roof = mesh_from_polygon(support)
    blocker_polygon = box(.5, .5, 2.5, 2.5)
    blocker = mesh_from_polygon(blocker_polygon, 1)
    full = Mesh(np.vstack((roof.vertices, blocker.vertices)),
                np.vstack((roof.faces, blocker.faces + len(roof.vertices))))
    image = tmp_path / 'roof.tif'
    with rasterio.open(image, 'w', driver='GTiff', width=80, height=80, count=3,
                       dtype='uint8', transform=from_origin(0, 8, .1, .1)) as destination:
        destination.write(np.full((3, 80, 80), 128, dtype=np.uint8))
    layout = infer_module_layout(roof, box(0, 0, 8, 8), image, .2, full)
    assert len(layout.cells) > 1
    assert sum(array.module_count for array in layout.arrays) == len(layout.cells)
    selected = next(candidate for candidate in layout.candidates if candidate.selected)
    assert selected.module_count == len(layout.cells)
    assert abs(selected.mask_recall - layout.mask_recall) < .03
    for cell in layout.cells:
        polygon = Polygon(cell.roof_corners_xyz[:, :2])
        assert support.buffer(INSTALLED_SUPPORT_TOLERANCE_M).covers(polygon)
        assert polygon.intersection(blocker_polygon).area <= 1e-10


def test_split_shared_edges_do_not_split_support_facets():
    # The long diagonal of the first triangle meets two half-length edges.
    vertices = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [1, 1, 0], [2, 2, 0]], dtype=float)
    mesh = Mesh(vertices, np.array([[0, 1, 2], [1, 4, 3], [3, 4, 2]]))
    facets = roof_facets(mesh)
    assert len(facets) == 1
    assert abs(facets[0].polygon_xy.area - 4.0) < 1e-8


def test_submillimetre_clipping_seam_does_not_split_support():
    vertices = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0],
                         [2, 0, .0005], [2, 2, .0005], [0, 2, .0005]], dtype=float)
    assert len(roof_facets(Mesh(vertices, np.array([[0, 1, 2], [3, 4, 5]])))) == 1


def test_projected_shared_edges_do_not_join_different_heights():
    vertices = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0],
                         [2, 0, 1], [2, 2, 1], [0, 2, 1]], dtype=float)
    assert len(roof_facets(Mesh(vertices, np.array([[0, 1, 2], [3, 4, 5]])))) == 2


def boundary_observation(blocked=Polygon()):
    geometry = box(-.0005, 0, 1.9995, 1.7)
    u, v = np.arange(.0495, 2, .1), np.arange(.05, 1.7, .1)
    mask = np.ones((len(v), len(u)), dtype=bool)
    return _Observation('array', None, geometry, u, v, mask,
                        np.zeros(mask.shape), np.zeros(mask.shape), .1,
                        box(0, 0, 2, 1.7).buffer(INSTALLED_SUPPORT_TOLERANCE_M), blocked)


def test_submillimetre_support_residual_does_not_delete_a_row():
    fit = _fit_phase(boundary_observation(), 1, 1.7, 'portrait', 0, 0)
    assert len(fit.cells) == 2


def test_several_millimetres_of_support_residual_are_tolerated():
    fit = _fit_phase(boundary_observation(), 1, 1.7, 'portrait', .005, 0)
    assert len(fit.cells) == 2


def test_real_overhang_is_removed_before_scoring():
    fit = _fit_phase(boundary_observation(), 1, 1.7, 'portrait', .02, 0)
    assert len(fit.cells) == 1
    assert fit.predicted_m2 == 1.7
    assert fit.mask_recall < .6


def test_support_tolerance_does_not_erase_a_thin_obstruction():
    fit = _fit_phase(boundary_observation(box(.4, 0, .4005, 1.7)), 1, 1.7, 'portrait', 0, 0)
    assert len(fit.cells) == 1


def test_roof_coordinates_are_stable_at_swiss_map_origin():
    from shapely.affinity import translate
    polygon = box(0, 0, 10, 8)
    a, b = .25, -.17
    near = _roof_frame('near', polygon, (a, b, 0))
    x, y, z = 2615000., 1200000., 800.
    far = _roof_frame('far', translate(polygon, x, y), (a, b, z-a*x-b*y))
    local = near.xy_to_uv(polygon)
    remote = far.xy_to_uv(translate(polygon, x, y))
    assert local.hausdorff_distance(remote) < 1e-8
