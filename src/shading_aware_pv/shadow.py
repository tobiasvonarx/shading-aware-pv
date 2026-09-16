from __future__ import annotations

import ctypes
from collections.abc import Callable
from dataclasses import dataclass

import moderngl
import numpy as np

from .models import Mesh

VERTEX_SHADER = """
#version 330
in vec3 position;
uniform vec3 center;
uniform vec3 right_axis;
uniform vec3 up_axis;
uniform vec3 sun_axis;
uniform float half_extent;
uniform float q_min;
uniform float q_span;

void main() {
    vec3 relative = position - center;
    float x = dot(relative, right_axis) / half_extent;
    float y = dot(relative, up_axis) / half_extent;
    float normalized_q = (dot(relative, sun_axis) - q_min) / q_span;
    gl_Position = vec4(x, y, 1.0 - 2.0 * normalized_q, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330
void main() {}
"""


def _depth_subimage_reader(ctx: moderngl.Context):
    """Load optional subimage reads using this renderer's EGL context.

    ModernGL 5.12 exposes only whole-texture reads. This extension uses the
    same depth conversion as those reads; framebuffer reads and shader texture
    sampling can differ by one float ULP. Keep the full read on older drivers.
    """
    if ctx.version_code < 450 and "GL_ARB_get_texture_sub_image" not in ctx.extensions:
        return None
    # Resolve through ModernGL's own context loader, without another GL library.
    address = ctx.mglo._context.load("glGetTextureSubImage")
    if not address:
        return None
    return ctypes.CFUNCTYPE(
        None,
        ctypes.c_uint,  # texture
        *([ctypes.c_int] * 7),  # level, offsets, dimensions
        ctypes.c_uint,
        ctypes.c_uint,  # format, type
        ctypes.c_int,
        ctypes.c_void_p,  # buffer size, destination
    )(address)


def sun_vectors(zenith: np.ndarray, azimuth: np.ndarray) -> np.ndarray:
    zenith_rad = np.radians(zenith)
    azimuth_rad = np.radians(azimuth)
    return np.column_stack(
        (
            np.sin(zenith_rad) * np.sin(azimuth_rad),
            np.sin(zenith_rad) * np.cos(azimuth_rad),
            np.cos(zenith_rad),
        )
    )


def _projection_axes(sun: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(helper, sun)) > 0.95:
        helper = np.array([0.0, 1.0, 0.0])
    right = np.cross(helper, sun)
    right /= np.linalg.norm(right)
    up = np.cross(sun, right)
    return right, up


def _box_corners(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [x, y, z]
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        ]
    )


@dataclass(frozen=True)
class RenderStats:
    resolution: int
    effective_pixel_size: float


@dataclass
class DepthRenderer:
    meshes: list[Mesh]
    query_points: np.ndarray
    pixel_size: float = 0.1
    depth_epsilon: float = 0.03
    max_resolution: int = 1024

    def __post_init__(self) -> None:
        geometry = [mesh.vertices for mesh in self.meshes if len(mesh.vertices)]
        all_vertices = np.vstack([*geometry, self.query_points])
        self.origin = self.query_points.mean(axis=0)
        local_vertices = all_vertices - self.origin
        self.lower = local_vertices.min(axis=0) - self.depth_epsilon * 2
        self.upper = local_vertices.max(axis=0) + self.depth_epsilon * 2
        self.center = (self.lower + self.upper) / 2
        self.half_extent = float(
            np.linalg.norm(self.upper - self.lower) / 2 + self.pixel_size
        )
        ideal_resolution = int(np.ceil(2 * self.half_extent / self.pixel_size))
        self.resolution = max(32, min(self.max_resolution, ideal_resolution))
        self.ctx = moderngl.create_standalone_context(backend="egl")
        self.program = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER,
        )
        self.depth_texture = self.ctx.depth_texture((self.resolution, self.resolution))
        self.framebuffer = self.ctx.framebuffer(depth_attachment=self.depth_texture)
        self.vertex_arrays = []
        for mesh in self.meshes:
            if not len(mesh.faces):
                self.vertex_arrays.append(None)
                continue
            triangles = (mesh.triangles - self.origin).astype("f4", copy=False)
            buffer = self.ctx.buffer(triangles.tobytes())
            vao = self.ctx.simple_vertex_array(self.program, buffer, "position")
            self.vertex_arrays.append((vao, buffer))
        self.local_queries = self.query_points - self.origin
        self.corners = _box_corners(self.lower, self.upper)
        self._read_subimage = _depth_subimage_reader(self.ctx)

    def close(self) -> None:
        arrays = [item for item in self.vertex_arrays if item is not None]
        for vao, buffer in arrays:
            vao.release()
            buffer.release()
        self.framebuffer.release()
        self.depth_texture.release()
        self.program.release()
        self.ctx.release()

    def visibility(self, mesh_index: int | None, sun: np.ndarray) -> np.ndarray:
        sun = np.asarray(sun, dtype=np.float64)
        sun /= np.linalg.norm(sun)
        right, up = _projection_axes(sun)
        relative_corners = self.corners - self.center
        q = relative_corners @ sun
        q_min = float(q.min() - self.depth_epsilon)
        q_span = float(q.max() - q.min() + 2 * self.depth_epsilon)

        self.program["center"].value = tuple(self.center.astype(float))
        self.program["right_axis"].value = tuple(right.astype(float))
        self.program["up_axis"].value = tuple(up.astype(float))
        self.program["sun_axis"].value = tuple(sun.astype(float))
        self.program["half_extent"].value = self.half_extent
        self.program["q_min"].value = q_min
        self.program["q_span"].value = q_span
        self.framebuffer.use()
        self.framebuffer.clear(depth=1.0)
        self.ctx.enable_only(moderngl.DEPTH_TEST)
        if mesh_index is not None:
            mesh_array = self.vertex_arrays[mesh_index]
            if mesh_array is not None:
                mesh_array[0].render(mode=moderngl.TRIANGLES)

        relative = self.local_queries - self.center
        ndc_x = relative @ right / self.half_extent
        ndc_y = relative @ up / self.half_extent
        pixel_x = np.clip(
            ((ndc_x + 1) * 0.5 * self.resolution).astype(int), 0, self.resolution - 1
        )
        pixel_y = np.clip(
            ((ndc_y + 1) * 0.5 * self.resolution).astype(int), 0, self.resolution - 1
        )
        expected_depth = 1.0 - ((relative @ sun - q_min) / q_span)
        sampled_depth = self._sample_depth(pixel_x, pixel_y)
        tolerance = self.depth_epsilon / q_span
        return expected_depth <= sampled_depth + tolerance

    def _sample_depth(self, pixel_x: np.ndarray, pixel_y: np.ndarray) -> np.ndarray:
        """Read only queried texels' rectangle, retaining depth conversion."""
        if not len(pixel_x):
            return np.empty(0, dtype=np.float32)
        if self._read_subimage is None:
            depth = np.frombuffer(
                self.depth_texture.read(alignment=1), dtype=np.float32
            )
            return depth.reshape(self.resolution, self.resolution)[pixel_y, pixel_x]
        x_min, y_min = int(pixel_x.min()), int(pixel_y.min())
        width = int(pixel_x.max()) - x_min + 1
        height = int(pixel_y.max()) - y_min + 1
        # This context has no pixel-pack buffer; float32 rows are 4-byte aligned.
        depth = np.empty((height, width), dtype=np.float32)
        self._read_subimage(
            self.depth_texture.glo,
            0,
            x_min,
            y_min,
            0,
            width,
            height,
            1,
            0x1902,
            0x1406,  # GL_DEPTH_COMPONENT, GL_FLOAT
            depth.nbytes,
            depth.ctypes.data,
        )
        return depth[pixel_y - y_min, pixel_x - x_min]


def render_visibility(
    meshes: list[Mesh],
    points: np.ndarray,
    normals: np.ndarray,
    zenith: np.ndarray,
    azimuth: np.ndarray,
    dni: np.ndarray,
    pixel_size: float,
    depth_epsilon: float,
    progress: Callable[[int, int, int], None] | None = None,
) -> list[np.ndarray]:
    sun = sun_vectors(zenith, azimuth)
    # Moving queries off the surface prevents depth equality from turning into acne.
    query_points = points + normals * depth_epsilon
    renderer = DepthRenderer(meshes, query_points, pixel_size, depth_epsilon)
    visibility = [np.ones((len(zenith), len(points)), dtype=bool) for _ in meshes]
    daylight = np.flatnonzero((dni > 0.5) & (zenith < 90.0))
    try:
        for position, hour in enumerate(daylight, start=1):
            for mesh_index in range(len(meshes)):
                visibility[mesh_index][hour] = renderer.visibility(mesh_index, sun[hour])
            if progress is not None and (
                position == 1
                or position % 250 == 0
                or position == len(daylight)
            ):
                progress(position, len(daylight), renderer.resolution)
    finally:
        renderer.close()
    return visibility


def render_context_visibilities(
    meshes: list[Mesh],
    points: np.ndarray,
    normals: np.ndarray,
    zenith: np.ndarray,
    azimuth: np.ndarray,
    dni: np.ndarray,
    pixel_size: float,
    depth_epsilon: float,
    progress: Callable[[int, int, int], None] | None = None,
) -> tuple[list[np.ndarray], RenderStats | None]:
    """Render related DSM variants at one shared resolution."""
    visibility = [
        np.ones((len(zenith), len(points)), dtype=bool) for _ in meshes
    ]
    if not any(len(mesh.faces) for mesh in meshes):
        return visibility, None

    sun = sun_vectors(zenith, azimuth)
    query_points = points + normals * depth_epsilon
    renderer = DepthRenderer(meshes, query_points, pixel_size, depth_epsilon)
    daylight = np.flatnonzero((dni > 0.5) & (zenith < 90.0))
    stats = RenderStats(
        resolution=renderer.resolution,
        effective_pixel_size=2.0 * renderer.half_extent / renderer.resolution,
    )
    try:
        for position, hour in enumerate(daylight, start=1):
            for mesh_index in range(len(meshes)):
                visibility[mesh_index][hour] = renderer.visibility(
                    mesh_index,
                    sun[hour],
                )
            if progress is not None and (
                position == 1
                or position % 250 == 0
                or position == len(daylight)
            ):
                progress(position, len(daylight), renderer.resolution)
    finally:
        renderer.close()
    return visibility, stats
