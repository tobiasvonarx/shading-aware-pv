"""Original bare-roof full-budget placement, without experiment/report output."""

from dataclasses import replace

import numpy as np

from .irradiance import facet_irradiance
from .models import IrradianceResult, RoofResource, RoofSamples
from .optimization import optimize_placements, select_milp


def annual_fields(receivers, weather, masks):
    """Integrate weather in bounded receiver chunks, sharing transposition."""
    normals, inverse = np.unique(receivers.normals, axis=0, return_inverse=True)
    representatives = RoofSamples(
        np.zeros_like(normals), normals, np.ones(len(normals)), np.arange(len(normals))
    )
    components = facet_irradiance(representatives, weather)
    fields = {stage: np.empty(len(receivers.points)) for stage in masks}
    for start in range(0, len(receivers.points), 256):
        chunk = slice(start, start + 256)
        face = inverse[chunk]
        diffuse = components["sky"][:, face] + components["ground"][:, face]
        for stage, visible in masks.items():
            fields[stage][chunk] = (
                components["direct"][:, face] * visible[:, chunk] + diffuse
            ).sum(axis=0, dtype=np.float64) / 1000
    return fields


def bare_roof_placement(scene, receivers, weather, base, local, neighborhood,
                        exclusions, **placement):
    """Preserve the original capacity and full-shading fixed-count MILP policy."""
    masks = {"usable_area": base, "surroundings": base & neighborhood,
             "local_only": base & local, "full": base & local & neighborhood}
    fields = annual_fields(receivers, weather, masks)
    resource = RoofResource(
        scene.samples, receivers, np.full(len(receivers.points), "free_roof"),
        {stage: IrradianceResult(field, np.zeros(len(field)))
         for stage, field in fields.items()},
    )
    designs = {
        stage: optimize_placements(
            scene, replace(resource, results={"full": resource.results[stage]}),
            exclusions, current_panel_count=0, **placement,
        )
        for stage in masks
    }
    capacity = min(design.maximum_count for design in designs.values())
    full = designs["full"]
    selected = select_milp(full.candidates, full.conflict_pairs, panel_count=capacity)
    return replace(full, maximum_count=capacity, maximum_layout=selected)
