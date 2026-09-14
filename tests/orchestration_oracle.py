"""Opt-in oracle loading the copied original pipeline, without rewriting its code.

Only legacy path/download adapters and removed report/benchmark sinks are supplied.
No original implementation is copied from shading_aware_pv into this namespace.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd

_WORKSPACES = {}


def _module(name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    sys.modules[name] = module
    return module


def original_package(reference_root: Path):
    if 'pv-placement-thesis' in reference_root.resolve().parts:
        raise ValueError('Use an independent copied reference, never the original thesis checkout')
    source = reference_root / 'src' / 'solar_poc'
    if not (source / 'simulation.py').is_file():
        raise ValueError('Reference root must contain src/solar_poc/simulation.py')
    # Deliberately do not execute the package __init__, which eagerly imports its
    # web-report stack. Individual original modules are loaded normally unchanged.
    package = '_solar_oracle'
    for name in list(sys.modules):
        if name == package or name.startswith(package + '.'):
            del sys.modules[name]
    _module(package, __path__=[str(source)], __package__=package)
    noop = lambda *args, **kwargs: None
    skipped = lambda *args, **kwargs: {'status':'not_applicable', 'reason':'external benchmark excluded from numerical oracle'}
    _module(package+'.comparison', compare_google_solar=skipped,
            compare_renewables_ninja=skipped, compare_sonnendach=skipped)
    _module(package+'.report', build_report=noop)
    _module(package+'.study_report', write_study_report=noop)
    # The original context reader expects these legacy acquisition entry points.
    # Paths are mapped to already-acquired, checksum-recorded source files below.
    if 'src' not in sys.modules:
        _module('src', __path__=[str(reference_root / 'src')])
    if 'src.emboss' not in sys.modules:
        _module('src.emboss', __path__=[str(reference_root / 'src' / 'emboss')])
    def unavailable(*args, **kwargs):
        raise RuntimeError('Oracle is offline: an unadapted source download was attempted')
    _module('src.emboss.acquisition_sources', search_source_candidates=unavailable)
    _module('src.emboss.lidar', materialize_las=lambda workspace:workspace.cloud_path,
            materialize_las_artifact=unavailable, read_las_points=unavailable)
    _module('src.emboss.workspace', load_workspace=lambda path:_WORKSPACES[str(path)])
    simulation = importlib.import_module(package+'.simulation')
    return SimpleNamespace(simulation=simulation,
        context_lidar=importlib.import_module(package+'.context_lidar'),
        yield_model=importlib.import_module(package+'.yield_model'),
        models=importlib.import_module(package+'.models'), source=source)


def capture_call(function, *args, **kwargs):
    captured = {}
    previous = sys.getprofile()
    def profile(frame, event, arg):
        if event == 'return' and frame.f_code is function.__code__:
            captured.update(frame.f_locals)
        elif event == 'return' and frame.f_code.co_name == 'evaluate_budgets' and '_solar_oracle' in frame.f_globals.get('__name__',''):
            captured['_budget_locals'] = dict(frame.f_locals)
    sys.setprofile(profile)
    try:
        result = function(*args, **kwargs)
    finally:
        sys.setprofile(previous)
    return result, captured


def prepare_original(oracle, inputs, output: Path, context_sources=()):
    output.mkdir(parents=True, exist_ok=True)
    legacy = output / 'legacy-input'
    legacy.mkdir(exist_ok=True)
    case_path = legacy / 'case.json'
    workspace_path = legacy / 'workspace.json'
    case_path.write_text(json.dumps({'case_id':'parity-roof','building_id':inputs.building_fid,
                                    'metadata':{'workspace':str(workspace_path)}}))
    paths = SimpleNamespace(case_json=case_path, tile_dir=legacy,
        roof_mesh=inputs.mesh_path, roof_details=inputs.roof_details_path,
        orthophoto=inputs.orthophoto_path, scaffold_path=inputs.scaffold_path)
    oracle.simulation.resolve_case_paths = lambda path:paths
    oracle.simulation.load_semantic_exclusions = lambda path:None
    for name in ('weather.csv','weather_meta.json','weather_no_horizon.csv','weather_no_horizon_meta.json'):
        shutil.copy2(inputs.cache_dir / name, output / name)
    for station in inputs.cache_dir.glob('ogd-smn_*.csv'):
        shutil.copy2(station, output / station.name)
    if context_sources:
        # Preserve original target-tile selection while making all other tiles
        # available via the old context cache's JSON convention.
        scene = oracle.simulation.load_roof_scene(inputs.mesh_path, sample_spacing=.2, detail_clearance=.15,
            roof_details_path=inputs.roof_details_path, scaffold_path=inputs.scaffold_path)
        center = scene.samples.points[:,:2].mean(axis=0)
        anchor = next(s for s in context_sources if s['bounds'][0] <= center[0] < s['bounds'][2]
                      and s['bounds'][1] <= center[1] < s['bounds'][3])
        workspace = SimpleNamespace(workspace_root=legacy/'workspace/group/tile/work',
            surfaces_vector_path=inputs.surfaces_path,tile_key=anchor['tile'],cloud_path=Path(anchor['path']))
        _WORKSPACES[str(workspace_path)] = workspace
        cache = workspace.workspace_root.parents[2] / 'cache/solar-context'
        cache.mkdir(parents=True, exist_ok=True)
        by_tile = {s['tile']:s for s in context_sources}
        oracle.context_lidar.materialize_tile = lambda tile,cache:Path(by_tile[tile['surface_item_id']]['path'])
        for source in context_sources:
            x,y = source['bounds'][:2]
            (cache/f"tile_{x}_{y}_year_{anchor['year']}.json").write_text(json.dumps({
                'surface_item_id':source['tile'], 'surface_asset_href':source['source']['asset_href'],
                'surface_file_name':source['source']['file_name']}))
    return paths


def difference(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape:
        return {'equal':False, 'left_shape':list(left.shape), 'right_shape':list(right.shape)}
    if left.dtype.kind in 'OUS' or right.dtype.kind in 'OUS':
        return {'equal':bool(np.array_equal(left,right)), 'mismatches':int(np.count_nonzero(left!=right))}
    matching = np.isclose(left,right,rtol=0,atol=0,equal_nan=True)
    error = np.where(matching,0,np.abs(left.astype(np.float64)-right.astype(np.float64)))
    return {'equal':bool(matching.all()), 'shape':list(left.shape),
            'max_abs':float(error.max(initial=0)), 'mismatches':int(np.count_nonzero(~matching))}


def corner_set(corners):
    # Compare physical rectangles independently of MILP/corner enumeration order.
    return sorted(tuple(sorted(tuple(map(float,p)) for p in cell)) for cell in corners)


def source_hashes(source):
    return {path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(source.glob('*.py'))}


def input_hashes(inputs, context_sources):
    paths = [inputs.mesh_path, inputs.roof_details_path, inputs.orthophoto_path,
             inputs.surfaces_path, inputs.scaffold_path, *inputs.cache_dir.glob('weather*'),
             *inputs.cache_dir.glob('ogd-smn_*.csv'),
             *(Path(source['path']) for source in context_sources)]
    result = {}
    for path in paths:
        if path.is_file():
            with path.open('rb') as stream:
                result[str(path)] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return result


def compare_case(reference_root: Path, inputs, config, output: Path, *, context_sources=()):
    from shading_aware_pv.simulation import simulate
    oracle = original_package(reference_root)
    paths = prepare_original(oracle, inputs, output/'original', context_sources)
    options = {key:value for key,value in asdict(config).items()
               if key in oracle.simulation.SimulationConfig.__dataclass_fields__}
    original_config = oracle.simulation.SimulationConfig(target_key='parity',case_json=paths.case_json,
        output_dir=output/'original',solar_data_dir=output/'empty-measurements',**options)
    target = oracle.simulation.RoofTarget('parity','Parity fixture',0.,0.)
    print('Executing unchanged original run_simulation',flush=True)
    _, old = capture_call(oracle.simulation.run_simulation,original_config,target)
    print('Executing new simulate independently',flush=True)
    result, new = capture_call(simulate,inputs,config)
    (output/'new-result.json').write_text(json.dumps(result,allow_nan=False))
    resource = old['resource']
    comparisons = {
        'receiver_points':difference(resource.receivers.points,new['resource'].receivers.points),
        'receiver_normals':difference(resource.receivers.normals,new['resource'].receivers.normals),
        'receiver_areas':difference(resource.receivers.areas,new['resource'].receivers.areas),
        'roof_use_labels':difference(resource.use_labels,new['resource'].use_labels),
        'weather':difference(old['weather'].hourly.to_numpy(),new['weather'].hourly.to_numpy()),
        'irradiation':{state:difference(resource.results[state].irradiation,
                     new['resource'].results[state].irradiation) for state in old['resource_visibility']},
        'visibility':{state:difference(mask,new['visibility'][state]) for state,mask in old['resource_visibility'].items()},
    }
    if old['context'] is not None:
        comparisons['context'] = {
            'points':difference(old['context'].source_points,new['context'].source_points),
            'classes':difference(old['context'].source_classes,new['context'].source_classes),
            'surface':difference(old['context'].surface_mesh.vertices,new['context'].surface_mesh.vertices),
            'without_vegetation':difference(old['context'].surface_mesh_without_vegetation.vertices,
                                           new['context'].surface_mesh_without_vegetation.vertices)}
    designs = {}
    if old['panel_layout'] is not None and 'optimization' in old:
        for name, analysis_key, solution in (
            ('installed','yield_analysis',None),
            ('relocated','relocated_analysis',old['optimization'].relocated_layout),
            ('clean_slate','clean_analysis',old['optimization'].maximum_layout)):
            analysis = old[analysis_key]
            expected = result['designs'][name]
            corners = ([cell.receiver_corners_xyz for cell in old['panel_layout'].modules.cells] if solution is None
                       else [old['optimization'].candidates[i].receiver_corners_xyz for i in solution.candidate_ids])
            designs[name] = {'panel_count':len(corners), 'new_panel_count':expected['panel_count'],
                'corners_equal':corner_set(corners)==corner_set(expected['corners']),
                'electrical':difference(list(asdict(analysis.inputs).values())[:-1],list(expected['electrical'].values())[:-1]),
                'states':{state:difference(analysis.hourly[f'modeled_{state}_kwh'], expected['hourly']['ac_kwh'][state])
                          for state in old['resource_visibility']},
                'snow_loss_kwh':{state:analysis.metrics[state]['snow_loss_kwh'] for state in old['resource_visibility']}}
    else:
        # This is the actual old no-PV output: 100%-capacity full-stage budget,
        # which is explicitly snow-free. Do not silently substitute its newer
        # installed-layout orchestration when judging the migration.
        row = next(row for row in old['study']['rows'] if row['experiment']=='capacity_100' and row['stage']=='full')
        expected=result['designs']['clean_slate']
        designs['old_no_pv_full_budget'] = {'panel_count':row['requested_count'],
            'new_panel_count':expected['panel_count'], 'corners_equal':corner_set(row['corners'])==corner_set(expected['corners']),
            'old_snow_free_ac_kwh':row['ac_kwh'], 'new_ac_kwh':expected['states']['full']['modeled_kwh'],
            'ac_difference_kwh':expected['states']['full']['modeled_kwh']-row['ac_kwh']}
        # The original no-PV path does not emit six-state yield. Compare its
        # unchanged yield function on the exact old full-budget selected cells,
        # labeling this as a component check rather than old pipeline output.
        budget_locals=old['_budget_locals']
        budget_design=budget_locals['designs']['full']
        selected_candidates=budget_locals['evaluated'][(row['requested_count'],'full')]['candidate_ids']
        selected=np.unique(np.concatenate([budget_design.candidates[i].roof_sample_indices for i in selected_candidates]))
        sample_type=oracle.models.RoofSamples
        samples=sample_type(*(getattr(resource.receivers,key)[selected] for key in ('points','normals','areas','face_ids')))
        electrical=oracle.simulation._electrical_inputs(original_config,row['requested_count'],
            old['study']['placement']['module_width_m']*old['study']['placement']['module_height_m'],
            module_wattage_w=old['study']['module_wattage_w'],snow_num_strings=1,
            snow_num_strings_basis='portrait module default')
        analysis=oracle.yield_model.analyze_yield(SimpleNamespace(samples=samples),old['weather'],
            {state:mask[:,selected] for state,mask in old['resource_visibility'].items()},electrical,
            snow=None,state_weather={'open_horizon':old['open_horizon_weather']})
        designs['old_snow_free_yield_component_on_old_budget'] = {
            'states':{state:difference(analysis.hourly[f'modeled_{state}_kwh'],expected['hourly']['ac_kwh'][state])
                      for state in old['resource_visibility']}}
    report = {'config':asdict(config), 'input_hashes':input_hashes(inputs,context_sources),
              'original_simulation_sha256':source_hashes(oracle.source)['simulation.py'],
              'reference_hashes':source_hashes(oracle.source), 'comparisons':comparisons,'designs':designs,
              'old_snow':old['snow'].metadata,'new_snow':result['snow'],
              'old_study_capacity':old['study']['capacity'],'new_capacity':new['optimization'].maximum_count}
    (output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    from shading_aware_pv.inputs import SimulationInputs
    from shading_aware_pv.simulation import SimulationConfig
    manifest_path, output = map(Path, sys.argv[1:])
    manifest = json.loads(manifest_path.read_text())
    for case in manifest['cases']:
        values = dict(case['inputs'])
        for key in ('mesh_path','roof_details_path','orthophoto_path','surfaces_path','scaffold_path',
                    'cache_dir','context_cache_dir'):
            if values.get(key) is not None:
                values[key] = Path(values[key])
        values['lidar_paths'] = tuple(map(Path, values.get('lidar_paths',())))
        values['sources'] = tuple(values.get('sources',()))
        compare_case(Path(manifest['reference_root']), SimulationInputs(**values),
                     SimulationConfig(**case.get('config',{})), output/case['name'],
                     context_sources=case.get('context_sources',()))
