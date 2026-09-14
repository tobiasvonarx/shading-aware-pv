from concurrent.futures import Future
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from shading_aware_pv import api, counts


def report_fixture(root, *, variants=None):
    job_id = 'a' * 32
    job = {'id': job_id, 'house_id': 'roof', 'status': 'complete', 'config': {},
           'created_at': '2026-01-01', 'result_url': f'/api/solar/jobs/{job_id}/result'}
    directory = root / 'solar/jobs';directory.mkdir(parents=True, exist_ok=True)
    (directory / f'{job_id}.json').write_text(json.dumps(job))
    report = {'designs_by_count': variants or {}, 'schema': 'original', 'provenance': {'method_revision': api.METHOD_REVISION}}
    (directory / f'{job_id}.result.json').write_text(json.dumps(report))
    return job, report


def test_count_preparation_persists_isolated_result_and_is_idempotent(tmp_path, monkeypatch):
    app=api.create_app(tmp_path);job, original=report_fixture(tmp_path);job_id=job['id']
    submitted=[]
    monkeypatch.setattr(counts,'prepare_count_report',lambda *args,**kwargs:{'schema':'prepared','designs_by_count':{'0':{'panel_count':0}}})
    def submit(*args):submitted.append(args);return Future()
    with TestClient(app) as client:
        monkeypatch.setattr(app.state.executor,'submit',submit)
        endpoint=f'/api/solar/jobs/{job_id}/counts'
        status=client.post(endpoint).json();assert status['status']=='queued'
        assert client.post(endpoint).json()==status;assert len(submitted)==1
        assert client.delete(f'/api/solar/jobs/{job_id}').status_code==409
        fn,*args=submitted[0];fn(*args)
        ready=client.get(endpoint).json();assert ready['status']=='complete'
        assert client.get(ready['result_url']).json()['schema']=='prepared'
        assert client.get(job['result_url']).json()==original
        assert client.post(endpoint).json()['status']=='complete';assert len(submitted)==1
        assert client.delete(f'/api/solar/jobs/{job_id}').status_code==200
        assert client.get(endpoint).status_code==404
        assert client.get(ready['result_url']).status_code==404
    with TestClient(api.create_app(tmp_path)) as client:
        client.post(f'/api/solar/jobs/{job_id}/restore')
        assert client.get(endpoint).json()['status']=='complete'


def test_existing_counts_return_original_result_without_worker(tmp_path, monkeypatch):
    app=api.create_app(tmp_path);job,_=report_fixture(tmp_path,variants={'0':{'panel_count':0}})
    with TestClient(app) as client:
        monkeypatch.setattr(app.state.executor,'submit',lambda *args:pytest.fail('Should use saved variants'))
        for method in (client.get,client.post):
            status=method(f"/api/solar/jobs/{job['id']}/counts").json()
            assert status['status']=='complete' and status['result_url']==job['result_url']


def test_interrupted_preparation_retries_and_stale_result_is_hidden(tmp_path, monkeypatch):
    app=api.create_app(tmp_path);job,_=report_fixture(tmp_path);job_id=job['id']
    directory=tmp_path/'solar/counts';directory.mkdir()
    path=directory/f'{job_id}.json';path.write_text(json.dumps({'id':job_id,'status':'running','attempt_id':'old'}))
    (directory/f'{job_id}.result.json').write_text('{"stale":true}')
    with TestClient(app) as client:
        monkeypatch.setattr(app.state.executor,'submit',lambda *args:Future())
        endpoint=f'/api/solar/jobs/{job_id}/counts'
        status=client.get(endpoint).json();assert status['status']=='failed' and 'restart' in status['message']
        assert client.get(endpoint+'/result').status_code==404
        restarted=client.post(endpoint).json();assert restarted['status']=='queued'
        assert restarted['attempt_id']!='old'
        api._run_count_job(str(tmp_path),job_id,'old')
        assert client.get(endpoint).json()['attempt_id']==restarted['attempt_id']


def test_process_failure_can_retry_and_symlink_paths_are_rejected(tmp_path, monkeypatch):
    app=api.create_app(tmp_path);job,_=report_fixture(tmp_path);job_id=job['id'];future=Future()
    with TestClient(app) as client:
        monkeypatch.setattr(app.state.executor,'submit',lambda *args:future)
        endpoint=f'/api/solar/jobs/{job_id}/counts';client.post(endpoint)
        future.set_exception(RuntimeError('stopped'))
        assert client.get(endpoint).json()['status']=='failed'
        path=tmp_path/'solar/counts'/f'{job_id}.result.json';path.symlink_to(tmp_path/'solar/jobs'/f'{job_id}.result.json')
        assert client.post(endpoint).status_code==404
        assert client.get(endpoint+'/result').status_code==404
        job['status']='running';(tmp_path/'solar/jobs'/f'{job_id}.json').write_text(json.dumps(job))
        assert client.post(endpoint).status_code==409


def test_missing_or_changed_cached_inputs_fail_without_reconstruction(tmp_path, monkeypatch):
    from emboss import api as emboss_api
    mesh=tmp_path/'mesh.ply';mesh.write_text('ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n')
    for name in ('details','scaffold','image'):(tmp_path/name).touch()
    reconstruction=SimpleNamespace(mesh_path=mesh,roof_details_path=tmp_path/'details',scaffold_path=tmp_path/'scaffold',orthophoto_path=tmp_path/'image')
    class Client:
        def __init__(self,*args,**kwargs):pass
        def load_result(self,house):return reconstruction
        def reconstruct(self,*args,**kwargs):pytest.fail('Must not reconstruct saved inputs')
    monkeypatch.setattr(emboss_api,'Client',Client)
    report={'scene':{'vertices':[[0,0,1],[1,0,1],[0,1,1]],'faces':[[0,1,2]]}}
    with pytest.raises(ValueError,match='reconstruction has changed'):
        counts.prepare_count_report(tmp_path,{'house_id':'roof'},report,progress=lambda _:None)
    (tmp_path/'image').unlink()
    with pytest.raises(FileNotFoundError,match='Cached input is missing'):
        counts.prepare_count_report(tmp_path,{'house_id':'roof'},report,progress=lambda _:None)


def test_replay_compares_floored_analysis_timestamps_and_keeps_cached_weather(tmp_path, monkeypatch):
    import pandas as pd
    from emboss import api as emboss_api
    mesh=tmp_path/'mesh.ply';mesh.write_text('ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n')
    for name in ('details','scaffold','image'):(tmp_path/name).touch()
    reconstruction=SimpleNamespace(mesh_path=mesh,roof_details_path=tmp_path/'details',scaffold_path=tmp_path/'scaffold',orthophoto_path=tmp_path/'image')
    class Client:
        def __init__(self,*args,**kwargs):pass
        def load_result(self,house):return reconstruction
        def reconstruct(self,*args,**kwargs):pytest.fail('No reconstruction during replay')
    monkeypatch.setattr(emboss_api,'Client',Client)
    monkeypatch.setattr(counts.SimulationInputs,'from_reconstruction',lambda *args,**kwargs:object())
    directory=tmp_path/'solar/weather/roof/2023';directory.mkdir(parents=True)
    index=pd.date_range('2023-01-01 00:10',periods=2,freq='h',tz='UTC')
    frame=pd.DataFrame({'ghi':[12.,34.]},index=index);frame.index.name='time'
    for name,horizon in [('weather',True),('weather_no_horizon',False)]:
        frame.to_csv(directory/f'{name}.csv')
        (directory/f'{name}_meta.json').write_text(json.dumps({'year':2023,'use_horizon':horizon,'latitude':47,'longitude':8,'elevation':450,'source':'cached fixture'}))
    def simulate(inputs,config,**kwargs):
        assert not config.refresh_weather
        assert kwargs['weather'].hourly.index.equals(index)
        assert kwargs['weather'].hourly.ghi.tolist()==[12.,34.]
        assert kwargs['open_horizon_weather'].hourly.index.equals(index)
        assert not kwargs['snow'].available
        return {'provenance':{'method_revision':api.METHOD_REVISION}}
    monkeypatch.setattr(counts,'simulate',simulate)
    report={'provenance':{'method_revision':2},'scene':{'vertices':[[0,0,0],[1,0,0],[0,1,0]],'faces':[[0,1,2]]},
            'weather_hours':2,'designs':{'clean_slate':{'hourly':{'timestamps':index.floor('h').astype(str).tolist()}}},
            'snow':{'available':False},'context':{'enabled':False}}
    job={'id':'a'*32,'house_id':'roof','config':{'year':2023,'refresh_weather':True}}
    result=counts.prepare_count_report(tmp_path,job,report,progress=lambda _:None)
    assert result['count_preparation']['method_migration']
    assert result['config']==job['config']


@pytest.mark.parametrize("prepared", [False, True])
def test_older_method_counts_are_rebuilt(tmp_path, monkeypatch, prepared):
    app = api.create_app(tmp_path)
    job, report = report_fixture(tmp_path, variants={'0': {'panel_count': 0}})
    job_id = job['id']
    report['provenance']['method_revision'] = 3
    (tmp_path / 'solar/jobs' / f'{job_id}.result.json').write_text(json.dumps(report))
    if prepared:
        directory = tmp_path / 'solar/counts'
        directory.mkdir()
        (directory / f'{job_id}.json').write_text(json.dumps({
            'id': job_id, 'status': 'complete', 'method_revision': 3,
        }))
        (directory / f'{job_id}.result.json').write_text(json.dumps(report))
    with TestClient(app) as client:
        submitted = []
        def submit(*args):
            submitted.append(args)
            return Future()
        monkeypatch.setattr(app.state.executor, 'submit', submit)
        endpoint = f'/api/solar/jobs/{job_id}/counts'
        assert client.get(endpoint + '/result').status_code == 404
        status = client.post(endpoint).json()
        assert status['status'] == 'queued'
        assert status['method_revision'] == api.METHOD_REVISION
        assert len(submitted) == 1
        assert client.get(job['result_url']).json() == report
