from types import SimpleNamespace
import numpy as np
import pytest
from shading_aware_pv.optimization import PlacementCandidate, select_milp
from shading_aware_pv.simulation import _designs_by_count
from shading_aware_pv.counts import _assert_same_baseline


def test_every_count_is_solved_instead_of_truncating_maximum():
    # Highest-value module conflicts with both lower-value modules. At count1
    # choose it; at count2 choose the other two, so no nested subset is valid.
    candidates=tuple(PlacementCandidate(i,'roof','portrait',np.zeros((4,3)),np.zeros((4,3)),np.array([i]),score)
                     for i,score in enumerate([10.,6.,6.]))
    conflicts=np.array([[0,1],[0,2]])
    maximum=select_milp(candidates,conflicts,panel_count=2)
    optimization=SimpleNamespace(candidates=candidates,conflict_pairs=conflicts,maximum_count=2,maximum_layout=maximum)
    maximum_design={'panel_count':2,'corners':['B','C'],'states':{'full':{'modeled_kwh':12}},'hourly':{'exact':'preserved'}}
    def evaluate(solution, *, include_hourly):
        assert not include_hourly
        return {'panel_count':solution.panel_count,'selected':solution.candidate_ids.tolist()}
    variants=_designs_by_count(optimization,maximum_design,evaluate,lambda _:None)
    assert variants['0']=={'panel_count':0,'selected':[]}
    assert variants['1']=={'panel_count':1,'selected':[0]}
    assert variants['2']=={key:value for key,value in maximum_design.items() if key!='hourly'}
    assert maximum_design['hourly']=={'exact':'preserved'}


def test_zero_capacity_is_a_valid_only_slider_entry():
    design={'panel_count':0,'corners':[],'states':{'full':{'modeled_kwh':0}}}
    variants=_designs_by_count(SimpleNamespace(maximum_count=0),design,lambda *a,**kw:pytest.fail('No solver/evaluation needed'),lambda _:None)
    assert variants=={'0':design}


def test_replay_detects_changed_same_method_inputs_but_allows_roundoff():
    saved={'module_dimensions_m':[1,1.7],'scene':{'vertices':[[0.,0.,2.]]},'designs':{'clean_slate':{'states':{'full':{'modeled_kwh':10.}}}},'context':{'enabled':False}}
    import copy
    equivalent=copy.deepcopy(saved);equivalent['designs']['clean_slate']['states']['full']['modeled_kwh']+=1e-10
    _assert_same_baseline(saved,equivalent)
    changed=copy.deepcopy(saved);changed['module_dimensions_m'][0]=1.1
    with pytest.raises(ValueError,match='no longer reproduce'):_assert_same_baseline(saved,changed)
    changed=copy.deepcopy(saved);changed['designs']['clean_slate']['states']['full']['modeled_kwh']=11.
    with pytest.raises(ValueError,match='no longer reproduce'):_assert_same_baseline(saved,changed)
