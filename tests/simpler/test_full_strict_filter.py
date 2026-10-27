import sys,unittest,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/simpler/eval_search'))
from full_filter_lookahead import evaluate_filter
# Reference values from the dataset filters (filter_strict_distance_success.distance_metrics,
# filter_first_try_grasp.grasp_metrics) on the episode below. Those scripts left this repo with
# the data collection code: github.com/harine/RoboMonkey @ 2634271, scriptsv2/data_collect/.
REF_BACKTRACK, REF_MAX_CONTACT_GAP, REF_DGE_RISE = 0.0, 1, 0.004
class Tests(unittest.TestCase):
 def test_exact_retrospective_metrics_match_dataset_code(self):
  d=np.array([.2,.2,.18,.1,.05]);g=np.array([.1,.02,.021,.023,.024]);c=np.array([0,1,1,0,1]);o=np.array([1,0,0,0,0])
  h=[dict(deb=float(a),dge=float(b),contact=bool(e),open=bool(f),success=i==4) for i,(a,b,e,f) in enumerate(zip(d,g,c,o))]
  x=evaluate_filter(h);json.dumps(x)
  self.assertTrue(x['passed']);self.assertAlmostEqual(x['backtrack_m'],REF_BACKTRACK)
  self.assertEqual(x['max_contact_gap'],REF_MAX_CONTACT_GAP);self.assertAlmostEqual(x['separation_rise_m'],REF_DGE_RISE)
  h[-1]['deb']=.25;self.assertFalse(evaluate_filter(h)['checks']['bounded_backtrack'])
 def test_missing_grasp_and_nonfinite_fail(self):
  h=[dict(deb=.2,dge=.1,contact=False,open=True,success=False)]
  self.assertFalse(evaluate_filter(h)['passed']);h[0]['deb']=float('nan')
  self.assertEqual(evaluate_filter(h)['score'],-1e6)
if __name__=='__main__':unittest.main()
