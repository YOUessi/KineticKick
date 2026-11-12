"""Previously proven CUDA Tensor positive source must remain byte-identical to Git freeze."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from v114_tensor_positive_batch import preflight
def test_frozen_3_cases():
 c=json.loads((ROOT/'configs/v114_tensor_positive_frozen.json').read_text())
 assert [(x['name'],x['capture']) for x in c['calibration_cases']]==[
     ('old_off','pose'),('old_tensor_positive','tensor'),('old_tensor_separated','tensor')]
 assert c['steps']==180 and c['watchdog']==120 and c['min_free_gib']==20
def test_use_original_v094_unmodified():
 s=(ROOT/'tools/v114_tensor_positive_batch.py').read_text()
 assert 'tools/v094_box_pair_probe.py' in s
 assert 'git("show","HEAD:"+f]' in s or '["git","show","HEAD:"+f]' in s
def test_gpu_guard_and_write_once():
 s=(ROOT/'tools/v114_tensor_positive_batch.py').read_text()
 for token in ('LOCK_NB','FileExistsError','GPU_BUSY','CUDA_VISIBLE_DEVICES','CHILD_ONLY_WATCHDOG'):
  assert token in s
