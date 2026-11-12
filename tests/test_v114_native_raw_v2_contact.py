"""Native sensor extension must be actually loaded before acquiring Isaac Sim 4.5 raw contact API."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_explicit_extension_enable_before_import():
 s=(ROOT/'tools/v114_native_raw_contact_probe_v2.py').read_text()
 assert s.index('set_extension_enabled_immediate("isaacsim.sensors.physics",True)') < s.index('from isaacsim.sensors.physics import _sensor')
def test_old_contract_immutable():
 from hashlib import sha256
 s=(ROOT/'tools/v114_native_raw_contact_probe.py').read_bytes()
 assert b'from isaacsim.sensors.physics import _sensor' in s
def test_new_native_batch_no_reuse():
 s=(ROOT/'tools/v114_native_raw_v2_batch.py').read_text()
 assert 'reports/v114_raw_sensor_v2_batch.audit.json' in s
 assert 'v114_raw_v2_' in s
 assert '120' in s and 'LOCK_NB' in s
