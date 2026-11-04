from pathlib import Path
import tempfile
import hashlib
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from show_runup_shooting import verify_runup_config
from targeted_shooting import controller_provenance, CONTROLLER_SOURCES, MOTION_FILE, UPSTREAM_SOURCE


class PreviewProvenanceTests(unittest.TestCase):
    def test_changed_upstream_dynamics_are_rejected_even_when_listed_files_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for name in (*CONTROLLER_SOURCES,MOTION_FILE,UPSTREAM_SOURCE+'/robots/g1.py'):
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('original')
            config=dict(source_sha256={CONTROLLER_SOURCES[0]:hashlib.sha256((root/CONTROLLER_SOURCES[0]).read_bytes()).hexdigest()},
                upstream_sources_sha256=controller_provenance(root)['controller_sources_sha256'][UPSTREAM_SOURCE])
            verify_runup_config(config,root)
            (root/UPSTREAM_SOURCE/'robots/g1.py').write_text('changed torque limit')
            with self.assertRaisesRegex(ValueError,'upstream'):
                verify_runup_config(config,root)

    def test_missing_frozen_metadata_is_rejected(self):
        with self.assertRaises(ValueError):
            verify_runup_config({'source_sha256':{}},Path('/tmp'))


if __name__=='__main__':
    unittest.main()
