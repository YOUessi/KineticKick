import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from eval_free_shooting import motion_provenance


class MotionProvenanceTests(unittest.TestCase):
    def test_sweep_records_actual_references_without_false_singular_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = [Path(directory) / name for name in ('first.npz', 'second.npz')]
            first.write_bytes(b'first reference')
            second.write_bytes(b'second reference')
            source = dict(motion_file=str(first), motion_sha256=hashlib.sha256(first.read_bytes()).hexdigest(),
                          controller_sources_sha256={'controller': 'hash'})
            single = motion_provenance(source, [str(first)])
            self.assertEqual(single['motion_file'], str(first))
            self.assertEqual(single['motion_sha256'], source['motion_sha256'])
            sweep = motion_provenance(source, [str(first), str(second)])
            self.assertNotIn('motion_file', sweep)
            self.assertNotIn('motion_sha256', sweep)
            self.assertEqual(sweep['motion_sources_sha256'],
                             {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (first, second)})
            self.assertEqual(source['motion_file'], str(first))
            first.write_bytes(b'changed since preflight')
            with self.assertRaises(ValueError):
                motion_provenance(source, [str(first)])


if __name__ == '__main__':
    unittest.main()
