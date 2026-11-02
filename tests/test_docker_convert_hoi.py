"""Check Docker arguments without building images or converting user data."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DockerHoiTests(unittest.TestCase):
    def test_read_only_inputs_exact_output_and_argument_forwarding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'hoi! input'
            source.mkdir()
            output = root / 'output parent'
            output.mkdir()
            fake = root / 'docker'
            fake.write_text('#!/usr/bin/env python3\nimport json,os,sys\n'
                            'with open(os.environ["DOCKER_LOG"],"a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n')
            fake.chmod(0o755)
            log = root / 'calls.jsonl'
            env = dict(os.environ, PATH=f'{root}:{os.environ["PATH"]}',
                       HOI_DIR=str(source), HOI_OUTPUT_ROOT=str(output / 'hoi'),
                       DOCKER_LOG=str(log), IMAGE='urdf-sapien-renderer:test')
            command = ['bash', str(ROOT / 'scripts/docker_convert_hoi.sh'),
                       '--locations', 'fridge_1', '--fetch-missing-metadata', '--dry-run']
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            build, run = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(build, ['build', '-t', env['IMAGE'], str(ROOT)])
            self.assertIn(f'type=bind,src={source},dst=/data/hoi,readonly', run)
            self.assertIn(f'type=bind,src={ROOT},dst=/workspace,readonly', run)
            self.assertIn(f'type=bind,src={output},dst=/data/output', run)
            self.assertIn(f'{os.getuid()}:{os.getgid()}', run)
            self.assertNotIn('--gpus', run)
            self.assertEqual(run[run.index('python3'):],
                             ['python3', '-B', 'scripts/convert_hoi_dataset.py',
                              '--input', '/data/hoi', '--output-root', '/data/output/hoi',
                              '--locations', 'fridge_1', '--fetch-missing-metadata', '--dry-run'])
            self.assertFalse((output / 'hoi').exists())
            log.unlink()
            source.rmdir()
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(log.exists())


if __name__ == '__main__':
    unittest.main()
