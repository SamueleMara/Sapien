"""CPU-only regression tests for the generic camera/profile/rewrite CLI."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('sapien_run', ROOT/'scripts'/'run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

class GenericRunnerTests(unittest.TestCase):
    def test_camera_auto_select_and_counts(self):
        for n in (8, 64, 100):
            p = runner.resolve_views_json(None, ROOT, n)
            self.assertEqual(len(json.loads(p.read_text())['cameras']), n)
        self.assertEqual(runner.resolve_views_json(None, ROOT, 8).name, 'cameras.PARIS.json')
        self.assertEqual(runner.resolve_views_json(None, ROOT, 64).name, 'cameras.hemisphere_64.json')
        self.assertEqual(runner.resolve_views_json(None, ROOT, 100).name, 'cameras.hemisphere_100.json')

    def test_output_root_profiles_not_hardcoded_per_case(self):
        for cams in (8, 64, 100):
            for profile in ('linear', 'ease-in-out', 'trapezoid', 'ease-out'):
                p = runner.resolve_output_root(None, ROOT, num_cams=cams, velocity_profile=profile)
                self.assertEqual(p.name, 'partnet_subset_fixed_render_' + profile)
                self.assertEqual(p.parent.name, 'preparation_' + str(cams) + 'cams')
        explicit = Path('/tmp/my_existing_data')
        self.assertEqual(runner.resolve_output_root(str(explicit), ROOT, num_cams=100, velocity_profile='linear'), explicit)

    def test_parse_case_insensitive_profiles(self):
        for raw, expected in [('LiNeAr','linear'), ('EASE_IN_OUT','ease-in-out'), ('Trapezoid','trapezoid')]:
            self.assertEqual(runner.normalize_velocity_profile(raw), expected)

    def test_dry_run_auto_configs_and_no_existing_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp/'one.urdf').write_text('<robot name="one"/>')
            for cams in (8, 64, 100):
                output = tmp / ('render_'+str(cams))
                (output/'one').mkdir(parents=True)
                protected = output/'one'/'SENTINEL'
                protected.write_text('not deleted')
                cmd=[sys.executable, str(ROOT/'scripts'/'run.py'), '--urdf-dir',str(tmp),
                     '--output-root',str(output), '--num-cams',str(cams),
                     '--velocity-profile','LiNeAr', '--dry-run', '--rewrite']
                result=subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
                self.assertIn('num_cameras='+str(cams),result.stdout)
                self.assertTrue(protected.exists(), 'dry-run deleted an output!')

    def test_explicit_config_count_mismatch_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            (tmp/'one.urdf').write_text('<robot name="one"/>')
            result=subprocess.run([sys.executable, str(ROOT/'scripts'/'run.py'),
                '--urdf-dir',str(tmp), '--output-root',str(tmp/'output'),
                '--num-cams','100','--camera-config-json',str(ROOT/'configs'/'cameras.PARIS.json'),
                '--dry-run'], capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('does not match',result.stderr)

    def test_rewrite_replaces_existing_and_runs_render_and_colmap(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            urdfroot=tmp/'urdfs';urdfroot.mkdir()
            (urdfroot/'one.urdf').write_text('<robot name="one"><link name="base"/></robot>')
            output=tmp/'output';(output/'one').mkdir(parents=True)
            old=(output/'one'/'OLD_FILE');old.write_text('old dataset')
            commands=[]
            def mock_list(command, dry_run=False):
                return subprocess.CompletedProcess(command, 0, stdout=json.dumps([
                    {'name':'joint_0','dof':1,'limits':[[0.0,1.0]]}]))
            def mock_command(command, dry_run=False):
                commands.append(command)
                if 'render' in command:
                    destination=Path(command[command.index('--output-dir')+1])
                    destination.mkdir(parents=True, exist_ok=True)
                    (destination/'NEW_RENDER').write_text('new')
                if command[1].endswith('export_colmap_dataset.py'):
                    destination=Path(command[command.index('--model-output')+1])
                    (destination/'colmap').mkdir(exist_ok=True)
                    (destination/'colmap_final').mkdir(exist_ok=True)
                return subprocess.CompletedProcess(command,0,stdout='')
            command=[str(ROOT/'scripts'/'run.py'), '--urdf-dir',str(urdfroot),
                     '--output-root',str(output), '--num-cams','8',
                     '--velocity-profile','LiNeAr', '--num-frames','60',
                     '--rewrite', '--skip-renderer-preflight']
            with patch.object(sys,'argv',command), \
                 patch.object(runner,'run_command',side_effect=mock_command), \
                 patch.object(runner,'run_list_joints_command',side_effect=mock_list):
                self.assertEqual(runner.main(),0)
            self.assertFalse(old.exists())
            self.assertTrue((output/'one'/'NEW_RENDER').exists())
            self.assertEqual(len([cmd for cmd in commands if 'render' in cmd]),1)
            self.assertTrue(any(cmd[1].endswith('export_colmap_dataset.py') for cmd in commands))
            render=next(cmd for cmd in commands if 'render' in cmd)
            self.assertEqual(render[render.index('--velocity-profile')+1], 'linear')
            self.assertEqual(Path(render[render.index('--views-json')+1]).name, 'cameras.PARIS.json')

    def test_no_rewrite_existing_object_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            urdfroot=tmp/'urdfs';urdfroot.mkdir()
            (urdfroot/'one.urdf').write_text('<robot name="one"/>')
            output=tmp/'output';(output/'one').mkdir(parents=True)
            (output/'one'/'SENTINEL').write_text('preserved')
            result=subprocess.run([sys.executable,str(ROOT/'scripts'/'run.py'),
                '--urdf-dir',str(urdfroot),'--output-root',str(output),
                '--skip-renderer-preflight'], capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
            self.assertIn('[SKIPPED existing]',result.stdout)
            self.assertTrue((output/'one'/'SENTINEL').exists())
            summary=json.loads((output/'pipeline_summary.json').read_text())
            self.assertEqual(summary['results'][0]['status'],'skipped_existing')

if __name__=='__main__':
    unittest.main()
