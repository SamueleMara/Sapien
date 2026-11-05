#!/usr/bin/env python3
"""Apply reviewed spatial SAM profiles to existing real datasets, without reconversion."""
import argparse
import json
from pathlib import Path
import sys

from PIL import Image

import create_standard_masks as masker
from robot_mask_guidance import prepare, guidance_path, load_cached
from sam2_point_guidance import read_guidance

PROFILES = [
    ('monitor','black_monitor_20261021_161154','black monitor'),
    ('laptop','final_black_laptop','black laptop'),
    ('laptop','laptop_20261021_161154','laptop'),
    ('matchbox','final_matchbox','matchbox'),
    ('robot_dog','final_robot_dog','robot dog'),
    ('wooden_box','final_wooden_box','wooden box'),
    ('black_box','final_black_box','black box'),
    ('black_box','black_box_20261021_161154','black box'),
    ('orange_suitcase','final_orange_suitecase','orange suitcase'),
]


def plan(root, groups, profiles_root):
    jobs=[]
    for group,prefix,prompt in PROFILES:
        if group not in groups: continue
        profile=profiles_root/(prefix+'.json')
        for scene in sorted(root.glob(prefix+'_episode_*')):
            if not scene.is_dir(): continue
            if scene.is_symlink() or scene.resolve().parent!=root:
                raise ValueError(f'Correction scene must be inside the output root: {scene}')
            metadata=json.loads((scene/'metadata/real_conversion.json').read_text())
            if (scene/'conversion_in_progress.json').exists():
                raise ValueError(f'RGB-D conversion is incomplete: {scene}')
            parquet=scene/f'metadata/raw/episode_{metadata["episode"]:06d}.parquet'
            if not parquet.is_file(): raise FileNotFoundError(parquet)
            frames=masker.list_frames(scene)
            seen=set()
            for frame in frames:
                if frame.parent.name in seen: continue
                seen.add(frame.parent.name)
                with Image.open(frame) as image: read_guidance(profile,frame.parent.name,image.size)
            jobs.append((scene,profile,prompt))
    for group in groups:
        if not any(PROFILE[0]==group and job[1].stem==PROFILE[1] for PROFILE in PROFILES for job in jobs):
            raise ValueError(f'No converted scenes for group {group} under {root}')
    return jobs


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True,help='Existing standard real output root')
    parser.add_argument('--groups',nargs='+',choices=['monitor','laptop','matchbox','robot_dog','wooden_box','black_box','orange_suitcase'],default=['monitor','laptop','matchbox','robot_dog','wooden_box','black_box','orange_suitcase'])
    parser.add_argument('--profiles-root',type=Path,default=Path(__file__).resolve().parents[1]/'configs/masks')
    parser.add_argument('--parquet-python',type=Path,default=Path(sys.executable),help='Python with polars for uncached robot poses; saved validated pose guidance is reused by default')
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--overwrite',action='store_true',help='Replace old text masks with the reviewed correction profiles')
    modes.add_argument('--resume',action='store_true',help='Reuse masks recorded with these same correction settings')
    parser.add_argument('--dry-run',action='store_true',help='Check scene/profile layout only; no writes or inference')
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--model-root',type=Path,default=masker.DEFAULT_MODEL_ROOT)
    parser.set_defaults(box_thr=.25,text_thr=.25,max_frames=0,leica_guidance=False,exclude_prompt=None,point_fraction=None)
    args=parser.parse_args(argv)
    root=args.root.expanduser().resolve()
    jobs=plan(root,args.groups,args.profiles_root.expanduser().resolve())
    print(f'{len(jobs)} scenes selected for spatial mask correction.',flush=True)
    for scene,profile,prompt in jobs:
        print(f'{scene.name}: {profile.name}; label={prompt!r}',flush=True)
    if args.dry_run:return 0
    backend=None
    def segment(path,prompt):
        nonlocal backend
        if backend is None: backend=masker.load_segmenter(args)
        return backend(path,prompt)
    report=[]
    for scene,profile,prompt in jobs:
        args.input=scene;args.prompt_points=profile;args.prompt=prompt
        try:
            profile_data=json.loads(profile.read_text())
            if profile_data.get('robot_state_guidance'):
                robot_file=guidance_path(scene)
                if not robot_file.resolve().is_relative_to(scene):
                    raise ValueError(f'Robot guidance output escapes scene: {robot_file}')
                extent=profile_data.get('robot_offset_max_m',.15)
                payload=load_cached(scene,extent)
                if payload is None:payload=prepare(scene,args.parquet_python,extent)
                temporary=robot_file.with_suffix('.json.tmp')
                temporary.write_text(json.dumps(payload,indent=2)+'\n');temporary.replace(robot_file)
            result=masker.generate(args,segment)
            row=dict(scene=scene.name,status='failed' if result['empty'] else 'complete',empty=result['empty'],completed=result['completed'])
        except Exception as error:
            row=dict(scene=scene.name,status='failed',error=str(error))
            print(f'FAILED {scene.name}: {error}',file=sys.stderr,flush=True)
        report.append(row)
        output=root/'mask_correction_report.json'
        temporary=output.with_suffix('.json.tmp');temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(output)
    return int(any(row['status']=='failed' for row in report))


if __name__=='__main__':sys.exit(main())
