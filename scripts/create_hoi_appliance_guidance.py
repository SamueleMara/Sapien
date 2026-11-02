#!/usr/bin/env python3
"""Prepare the user-selected fridge appliance groups as SAM positive/negative references."""
import argparse
import json
from pathlib import Path

import numpy as np

from refine_hoi_geometry import read_leica, write_json


def create_guidance(scene,source_root):
    scene=Path(scene);conversion=json.loads((scene/'metadata/hoi_conversion.json').read_text())
    if conversion['location']!='fridge_1':raise ValueError('Fridge grouping is configured only for fridge_1')
    index=conversion['object_index']
    if index not in [1,2,3,4]:raise ValueError('No configured appliance group for object')
    positive=[1,2] if index<=2 else [3,4];negative=[3,4] if index<=2 else [1,2]
    base=Path(source_root)/conversion['location']/'leica/001/instance_annotations_3d/points'
    def sample(indices):
        points=np.concatenate([read_leica(base/f'{i:03d}.ply') for i in indices])
        _,unique=np.unique(np.floor(points/.015).astype(np.int64),axis=0,return_index=True)
        points=points[np.sort(unique)]
        return points[np.linspace(0,len(points)-1,min(4096,len(points))).astype(int)].tolist()
    write_json(scene/'metadata/leica_appliance_guidance.json',dict(
        coordinate_frame='hoi_leica_world',units='meters',positive_instance_ids=positive,negative_instance_ids=negative,
        positive_world_xyz=sample(positive),negative_world_xyz=sample(negative),
        grouping='User requested grey fridge for 001/002 and white fridge for 003/004; each pair belongs to one appliance',
        purpose='SAM appliance selection only; static scan points are not per-frame articulation or depth GT'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',nargs='+',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args()
    for scene in args.input:create_guidance(scene,args.source_root)
