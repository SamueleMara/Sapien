# Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import collections
import os
import struct
from typing import BinaryIO

import numpy as np


CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
Image = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_NAMES = {
    camera_model.model_name: camera_model for camera_model in CAMERA_MODELS
}


def _write_next_bytes(
    fid: BinaryIO,
    data: int
    | float
    | bytes
    | list[int | float | bytes]
    | tuple[int | float | bytes, ...],
    format_char_sequence: str,
    endian_character: str = "<",
) -> None:
    if isinstance(data, (list, tuple)):
        packed = struct.pack(endian_character + format_char_sequence, *data)
    else:
        packed = struct.pack(endian_character + format_char_sequence, data)
    fid.write(packed)


def write_cameras_text(cameras: dict[int, Camera], path: str) -> None:
    header = (
        "# Camera list with one line of data per camera:\n"
        f"#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"# Number of cameras: {len(cameras)}\n"
    )
    with open(path, "w", encoding="utf-8") as fid:
        fid.write(header)
        for _, cam in cameras.items():
            fid.write(
                " ".join(
                    str(elem)
                    for elem in [cam.id, cam.model, cam.width, cam.height, *cam.params]
                )
            )
            fid.write("\n")


def write_cameras_binary(cameras: dict[int, Camera], path: str) -> None:
    with open(path, "wb") as fid:
        _write_next_bytes(fid, len(cameras), "Q")
        for _, cam in cameras.items():
            model_id = CAMERA_MODEL_NAMES[cam.model].model_id
            _write_next_bytes(fid, [cam.id, model_id, cam.width, cam.height], "iiQQ")
            for param in cam.params:
                _write_next_bytes(fid, float(param), "d")


def write_images_text(images: dict[int, Image], path: str) -> None:
    if len(images) == 0:
        mean_observations = 0
    else:
        mean_observations = sum(
            len(img.point3D_ids) for _, img in images.items()
        ) / len(images)
    header = (
        "# Image list with two lines of data per image:\n"
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        f"# Number of images: {len(images)}, mean observations per image: {mean_observations}\n"
    )

    with open(path, "w", encoding="utf-8") as fid:
        fid.write(header)
        for _, img in images.items():
            image_header = [img.id, *img.qvec, *img.tvec, img.camera_id, img.name]
            fid.write(" ".join(map(str, image_header)) + "\n")

            points_strings = []
            for xy, point3D_id in zip(img.xys, img.point3D_ids):
                points_strings.append(" ".join(map(str, [*xy, point3D_id])))
            fid.write(" ".join(points_strings) + "\n")


def write_images_binary(images: dict[int, Image], path: str) -> None:
    with open(path, "wb") as fid:
        _write_next_bytes(fid, len(images), "Q")
        for _, img in images.items():
            _write_next_bytes(fid, img.id, "i")
            _write_next_bytes(fid, img.qvec.tolist(), "dddd")
            _write_next_bytes(fid, img.tvec.tolist(), "ddd")
            _write_next_bytes(fid, img.camera_id, "i")
            for char in img.name:
                _write_next_bytes(fid, char.encode("utf-8"), "c")
            _write_next_bytes(fid, b"\x00", "c")
            _write_next_bytes(fid, len(img.point3D_ids), "Q")
            for xy, point3D_id in zip(img.xys, img.point3D_ids):
                _write_next_bytes(fid, [*xy, point3D_id], "ddq")


def write_points3D_text(points3D: dict[int, Point3D], path: str) -> None:
    if len(points3D) == 0:
        mean_track_length = 0
    else:
        mean_track_length = sum(len(pt.image_ids) for _, pt in points3D.items()) / len(
            points3D
        )
    header = (
        "# 3D point list with one line of data per point:\n"
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        f"# Number of points: {len(points3D)}, mean track length: {mean_track_length}\n"
    )

    with open(path, "w", encoding="utf-8") as fid:
        fid.write(header)
        for _, pt in points3D.items():
            fid.write(" ".join(map(str, [pt.id, *pt.xyz, *pt.rgb, pt.error])) + " ")
            track_strings = []
            for image_id, point2D in zip(pt.image_ids, pt.point2D_idxs):
                track_strings.append(" ".join(map(str, [image_id, point2D])))
            fid.write(" ".join(track_strings) + "\n")


def write_points3D_binary(points3D: dict[int, Point3D], path: str) -> None:
    with open(path, "wb") as fid:
        _write_next_bytes(fid, len(points3D), "Q")
        for _, pt in points3D.items():
            _write_next_bytes(fid, pt.id, "Q")
            _write_next_bytes(fid, pt.xyz.tolist(), "ddd")
            _write_next_bytes(fid, pt.rgb.tolist(), "BBB")
            _write_next_bytes(fid, pt.error, "d")
            _write_next_bytes(fid, pt.image_ids.shape[0], "Q")
            for image_id, point2D_id in zip(pt.image_ids, pt.point2D_idxs):
                _write_next_bytes(fid, [image_id, point2D_id], "ii")


def write_model(
    cameras: dict[int, Camera],
    images: dict[int, Image],
    points3D: dict[int, Point3D],
    path: str,
    ext: str = ".bin",
) -> tuple[dict[int, Camera], dict[int, Image], dict[int, Point3D]]:
    if ext == ".txt":
        write_cameras_text(cameras, os.path.join(path, "cameras" + ext))
        write_images_text(images, os.path.join(path, "images" + ext))
        write_points3D_text(points3D, os.path.join(path, "points3D" + ext))
    elif ext == ".bin":
        write_cameras_binary(cameras, os.path.join(path, "cameras" + ext))
        write_images_binary(images, os.path.join(path, "images" + ext))
        write_points3D_binary(points3D, os.path.join(path, "points3D" + ext))
    else:
        raise ValueError("COLMAP model extension must be '.txt' or '.bin'")
    return cameras, images, points3D


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ],
        dtype=np.float64,
    )


def rotmat2qvec(rotation: np.ndarray) -> np.ndarray:
    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = rotation.flat
    key_matrix = (
        np.array(
            [
                [rxx - ryy - rzz, 0, 0, 0],
                [ryx + rxy, ryy - rxx - rzz, 0, 0],
                [rzx + rxz, rzy + ryz, rzz - rxx - ryy, 0],
                [ryz - rzy, rzx - rxz, rxy - ryx, rxx + ryy + rzz],
            ],
            dtype=np.float64,
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(key_matrix)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec
