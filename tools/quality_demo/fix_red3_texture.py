"""Recolor only the approved digit 3 in this specific quality texture atlas.

Append a lossless base-color PNG to the GLB; preserve all existing binary
payloads, mesh accessors, UVs, materials and metallic/roughness image bytes.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import io
import json
import struct
import time

import numpy as np
from PIL import Image
from scipy import ndimage
import trimesh

EXPECTED_SOURCE_SHA256 = 'd5aa1ac89fd99db20b3fa2bed3747f58af1dbb67c6acd340b1c12ed62223e8c8'


def read_glb(data):
    magic, version, total = struct.unpack_from('<4sII', data)
    assert magic == b'glTF' and version == 2 and total == len(data)
    chunks = {}
    offset = 12
    while offset < total:
        length, kind = struct.unpack_from('<II', data, offset)
        offset += 8
        chunks[kind] = data[offset:offset + length]
        offset += length
    assert offset == total
    return json.loads(chunks[0x4E4F534A]), chunks[0x004E4942]


def image_bytes(document, binary, image_index):
    view = document['bufferViews'][document['images'][image_index]['bufferView']]
    start = view.get('byteOffset', 0)
    return binary[start:start + view['byteLength']]


def main(root):
    started = time.perf_counter()
    source = root / 'shape_quality_textured.glb'
    output_path = root / 'shape_quality_textured_red3.glb'
    for suffix in ('.glb', '_basecolor.png', '_verification.json', '_comparison.png'):
        candidate = root / ('shape_quality_textured_red3' + suffix)
        if candidate.exists():
            raise FileExistsError(f'Refusing to overwrite {candidate}')
    assert not output_path.exists(), 'Refusing to overwrite the corrected model'
    original = source.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    if digest != EXPECTED_SOURCE_SHA256:
        raise ValueError('This sample-specific UV repair requires the original quality texture GLB')
    document, binary = read_glb(original)
    texture_index = document['materials'][0]['pbrMetallicRoughness']['baseColorTexture']['index']
    image_index = document['textures'][texture_index]['source']
    source_image = Image.open(io.BytesIO(image_bytes(document, binary, image_index))).convert('RGB')
    before = np.array(source_image)
    assert before.shape == (2048, 2048, 3)

    # The independently inspected upside-down digit 3 lies inside this ROI.
    x0, y0, x1, y1 = 1040, 450, 1240, 730
    roi = before[y0:y1, x0:x1]
    labels, _ = ndimage.label(roi.max(axis=2) < 150)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    component = labels == sizes.argmax()
    assert 20000 < int(component.sum()) < 27000
    ys, xs = np.where(component)
    bbox = [int(xs.min()+x0), int(ys.min()+y0), int(xs.max()+x0+1), int(ys.max()+y0+1)]
    assert 1050 <= bbox[0] <= 1070 and 1200 <= bbox[2] <= 1230
    assert 460 <= bbox[1] <= 480 and 700 <= bbox[3] <= 720
    mask = np.zeros(before.shape[:2], dtype=bool)
    mask[y0:y1, x0:x1] = component

    reference = np.array(Image.open(root/'demo.png').convert('RGB')).astype(np.int16)[40:140,245:315]
    red_mask = ((reference[:,:,0] > 100) & (reference[:,:,0] > 2*reference[:,:,1])
                & (reference[:,:,0] > 2*reference[:,:,2]))
    assert red_mask.sum() > 1000
    red = np.median(reference[red_mask], axis=0)
    # Retain antialiasing by blending the thin gray edge into the existing sign.
    level = roi.astype(float).max(axis=2)
    alpha = np.clip((170-level)/90, 0, 1)
    colored = np.rint(red[None,None,:]*alpha[:,:,None]
                      + roi*(1-alpha[:,:,None])).clip(0,255).astype(np.uint8)
    after = before.copy()
    after[y0:y1,x0:x1][component] = colored[component]
    assert np.array_equal(after[~mask], before[~mask])
    assert np.all(after[mask,0] > after[mask,1])
    png = io.BytesIO()
    Image.fromarray(after).save(png, format='PNG')
    payload = png.getvalue()
    png_path = root/'shape_quality_textured_red3_basecolor.png'
    assert not png_path.exists()
    png_path.write_bytes(payload)

    corrected = copy.deepcopy(document)
    offset = len(binary)
    assert offset % 4 == 0
    corrected['images'][image_index] = dict(corrected['images'][image_index],
        bufferView=len(corrected['bufferViews']), mimeType='image/png')
    corrected['bufferViews'].append({'buffer':0, 'byteOffset':offset, 'byteLength':len(payload)})
    updated_binary = binary + payload
    corrected['buffers'][0]['byteLength'] = len(updated_binary)
    updated_binary += b'\0' * (-len(updated_binary) % 4)
    json_chunk = json.dumps(corrected, separators=(',',':')).encode()
    json_chunk += b' ' * (-len(json_chunk) % 4)
    body = (struct.pack('<II',len(json_chunk),0x4E4F534A)+json_chunk
            +struct.pack('<II',len(updated_binary),0x004E4942)+updated_binary)
    result = struct.pack('<4sII',b'glTF',2,12+len(body))+body
    with output_path.open('xb') as output:
        output.write(result)

    actual, actual_binary = read_glb(output_path.read_bytes())
    decoded = np.array(Image.open(io.BytesIO(image_bytes(actual,actual_binary,image_index))).convert('RGB'))
    assert np.array_equal(decoded,after)
    assert np.array_equal(decoded[~mask],before[~mask])
    assert actual_binary[:len(binary)] == binary
    for key in ('accessors','meshes','materials','textures','nodes','scenes'):
        assert actual.get(key) == document.get(key), key
    for i in range(len(document['images'])):
        if i != image_index:
            assert image_bytes(actual,actual_binary,i)==image_bytes(document,binary,i)
    a = trimesh.load(source,force='mesh',process=False)
    b = trimesh.load(output_path,force='mesh',process=False)
    for name in ('vertices','faces'):
        assert np.array_equal(getattr(a,name),getattr(b,name)),name
    assert np.array_equal(a.visual.uv,b.visual.uv)
    assert np.array_equal(np.array(b.visual.material.baseColorTexture.convert('RGB')),after)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    report = {'source':str(source),'output':str(output_path),'size_bytes':output_path.stat().st_size,
        'source_sha256':digest,'output_sha256':hashlib.sha256(result).hexdigest(),
        'method':'deterministic Python local base-color recolor, user authorized',
        'reference_red_rgb':red.tolist(),'digit_bbox_xyxy':bbox,'changed_pixels':int(np.any(after!=before,axis=2).sum()),
        'outside_mask_pixels_identical':True,'original_binary_prefix_identical':True,
        'geometry_identical':True,'uv_identical':True,'other_images_identical':True,
        'source_unchanged':True,'texture_size':[2048,2048],'vertices':len(b.vertices),'faces':len(b.faces),
        'elapsed_s':round(time.perf_counter()-started,3)}
    (root/'shape_quality_textured_red3_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    # Inspection preview only: sign island rotated upright, before and after.
    left = source_image.crop((800,430,1650,780)).rotate(180)
    right = Image.fromarray(after).crop((800,430,1650,780)).rotate(180)
    preview=Image.new('RGB',(1700,350),'white')
    preview.paste(left,(0,0));preview.paste(right,(850,0))
    preview.save(root/'shape_quality_textured_red3_comparison.png')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-dir', required=True, type=Path,
                        help='Directory containing the original quality GLB and demo.png')
    args = parser.parse_args()
    main(args.asset_dir.resolve())
