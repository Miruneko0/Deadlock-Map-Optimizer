bl_info = {
    "name": "Map Crop",
    "version": (1, 6, 0),
    "blender": (5, 1, 0),
    "location": "3D Viewport > Sidebar (N) > Map Crop",
    "description": "Keeps only the needed part of a large map: hides the rest, or deletes it for good",
    "category": "Object",
}


import json
import math
import os
import re
import time

import bmesh
import bpy
import gpu
import numpy as np
from bpy.props import (BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty,
                       StringProperty)
from mathutils import Euler, Matrix, Vector

ZONE_PROP = "mapcrop_zone"
SRC_PROP = "mapcrop_src"
COPY_PROP = "mapcrop_copy"
LIGHT_PROP = "mapcrop_light"
ROLE_PROP = "mapcrop_role"
ROOT_TOKEN = "<scene>"
COLL_NAMES = {"KEEP": "MAP_KEEP", "FAR": "MAP_FAR", "OFF": "MAP_OFF", "LIGHT": "MAP_LIGHT"}

NORMAL_ATTR = "mapcrop_normal"

TRIM_MIN_DROP = 2000
TRIM_MAX_INSIDE = 0.8

CUT_SNAP = 1e-4

STILL_CAMERA = 1e-4
VIEW_LIMIT = math.radians(89.9)
CHUNK = 1 << 20

ID_PIXELS = 1 << 23
CHUNK_TRIS = 256
TEST_TRIS = 1 << 21
HZB_EPS = 1e-6
SPLIT_PIXELS = 16
SPLIT_LEVELS = 7
LIGHT_PIXELS = 2048
LAMP_PIXELS = 1024
LAMP_NEAR = 0.05
MAX_LAMPS = 16
SUN_CONTRAST = 20.0
SKY_ELEVATION = math.radians(15.0)
SEE_THROUGH_NODES = {'ShaderNodeBsdfTransparent', 'ShaderNodeBsdfGlass', 'ShaderNodeBsdfRefraction'}
CULLING_ENGINES = {'BLENDER_EEVEE', 'BLENDER_EEVEE_NEXT', 'BLENDER_WORKBENCH'}

OUTSIDE, INSIDE, PARTIAL = 0, 1, 2

CULL_TYPES = {'MESH', 'CURVE', 'CURVES', 'SURFACE', 'META', 'FONT', 'POINTCLOUD', 'VOLUME'}

IGNORED_USERS = (bpy.types.Collection, bpy.types.Scene, bpy.types.WindowManager,
                 bpy.types.Screen, bpy.types.WorkSpace)


def tri_count(me):
    return max(len(me.loops) - 2 * len(me.polygons), 0)


def world_matrix(ob, cache):
    key = ob.as_pointer()
    m = cache.get(key)
    if m is None:
        m = ob.matrix_basis.copy()
        if ob.parent is not None:
            m = world_matrix(ob.parent, cache) @ ob.matrix_parent_inverse @ m
        cache[key] = m
    return m


def zone_boxes(zones, margin, full_height):
    out = []
    for mw, half in zones:
        scale = [mw.col[i].to_3d().length for i in range(3)]
        if min(scale) < 1e-9 or half <= 0.0:
            continue
        inv = np.array(mw.inverted(), dtype=np.float64)
        ext = np.array([half + margin / s for s in scale], dtype=np.float64)
        bmin, bmax = -ext, ext.copy()
        if full_height:
            bmin[2], bmax[2] = -np.inf, np.inf
        out.append((inv, bmin, bmax))
    return out


def read_positions(me):
    nv = len(me.vertices)
    co = np.empty(nv * 3, dtype=np.float32)
    attr = me.attributes.get("position")
    if attr is not None and len(attr.data) == nv:
        attr.data.foreach_get("vector", co)
    else:
        me.vertices.foreach_get("co", co)
    return co.reshape(nv, 3)


def read_corner_verts(me):
    n_loops = len(me.loops)
    out = np.empty(n_loops, dtype=np.int32)
    attr = me.attributes.get(".corner_vert")
    if attr is not None and len(attr.data) == n_loops:
        attr.data.foreach_get("value", out)
    else:
        me.loops.foreach_get("vertex_index", out)
    return out


def poly_starts(me):
    npoly = len(me.polygons)
    if len(me.loops) == 3 * npoly:
        return None
    starts = np.empty(npoly, dtype=np.int32)
    me.polygons.foreach_get("loop_start", starts)
    return starts


def classify_object(ob, mw, boxes, exact=False):
    me = ob.data
    if len(me.vertices) == 0:
        return OUTSIDE, None, None
    co = read_positions(me)
    lo = [float(co[:, j].min()) for j in range(3)]
    hi = [float(co[:, j].max()) for j in range(3)]
    aabb = np.array([(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])],
                    dtype=np.float64)
    obj_m = np.array(mw, dtype=np.float64)
    npoly = len(me.polygons)
    corner_vert = starts = touch = inside = None
    cutters = []
    for inv, bmin, bmax in boxes:
        m = inv @ obj_m
        c = aabb @ m[:3, :3].T + m[:3, 3]
        if np.any(c.max(axis=0) < bmin - 1e-3) or np.any(c.min(axis=0) > bmax + 1e-3):
            continue
        loc = co @ m[:3, :3].T.astype(np.float32) + m[:3, 3].astype(np.float32)
        codes = np.packbits(np.concatenate((loc >= bmin, loc <= bmax), axis=1),
                            axis=1, bitorder="little").ravel()
        if np.bitwise_or.reduce(codes) != 63:
            continue
        all_bits = int(np.bitwise_and.reduce(codes))
        if npoly == 0 or all_bits == 63:
            return INSIDE, None, None
        if corner_vert is None:
            corner_vert = read_corner_verts(me)
            starts = poly_starts(me)
        poly_codes = codes[corner_vert]
        if starts is None:
            t = poly_codes.reshape(npoly, 3)
            hit = (t[:, 0] | t[:, 1] | t[:, 2]) == 63
            full = (t[:, 0] & t[:, 1] & t[:, 2]) == 63 if exact else None
        else:
            hit = np.bitwise_or.reduceat(poly_codes, starts) == 63
            full = np.bitwise_and.reduceat(poly_codes, starts) == 63 if exact else None
        if not hit.any():
            continue
        touch = hit if touch is None else (touch | hit)
        if exact:
            inside = full if inside is None else (inside | full)
            cutters.append((m, bmin, bmax, all_bits))
    if touch is None:
        return OUTSIDE, None, None
    if not exact:
        return (INSIDE, None, None) if touch.all() else (PARTIAL, touch, None)
    border = touch & ~inside
    if touch.all() and not border.any():
        return INSIDE, None, None
    scale = float(np.linalg.norm(obj_m[:3, :3], axis=0).max())
    return PARTIAL, touch, (border, cutters, CUT_SNAP / max(scale, 1e-12))


def camera_view(cam_ob, depsgraph, scene):
    ob = cam_ob.evaluated_get(depsgraph)
    cam = ob.data
    loc, rot, _scale = ob.matrix_world.decompose()
    corners = cam.view_frame(scene=scene)
    xs = [v.x for v in corners]
    ys = [v.y for v in corners]
    if cam.type == 'PERSP':
        depth = -corners[0].z
        frame = (min(xs) / depth, max(xs) / depth, min(ys) / depth, max(ys) / depth)
    else:
        frame = (min(xs), max(xs), min(ys), max(ys))
    return (np.array(rot.to_matrix(), dtype=np.float64), np.array(loc, dtype=np.float64),
            cam.type, frame, cam.clip_start, cam.clip_end)


def view_planes(view, fov_margin, margin):
    rot, loc, kind, (left, right, bottom, top), near, far = view
    if kind == 'PERSP':
        half = fov_margin / 2
        a_l = max(math.atan(left) - half, -VIEW_LIMIT)
        a_r = min(math.atan(right) + half, VIEW_LIMIT)
        a_b = max(math.atan(bottom) - half, -VIEW_LIMIT)
        a_t = min(math.atan(top) + half, VIEW_LIMIT)
        normals = [(-math.cos(a_l), 0.0, -math.sin(a_l)), (math.cos(a_r), 0.0, math.sin(a_r)),
                   (0.0, -math.cos(a_b), -math.sin(a_b)), (0.0, math.cos(a_t), math.sin(a_t))]
        offsets = [margin] * 4
    else:
        normals = [(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 1.0, 0.0)]
        offsets = [margin - left, margin + right, margin - bottom, margin + top]
    normals += [(0.0, 0.0, 1.0), (0.0, 0.0, -1.0)]
    offsets += [margin - near, margin + far]
    normals = np.array(normals) @ rot.T
    return normals, np.array(offsets) + normals @ loc


def view_corners(view):
    rot, loc, kind, (left, right, bottom, top), near, far = view
    points = []
    for dist in (near, far):
        scale = dist if kind == 'PERSP' else 1.0
        points += [(x * scale, y * scale, -dist) for x in (left, right) for y in (bottom, top)]
    return np.array(points) @ rot.T + loc


def camera_zone(views, fov_margin, margin):
    corners = [view_corners(v) for v in views]
    still = all(np.abs(c - corners[0]).max() <= STILL_CAMERA for c in corners)
    if still:
        views, corners = views[:1], corners[:1]
    taken, planes = [], []
    for view, c in zip(views, corners):
        if taken and np.linalg.norm(np.array(taken) - c, axis=2).max(axis=1).min() <= margin / 2:
            continue
        taken.append(c)
        planes.append(view_planes(view, fov_margin, margin))
    return {"normals": np.array([n for n, _ in planes]), "offsets": np.array([o for _, o in planes]),
            "exact": still}


def classify_view(ob, mw, camera):
    me = ob.data
    if len(me.vertices) == 0:
        return OUTSIDE, None, None
    exact = camera["exact"]
    co = read_positions(me)
    lo = co.min(axis=0).astype(np.float64)
    hi = co.max(axis=0).astype(np.float64)
    aabb = np.array([(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    obj_m = np.array(mw, dtype=np.float64)
    rot, loc = obj_m[:3, :3], obj_m[:3, 3]
    normals, offsets = camera["normals"], camera["offsets"]
    d = np.einsum("fpj,kj->fpk", normals, aabb @ rot.T + loc) - offsets[:, :, None]
    near = ~(d > 1e-3).all(axis=2).any(axis=1)
    if not near.any():
        return OUTSIDE, None, None
    if (d[near] <= 0.0).all(axis=(1, 2)).any():
        return INSIDE, None, None
    obj_n = normals[near] @ rot
    obj_c = offsets[near] - normals[near] @ loc
    nv, npoly = len(co), len(me.polygons)
    corner_vert = starts = touch = inside = None
    cut_bits = 0
    step = max(1, CHUNK // max(nv, len(me.loops), 1))
    for k in range(0, len(obj_n), step):
        part_n, part_c = obj_n[k:k + step], obj_c[k:k + step]
        dist = co @ part_n.reshape(-1, 3).T.astype(np.float32)
        codes = np.packbits((dist <= part_c.reshape(-1).astype(np.float32)).reshape(nv, len(part_n), 6),
                            axis=2, bitorder="little")[:, :, 0]
        reach = np.bitwise_or.reduce(codes, axis=0) == 63
        if not reach.any():
            continue
        codes = codes[:, reach]
        all_bits = np.bitwise_and.reduce(codes, axis=0)
        if npoly == 0 or (all_bits == 63).any():
            return INSIDE, None, None
        if corner_vert is None:
            corner_vert = read_corner_verts(me)
            starts = poly_starts(me)
        poly_codes = codes[corner_vert]
        if starts is None:
            t = poly_codes.reshape(npoly, 3, -1)
            hit = ((t[:, 0] | t[:, 1] | t[:, 2]) == 63).any(axis=1)
            full = ((t[:, 0] & t[:, 1] & t[:, 2]) == 63).any(axis=1) if exact else None
        else:
            hit = (np.bitwise_or.reduceat(poly_codes, starts, axis=0) == 63).any(axis=1)
            full = (np.bitwise_and.reduceat(poly_codes, starts, axis=0) == 63).any(axis=1) if exact else None
        touch = hit if touch is None else (touch | hit)
        if exact:
            inside, cut_bits = full, int(all_bits[0])
        elif touch.all():
            return INSIDE, None, None
    if touch is None or not touch.any():
        return OUTSIDE, None, None
    if not exact:
        return (INSIDE, None, None) if touch.all() else (PARTIAL, touch, None)
    border = touch & ~inside
    if touch.all() and not border.any():
        return INSIDE, None, None
    planes = []
    for j in range(6):
        if not (cut_bits >> j) & 1:
            n = obj_n[0, j]
            length2 = float(n @ n)
            planes.append((Vector(n * (obj_c[0, j] / length2)), Vector(n / math.sqrt(length2))))
    scale = float(np.linalg.norm(rot, axis=0).max())
    return PARTIAL, touch, (border, planes, CUT_SNAP / max(scale, 1e-12))


def distinct_views(views):
    out, last = [], None
    for view in views:
        corners = view_corners(view)
        if last is None or np.abs(corners - last).max() > STILL_CAMERA:
            out.append(view)
            last = corners
    return out


def see_through(mat):
    if mat is None:
        return False
    if getattr(mat, "surface_render_method", None) == 'BLENDED':
        return True
    return mat.node_tree is not None and tree_see_through(mat.node_tree, set())


def tree_see_through(tree, done):
    done.add(tree.as_pointer())
    for node in tree.nodes:
        if node.bl_idname in SEE_THROUGH_NODES:
            return True
        if node.bl_idname == 'ShaderNodeBsdfPrincipled':
            alpha = node.inputs.get("Alpha")
            weight = node.inputs.get("Transmission Weight")
            if alpha is not None and (alpha.is_linked or alpha.default_value < 1.0):
                return True
            if weight is not None and (weight.is_linked or weight.default_value > 0.0):
                return True
        sub = getattr(node, "node_tree", None)
        if isinstance(sub, bpy.types.NodeTree) and sub.as_pointer() not in done and tree_see_through(sub, done):
            return True
    return False


def view_matrices(rot, loc, kind, l2, r2, b2, t2, n, f):
    if kind == 'PERSP':
        L, R, B, T = l2 * n, r2 * n, b2 * n, t2 * n
        proj = [[2 * n / (R - L), 0.0, (R + L) / (R - L), 0.0],
                [0.0, 2 * n / (T - B), (T + B) / (T - B), 0.0],
                [0.0, 0.0, -(f + n) / (f - n), -2 * f * n / (f - n)],
                [0.0, 0.0, -1.0, 0.0]]
    else:
        proj = [[2 / (r2 - l2), 0.0, 0.0, -(r2 + l2) / (r2 - l2)],
                [0.0, 2 / (t2 - b2), 0.0, -(t2 + b2) / (t2 - b2)],
                [0.0, 0.0, -2 / (f - n), -(f + n) / (f - n)],
                [0.0, 0.0, 0.0, 1.0]]
    world_to_camera = np.eye(4)
    world_to_camera[:3, :3] = rot.T
    world_to_camera[:3, 3] = -rot.T @ loc
    axis = rot[:, 2]
    depth_row = np.append(-axis, axis @ loc - n) / (f - n)
    return np.array(proj) @ world_to_camera, depth_row


def view_buffer(view, fov_margin, scene, detail, limit):
    rot, loc, kind, (left, right, bottom, top), near, far = view
    r = scene.render
    width = r.resolution_x * r.resolution_percentage / 100
    height = r.resolution_y * r.resolution_percentage / 100
    if kind == 'PERSP':
        half = fov_margin / 2
        l2 = math.tan(max(math.atan(left) - half, -VIEW_LIMIT))
        r2 = math.tan(min(math.atan(right) + half, VIEW_LIMIT))
        b2 = math.tan(max(math.atan(bottom) - half, -VIEW_LIMIT))
        t2 = math.tan(min(math.atan(top) + half, VIEW_LIMIT))
    else:
        l2, r2, b2, t2 = left, right, bottom, top
    w = (r2 - l2) * width * detail / (right - left)
    h = (t2 - b2) * height * detail / (top - bottom)
    shrink = min(1.0, limit / w, limit / h, math.sqrt(ID_PIXELS / (w * h)))
    w, h = max(1, math.ceil(w * shrink)), max(1, math.ceil(h * shrink))
    matrix, depth_row = view_matrices(rot, loc, kind, l2, r2, b2, t2, near, far)
    return matrix, depth_row, w, h


def depth_shader():
    iface = gpu.types.GPUStageInterfaceInfo("mapcrop_depth")
    iface.smooth('FLOAT', 'lin_depth')
    info = gpu.types.GPUShaderCreateInfo()
    info.push_constant('MAT4', 'view_proj')
    info.push_constant('VEC4', 'depth_row')
    info.vertex_in(0, 'VEC3', 'pos')
    info.vertex_out(iface)
    info.fragment_out(0, 'FLOAT', 'out_depth')
    info.vertex_source("void main() { lin_depth = dot(depth_row, vec4(pos, 1.0)); "
                       "gl_Position = view_proj * vec4(pos, 1.0); }")
    info.fragment_source("void main() { out_depth = lin_depth; }")
    return gpu.shader.create_from_info(info)


def max_pyramid(depth, pad=0.0):
    levels = [depth]
    a = depth
    while max(a.shape) > 1:
        if a.shape[0] % 2:
            a = np.vstack([a, np.full((1, a.shape[1]), pad, a.dtype)])
        if a.shape[1] % 2:
            a = np.hstack([a, np.full((a.shape[0], 1), pad, a.dtype)])
        a = np.maximum(np.maximum(a[0::2, 0::2], a[0::2, 1::2]), np.maximum(a[1::2, 0::2], a[1::2, 1::2]))
        levels.append(a)
    return levels


def may_be_seen(levels, sx, sy, near):
    h, w = levels[0].shape
    x0, x1 = np.clip(sx[:, 0], 0, w - 1), np.clip(sx[:, 1], 0, w - 1)
    y0, y1 = np.clip(sy[:, 0], 0, h - 1), np.clip(sy[:, 1], 0, h - 1)
    size = np.maximum(x1 - x0, y1 - y0) + 1
    level = np.maximum(np.ceil(np.log2(size)).astype(np.int64) - 1, 0)
    seen = np.zeros(len(near), dtype=bool)
    for lv in np.unique(level):
        pick = np.flatnonzero(level == lv)
        grid = levels[min(int(lv), len(levels) - 1)]
        tx0, tx1, ty0, ty1 = x0[pick] >> lv, x1[pick] >> lv, y0[pick] >> lv, y1[pick] >> lv
        far = np.full(len(pick), -np.inf, dtype=np.float32)
        for dx in range(3):
            for dy in range(3):
                far = np.maximum(far, grid[np.minimum(ty0 + dy, ty1), np.minimum(tx0 + dx, tx1)])
        seen[pick] = near[pick] <= far + HZB_EPS
    return seen


def screen_boxes(clip, depth, w, h):
    nearest = depth.min(axis=1)
    state = np.ones(len(depth), dtype=np.int8)
    wc = np.where(clip[:, :, 3] > 1e-9, clip[:, :, 3], 1.0)
    px = (clip[:, :, 0] / wc * 0.5 + 0.5) * w
    py = (clip[:, :, 1] / wc * 0.5 + 0.5) * h
    sx = np.floor(np.stack([px.min(axis=1), px.max(axis=1)], axis=1)).astype(np.int64)
    sy = np.floor(np.stack([py.min(axis=1), py.max(axis=1)], axis=1)).astype(np.int64)
    off = (sx[:, 1] < 0) | (sx[:, 0] >= w) | (sy[:, 1] < 0) | (sy[:, 0] >= h) | (nearest > 1.0)
    state[off] = 0
    state[nearest < 0.0] = 2
    state[depth.max(axis=1) < 0.0] = 0
    return sx, sy, np.maximum(nearest, 0.0), state


def pieces_seen(levels, clip, depth, w, h):
    owner = np.arange(len(clip))
    seen = np.zeros(len(clip), dtype=bool)
    for level in range(SPLIT_LEVELS + 1):
        if level:
            m01, m12, m20 = (clip[:, 0] + clip[:, 1]) / 2, (clip[:, 1] + clip[:, 2]) / 2, (clip[:, 2] + clip[:, 0]) / 2
            d01, d12, d20 = (depth[:, 0] + depth[:, 1]) / 2, (depth[:, 1] + depth[:, 2]) / 2, (depth[:, 2] + depth[:, 0]) / 2
            clip = np.concatenate([np.stack(c, axis=1) for c in ((clip[:, 0], m01, m20), (m01, clip[:, 1], m12),
                                                                    (m20, m12, clip[:, 2]), (m01, m12, m20))])
            depth = np.concatenate([np.stack(c, axis=1) for c in ((depth[:, 0], d01, d20), (d01, depth[:, 1], d12),
                                                                     (d20, d12, depth[:, 2]), (d01, d12, d20))])
            owner = np.tile(owner, 4)
        sx, sy, near, state = screen_boxes(clip, depth, w, h)
        passing = state == 2
        test = np.flatnonzero(state == 1)
        passing[test] = may_be_seen(levels, sx[test], sy[test], near[test])
        big = (np.maximum(sx[:, 1] - sx[:, 0], sy[:, 1] - sy[:, 0]) + 1 > SPLIT_PIXELS) & (state == 1)
        done = passing & (~big | (level == SPLIT_LEVELS))
        seen[owner[done]] = True
        go = passing & ~done & ~seen[owner]
        if not go.any():
            break
        clip, depth, owner = clip[go], depth[go], owner[go]
    return seen


def material_kinds(ob, culling):
    return [(see_through(s.material), culling and s.material is not None and s.material.use_backface_culling,
             glows(s.material)) for s in ob.material_slots] or [(False, False, False)]


def triangle_set(objs, culling):
    cache = {}
    parts = {k: [] for k in ("points", "tris", "poly", "obj", "through", "one_sided", "glow")}
    spans = []
    nv = npoly = 0
    for k, ob in enumerate(objs):
        me = ob.data
        m = np.array(world_matrix(ob, cache), dtype=np.float64)
        world = read_positions(me) @ m[:3, :3].T.astype(np.float32) + m[:3, 3].astype(np.float32)
        tris = np.empty(len(me.loop_triangles) * 3, dtype=np.int32)
        me.loop_triangles.foreach_get("vertices", tris)
        tris = tris.reshape(-1, 3)
        tpoly = np.empty(len(tris), dtype=np.int32)
        me.loop_triangles.foreach_get("polygon_index", tpoly)
        if np.linalg.det(m[:3, :3]) < 0.0:
            tris = tris[:, ::-1]
        n = len(me.polygons)
        kinds = np.array(material_kinds(ob, culling), dtype=bool)
        slot = np.zeros(n, dtype=np.int32)
        attr = me.attributes.get("material_index")
        if len(kinds) > 1 and attr is not None and attr.domain == 'FACE' and len(attr.data) == n:
            attr.data.foreach_get("value", slot)
            slot = np.clip(slot, 0, len(kinds) - 1)
        parts["points"].append(world)
        parts["tris"].append(tris + nv)
        parts["poly"].append(tpoly + npoly)
        parts["obj"].append(np.full(len(tris), k, dtype=np.int32))
        parts["through"].append(kinds[slot[tpoly], 0])
        parts["one_sided"].append(kinds[slot[tpoly], 1])
        parts["glow"].append(kinds[slot, 2])
        spans.append((npoly, n))
        nv += len(world)
        npoly += n
    tset = {key: np.concatenate(value) for key, value in parts.items()}
    tset["points"] = np.hstack([tset["points"], np.ones((nv, 1), dtype=np.float32)])
    tset["spans"], tset["npoly"] = spans, npoly
    tris, pts = tset["tris"], tset["points"][:, :3]
    starts = np.arange(0, len(tris), CHUNK_TRIS)
    lo = np.empty((len(starts), 3), dtype=np.float32)
    hi = np.empty((len(starts), 3), dtype=np.float32)
    step = CHUNK_TRIS * 4096
    for a in range(0, len(tris), step):
        t = tris[a:a + step]
        p0, p1, p2 = pts[t[:, 0]], pts[t[:, 1]], pts[t[:, 2]]
        local = np.arange(0, len(t), CHUNK_TRIS)
        runs = slice(a // CHUNK_TRIS, a // CHUNK_TRIS + len(local))
        lo[runs] = np.minimum.reduceat(np.minimum(np.minimum(p0, p1), p2), local)
        hi[runs] = np.maximum.reduceat(np.maximum(np.maximum(p0, p1), p2), local)
    one = np.ones(len(starts), dtype=np.float32)
    tset["boxes"] = np.stack([np.stack([hi[:, 0] if c & 1 else lo[:, 0], hi[:, 1] if c & 2 else lo[:, 1],
                                        hi[:, 2] if c & 4 else lo[:, 2], one], axis=1) for c in range(8)], axis=1)
    tset["run_of"] = np.repeat(np.arange(len(starts)), np.diff(np.append(starts, len(tris))))
    return tset


def gpu_start():
    if bpy.app.background:
        try:
            gpu.platform.backend_type_get()
        except SystemError:
            gpu.init()


def sight(tset, frames, draw, pending, farthest=False, progress=None):
    hit = np.zeros(tset["npoly"], dtype=bool)
    pending = pending.copy()
    if not frames or not pending.any():
        return hit
    gpu_start()
    points, tris, poly = tset["points"], tset["tris"], tset["poly"]
    if "vbo" not in tset:
        fmt = gpu.types.GPUVertFormat()
        fmt.attr_add(id="pos", comp_type='F32', len=3, fetch_mode='FLOAT')
        vbo = gpu.types.GPUVertBuf(fmt, len(points))
        vbo.attr_fill("pos", np.ascontiguousarray(points[:, :3]))
        tset["vbo"] = vbo
    batches = []
    for culled in (False, True):
        sel = draw & (tset["one_sided"] == culled) if not farthest else (draw if not culled else None)
        if sel is not None and sel.any():
            ibo = gpu.types.GPUIndexBuf(type='TRIS', seq=tris[sel].astype(np.uint32))
            batches.append((culled, gpu.types.GPUBatch(type='TRIS', buf=tset["vbo"], elem=ibo)))
    width = max(f[2] for f in frames)
    height = max(f[3] for f in frames)
    color = gpu.types.GPUTexture((width, height), format='R32F')
    depth = gpu.types.GPUTexture((width, height), format='DEPTH_COMPONENT32F')
    fb = gpu.types.GPUFrameBuffer(depth_slot=depth, color_slots=(color,))
    shader = depth_shader()
    background, pad = (-1.0, -1.0) if farthest else (2.0, 0.0)
    try:
        for i, (matrix, depth_row, w, h) in enumerate(frames):
            if progress is not None:
                progress(i)
            with fb.bind():
                gpu.state.viewport_set(0, 0, w, h)
                color.clear(format='FLOAT', value=(background,))
                depth.clear(format='FLOAT', value=(0.0 if farthest else 1.0,))
                gpu.state.depth_test_set('GREATER_EQUAL' if farthest else 'LESS_EQUAL')
                gpu.state.depth_mask_set(True)
                shader.bind()
                shader.uniform_float("view_proj", Matrix(matrix.tolist()))
                shader.uniform_float("depth_row", depth_row.tolist())
                for culled, batch in batches:
                    gpu.state.face_culling_set('BACK' if culled else 'NONE')
                    batch.draw(shader)
                buffer = np.frombuffer(fb.read_color(0, 0, w, h, 1, 0, 'FLOAT'), dtype=np.float32).reshape(h, w)
            levels = max_pyramid(buffer, pad)
            m32, d32 = matrix.astype(np.float32), depth_row.astype(np.float32)
            boxes = tset["boxes"]
            sx, sy, near, state = screen_boxes(boxes @ m32.T, boxes @ d32, w, h)
            open_runs = state == 2
            test = np.flatnonzero(state == 1)
            open_runs[test] = may_be_seen(levels, sx[test], sy[test], near[test])
            todo = np.flatnonzero(pending & open_runs[tset["run_of"]])
            if len(todo):
                clip, lin = points @ m32.T, points @ d32
                for a in range(0, len(todo), TEST_TRIS):
                    part = todo[a:a + TEST_TRIS]
                    t = tris[part]
                    sx, sy, near, state = screen_boxes(clip[t], lin[t], w, h)
                    seen = state == 2
                    test = np.flatnonzero(state == 1)
                    seen[test] = may_be_seen(levels, sx[test], sy[test], near[test])
                    big = np.flatnonzero(seen & (state == 1) & (
                        np.maximum(sx[:, 1] - sx[:, 0], sy[:, 1] - sy[:, 0]) + 1 > SPLIT_PIXELS))
                    if len(big):
                        seen[big] = pieces_seen(levels, clip[t[big]], lin[t[big]], w, h)
                    hit[poly[part[seen]]] = True
                pending &= ~hit[poly]
            if not pending.any():
                break
    finally:
        gpu.state.face_culling_set('NONE')
        gpu.state.depth_mask_set(False)
        gpu.state.depth_test_set('NONE')
    return hit


def glows(mat):
    return mat is not None and mat.node_tree is not None and tree_glows(mat.node_tree, set())


def tree_glows(tree, done):
    done.add(tree.as_pointer())
    for node in tree.nodes:
        if node.bl_idname == 'ShaderNodeEmission':
            return True
        if node.bl_idname == 'ShaderNodeBsdfPrincipled':
            strength = node.inputs.get("Emission Strength")
            color = node.inputs.get("Emission Color")
            if (strength is not None and color is not None and (strength.is_linked or strength.default_value > 0.0)
                    and (color.is_linked or max(color.default_value[:3]) > 0.0)):
                return True
        sub = getattr(node, "node_tree", None)
        if isinstance(sub, bpy.types.NodeTree) and sub.as_pointer() not in done and tree_glows(sub, done):
            return True
    return False


def cell_keys(points, size):
    c = np.floor(points / size).astype(np.int64) + (1 << 20)
    return (c[:, 0] << 42) | (c[:, 1] << 21) | c[:, 2]


def around(keys):
    steps = np.array([(dx << 42) + (dy << 21) + dz for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
                     dtype=np.int64)
    return np.unique((np.unique(keys)[:, None] + steps[None, :]).ravel())


def triangle_samples(points, tris, spacing):
    if not len(tris):
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.int64)
    a, b, c = points[tris[:, 0]], points[tris[:, 1]], points[tris[:, 2]]
    edge = np.maximum(np.maximum(np.linalg.norm(a - b, axis=1), np.linalg.norm(b - c, axis=1)),
                      np.linalg.norm(c - a, axis=1))
    k = np.clip(np.ceil(edge / spacing), 1, 32).astype(np.int64)
    out, owner = [np.zeros((0, 3), dtype=np.float32)], [np.zeros(0, dtype=np.int64)]
    for kk in np.unique(k[k > 1]):
        sel = np.flatnonzero(k == kk)
        i, j = np.meshgrid(np.arange(kk + 1), np.arange(kk + 1), indexing="ij")
        inner = i + j <= kk
        wa, wb = (i[inner] / kk).astype(np.float32), (j[inner] / kk).astype(np.float32)
        wc = 1.0 - wa - wb
        pts = a[sel, None] * wa[None, :, None] + b[sel, None] * wb[None, :, None] + c[sel, None] * wc[None, :, None]
        out.append(pts.reshape(-1, 3))
        owner.append(np.repeat(sel, len(wa)))
    return np.concatenate(out), np.concatenate(owner)


def polygons_near(tset, cells, size, tri_mask):
    pts, tris = tset["points"][:, :3], tset["tris"]
    out = np.zeros(tset["npoly"], dtype=bool)
    if not len(cells):
        return out
    vert = np.isin(cell_keys(pts, size), cells)
    idx = np.flatnonzero(tri_mask)
    close = vert[tris[idx]].any(axis=1)
    samples, owner = triangle_samples(pts, tris[idx[~close]], size)
    if len(samples):
        close[np.flatnonzero(~close)[owner[np.isin(cell_keys(samples, size), cells)]]] = True
    out[tset["poly"][idx[close]]] = True
    return out


def receiver_cells(tset, receivers, size):
    pts, tris = tset["points"][:, :3], tset["tris"][receivers]
    samples, _owner = triangle_samples(pts, tris, size)
    return around(np.concatenate([cell_keys(pts[np.unique(tris.ravel())], size), cell_keys(samples, size)]))


def environment_luminance(image, width=512):
    w, h = image.size
    if not w or not h:
        return None
    if w * h <= 1 << 24:
        px = np.empty(w * h * 4, dtype=np.float32)
        image.pixels.foreach_get(px)
        px = px.reshape(h, w, 4)
    else:
        rows = range(0, h, math.ceil(h / (width // 2)))
        px = np.array([image.pixels[r * w * 4:(r + 1) * w * 4] for r in rows], dtype=np.float32).reshape(-1, w, 4)
    lum = px[:, :, 0] * 0.2126 + px[:, :, 1] * 0.7152 + px[:, :, 2] * 0.0722
    fx = max(1, lum.shape[1] // width)
    fy = max(1, lum.shape[0] // (width // 2))
    hh, ww = lum.shape[0] // fy * fy, lum.shape[1] // fx * fx
    return lum[:hh, :ww].reshape(hh // fy, fy, ww // fx, fx).max(axis=(1, 3))


def environment_to_world(env):
    link = env.inputs["Vector"].links[0] if env.inputs["Vector"].is_linked else None
    node = link.from_node if link is not None else None
    if node is None or node.bl_idname != 'ShaderNodeMapping' or node.inputs["Rotation"].is_linked:
        return Matrix.Identity(3)
    rot = Euler(node.inputs["Rotation"].default_value).to_matrix()
    return rot if node.vector_type == 'TEXTURE' else rot.transposed()


def hdri_suns(scene):
    world = scene.world
    if world is None or world.node_tree is None:
        return []
    suns = []
    for env in world.node_tree.nodes:
        if env.bl_idname != 'ShaderNodeTexEnvironment' or env.image is None or env.mute:
            continue
        lum = environment_luminance(env.image)
        if lum is None or not lum.size:
            continue
        h, w = lum.shape
        u = (np.arange(w) + 0.5) / w
        v = (np.arange(h) + 0.5) / h
        phi = (0.5 - u)[None, :] * 2 * math.pi
        lat = (v - 0.5)[:, None] * math.pi
        dirs = np.stack([np.cos(lat) * np.cos(phi), np.cos(lat) * np.sin(phi), np.sin(lat) * np.ones_like(phi)], axis=-1)
        to_world = np.array(environment_to_world(env))
        median = float(np.median(lum))
        work = lum.copy()
        for _ in range(2):
            row, col = np.unravel_index(int(np.argmax(work)), work.shape)
            peak = float(work[row, col])
            if peak <= max(SUN_CONTRAST * median, 1e-6):
                break
            d = dirs[row, col]
            bright = (lum >= peak * 0.5) & ((dirs @ d) > math.cos(math.radians(20.0)))
            solid = float((np.cos(lat) * np.ones_like(phi))[bright].sum()) * (2 * math.pi / w) * (math.pi / h)
            radius = min(max(math.sqrt(solid / math.pi), math.radians(0.5)), math.radians(5.0))
            suns.append((to_world @ d, radius))
            work[(dirs @ d) > math.cos(math.radians(20.0))] = 0.0
    return suns


def sky_directions(count):
    if count <= 0:
        return np.zeros((0, 3))
    z0 = math.sin(SKY_ELEVATION)
    k = np.arange(count)
    z = 1.0 - (1.0 - z0) * (k + 0.5) / count
    phi = k * math.pi * (3.0 - math.sqrt(5.0))
    r = np.sqrt(1.0 - z * z)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def disk_directions(d, radius):
    d = np.asarray(d, dtype=np.float64)
    d = d / np.linalg.norm(d)
    rot = np.array(Vector(d).to_track_quat('Z', 'Y').to_matrix())
    out = [d]
    for k in range(4):
        a = k * math.pi / 2
        v = rot @ np.array([math.sin(radius) * math.cos(a), math.sin(radius) * math.sin(a), math.cos(radius)])
        out.append(v / np.linalg.norm(v))
    return out


def scene_lamps(scene):
    shown = bpy.context.view_layer.objects
    lamps = []
    for ob in scene.objects:
        if ob.type != 'LIGHT' or ob.hide_render or shown.get(ob.name) != ob:
            continue
        lamp, mw = ob.data, ob.matrix_world
        if lamp.type == 'SUN':
            lamps.append(('SUN', None, np.array((mw.to_3x3() @ Vector((0, 0, 1))).normalized()),
                          max(lamp.angle / 2, 1e-3), lamp.energy))
        else:
            lamps.append(('POINT', np.array(mw.translation), None, 0.0, lamp.energy))
    return lamps


def ortho_frame(d, receivers, points):
    rot = np.array(Vector(d).to_track_quat('Z', 'Y').to_matrix())
    rl = receivers @ rot
    x0, x1 = rl[:, 0].min() - 1.0, rl[:, 0].max() + 1.0
    y0, y1 = rl[:, 1].min() - 1.0, rl[:, 1].max() + 1.0
    top = float((points @ rot[:, 2]).max()) + 1.0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    loc = rot @ np.array([cx, cy, top])
    far = top - float(rl[:, 2].min()) + 1.0
    scale = LIGHT_PIXELS / max(x1 - x0, y1 - y0)
    w, h = max(16, math.ceil((x1 - x0) * scale)), max(16, math.ceil((y1 - y0) * scale))
    matrix, depth_row = view_matrices(rot, loc, 'ORTHO', x0 - cx, x1 - cx, y0 - cy, y1 - cy, 0.0, far)
    return matrix, depth_row, w, h


def cube_frames(position, receivers):
    far = float(np.linalg.norm(receivers - position, axis=1).max()) + 1.0
    frames = []
    for axis in np.vstack([np.eye(3), -np.eye(3)]):
        rot = np.array(Vector(-axis).to_track_quat('Z', 'Y').to_matrix())
        matrix, depth_row = view_matrices(rot, position, 'PERSP', -1.0, 1.0, -1.0, 1.0, LAMP_NEAR, max(far, 1.0))
        frames.append((matrix, depth_row, LAMP_PIXELS, LAMP_PIXELS))
    return frames


def light_frames(tset, receivers, scene, sky_samples):
    pts = tset["points"][:, :3]
    rec = pts[np.unique(tset["tris"][receivers].ravel())]
    if not len(rec):
        return []
    directions = []
    lamps = scene_lamps(scene)
    for kind, _pos, d, radius, _power in lamps:
        if kind == 'SUN':
            directions += disk_directions(d, radius)
    for d, radius in hdri_suns(scene):
        directions += disk_directions(d, radius)
    directions += list(sky_directions(sky_samples))
    frames = [ortho_frame(d, rec, pts) for d in directions]
    lo, hi = rec.min(axis=0), rec.max(axis=0)
    points = [(power / max(float(np.sum((np.clip(pos, lo, hi) - pos) ** 2)), 1.0), pos)
              for kind, pos, _d, _r, power in lamps if kind == 'POINT' and power > 0.0]
    for _reach, pos in sorted(points, key=lambda p: -p[0])[:MAX_LAMPS]:
        frames += cube_frames(pos, rec)
    return frames


def light_polygons(tset, visible, receiving, scene, camera, progress=None):
    poly = tset["poly"]
    light = tset["glow"] & ~visible
    receivers = visible[poly] & receiving[tset["obj"]]
    if not receivers.any():
        return light
    hidden = ~visible[poly]
    margin = camera["light_margin"]
    if margin > 0.0:
        light |= polygons_near(tset, receiver_cells(tset, receivers, margin), margin, hidden)
    frames = light_frames(tset, receivers, scene, camera["sky_samples"])
    light |= sight(tset, frames, receivers, hidden & ~light[poly], farthest=True, progress=progress)
    return light & ~visible


def camera_near(tset, positions, margin):
    out = np.zeros(tset["npoly"], dtype=bool)
    if margin <= 0.0 or not len(positions):
        return out
    pts = tset["points"][:, :3]
    maybe = np.flatnonzero(np.isin(cell_keys(pts, margin), around(cell_keys(positions, margin))))
    near = np.zeros(len(pts), dtype=bool)
    cams = positions.astype(np.float32)
    for a in range(0, len(maybe), 1 << 16):
        part = maybe[a:a + (1 << 16)]
        d2 = ((pts[part, None, :] - cams[None, :, :]) ** 2).sum(axis=2)
        near[part] = d2.min(axis=1) <= margin * margin
    out[tset["poly"][near[tset["tris"]].any(axis=1)]] = True
    return out


def is_static_map_mesh(ob):
    if ob.type != 'MESH' or ob.data is None:
        return False
    if ob.library is not None or ob.override_library is not None:
        return False
    if ob.get(ZONE_PROP) or ob.get(COPY_PROP) or ob.get(LIGHT_PROP):
        return False
    if len(ob.modifiers) or ob.data.shape_keys is not None:
        return False
    p = ob
    while p.parent is not None:
        if p.parent_type != 'OBJECT' or p.parent.type == 'ARMATURE':
            return False
        p = p.parent
    return True


def guess_source(scene):
    best, best_n = None, 0
    for c in scene.collection.children:
        if c.get(ROLE_PROP):
            continue
        n = sum(1 for o in c.all_objects if o.type == 'MESH')
        if n > best_n:
            best, best_n = c, n
    return best


def map_objects(scene, settings):
    src = settings.source
    if src is not None and src.get(ROLE_PROP):
        src = None
    pool = list(src.all_objects) if src is not None else list(scene.collection.objects)
    pool += [o for o in scene.objects if SRC_PROP in o]
    seen, out = set(), []
    for ob in pool:
        key = ob.as_pointer()
        if key not in seen and is_static_map_mesh(ob):
            seen.add(key)
            out.append(ob)
    return out


def scene_zones(scene):
    return [(o.matrix_world.copy(), o.empty_display_size)
            for o in scene.objects if o.type == 'EMPTY' and o.get(ZONE_PROP)]


def parse_words(text):
    return [w.strip().lower() for w in text.replace(";", ",").split(",") if w.strip()]


def is_far(name, words):
    if not words:
        return False
    low = name.lower()
    tokens = set(re.split(r"[\W_]+", low))
    return any(w in tokens or ("_" in w and w in low) for w in words)


def build_plan(objs, boxes, mode, far_words, progress=None, camera=None):
    plan = {"keep": [], "copy": [], "far": [], "off": [], "tris_total": 0}
    exact = mode == 'EXACT'
    cache = {}
    for i, ob in enumerate(objs):
        if progress is not None and i % 64 == 0:
            progress(i)
        plan["tris_total"] += tri_count(ob.data)
        if is_far(ob.name, far_words):
            if (camera is not None and camera["cull_far"]
                    and classify_view(ob, world_matrix(ob, cache), camera)[0] == OUTSIDE):
                plan["off"].append(ob)
            else:
                plan["far"].append(ob)
            continue
        if boxes is None:
            state, touch, cut = INSIDE, None, None
        else:
            state, touch, cut = classify_object(ob, world_matrix(ob, cache), boxes, exact)
        if camera is not None:
            plan_with_view(plan, ob, mode, (state, touch, cut), camera, cache)
            continue
        if state == OUTSIDE:
            plan["off"].append(ob)
        elif state == INSIDE or mode == 'WHOLE':
            plan["keep"].append(ob)
        elif exact:
            plan["copy"].append((ob, touch, cut))
        elif (touch.size - int(touch.sum()) >= TRIM_MIN_DROP
              and touch.mean() < TRIM_MAX_INSIDE):
            plan["copy"].append((ob, touch, None))
        else:
            plan["keep"].append(ob)
    return plan


def plan_with_view(plan, ob, mode, box, camera, cache):
    state, touch, cut = box
    if state == OUTSIDE:
        plan["off"].append(ob)
        return
    view_state, view_touch, view_cut = classify_view(ob, world_matrix(ob, cache), camera)
    if view_state == OUTSIDE:
        plan["off"].append(ob)
        return
    if state == INSIDE or mode == 'WHOLE':
        touch = cut = None
    if view_state == INSIDE:
        view_touch = view_cut = None
    if touch is None and view_touch is None:
        plan["keep"].append(ob)
        return
    if touch is None:
        mask = view_touch
    elif view_touch is None:
        mask = touch
    else:
        mask = touch & view_touch
    if not mask.any():
        plan["off"].append(ob)
    elif cut is not None or view_cut is not None:
        plan["copy"].append((ob, mask, merge_cuts(mask, cut, view_cut)))
    elif mask.size - int(mask.sum()) >= TRIM_MIN_DROP and mask.mean() < TRIM_MAX_INSIDE:
        plan["copy"].append((ob, mask, None))
    else:
        plan["keep"].append(ob)


def merge_cuts(mask, cut, view_cut):
    border = np.zeros(mask.size, dtype=bool)
    cutters, snap, planes = [], None, None
    if cut is not None:
        border |= cut[0]
        cutters, snap = cut[1], cut[2]
    if view_cut is not None:
        border |= view_cut[0]
        planes = view_cut[1]
        snap = view_cut[2] if snap is None else snap
    return border & mask, cutters, snap, planes


def our_collections():
    return [c for c in bpy.data.collections if c.get(ROLE_PROP)]


def find_coll(role):
    return next((c for c in bpy.data.collections if c.get(ROLE_PROP) == role), None)


def get_coll(scene, role, create=True):
    coll = find_coll(role)
    if coll is None:
        if not create:
            return None
        coll = bpy.data.collections.new(COLL_NAMES[role])
        coll[ROLE_PROP] = role
    if not any(c == coll for c in scene.collection.children_recursive):
        scene.collection.children.link(coll)
    return coll


def find_layer_coll(layer_coll, coll):
    if layer_coll.collection == coll:
        return layer_coll
    for child in layer_coll.children:
        found = find_layer_coll(child, coll)
        if found is not None:
            return found
    return None


def set_exclude(scene, coll, state):
    for view_layer in scene.view_layers:
        lc = find_layer_coll(view_layer.layer_collection, coll)
        if lc is not None and lc.exclude != state:
            lc.exclude = state


def collection_index(scene):
    index = {}
    for coll in (scene.collection, *bpy.data.collections):
        for ob in coll.objects:
            index.setdefault(ob.as_pointer(), []).append(coll)
    return index


def move_to(scene, ob, target, ours, index):
    key = ob.as_pointer()
    current = index.get(key, [])
    if SRC_PROP not in ob:
        names = [ROOT_TOKEN if c == scene.collection else c.name
                 for c in current if not any(c == o for o in ours)]
        ob[SRC_PROP] = "\n".join(names) if names else ROOT_TOKEN
    if not any(c == target for c in current):
        target.objects.link(ob)
    for c in current:
        if c != target:
            c.objects.unlink(ob)
    index[key] = [target]


def restore_object(scene, ob, ours, index):
    targets = []
    for name in str(ob.get(SRC_PROP, "")).split("\n"):
        if not name:
            continue
        c = scene.collection if name == ROOT_TOKEN else bpy.data.collections.get(name)
        if c is not None and not any(c == o for o in ours):
            targets.append(c)
    if not targets:
        targets = [scene.collection]
    current = index.get(ob.as_pointer(), [])
    for c in targets:
        if not any(c == u for u in current):
            c.objects.link(ob)
    for c in current:
        if any(c == o for o in ours):
            c.objects.unlink(ob)
    del ob[SRC_PROP]


def fix_broken_fans(me, wanted):
    n_loops = len(me.loops)
    got = np.empty(n_loops * 3, dtype=np.float32)
    me.corner_normals.foreach_get("vector", got)
    got = got.reshape(n_loops, 3)
    bad = (np.abs(got).sum(axis=1) < 0.5) & (np.abs(wanted).sum(axis=1) > 0.5)
    if not bad.any():
        return 0
    bad_verts = np.zeros(len(me.vertices), dtype=bool)
    bad_verts[read_corner_verts(me)[bad]] = True
    edge_verts = np.empty(len(me.edges) * 2, dtype=np.int32)
    me.edges.foreach_get("vertices", edge_verts)
    touching = bad_verts[edge_verts.reshape(-1, 2)].any(axis=1)
    attr = me.attributes.get("sharp_edge")
    if attr is None:
        attr = me.attributes.new("sharp_edge", 'BOOLEAN', 'EDGE')
    sharp = np.zeros(len(me.edges), dtype=bool)
    attr.data.foreach_get("value", sharp)
    attr.data.foreach_set("value", sharp | touching)
    me.update()
    me.normals_split_custom_set(wanted.tolist())
    return int(bad_verts.sum())


def cutter_planes(m, bmin, bmax, all_bits):
    planes = []
    for axis in range(3):
        row = m[axis, :3]
        length2 = float(row @ row)
        normal = Vector(row / np.sqrt(length2))
        for bit, bound in ((axis, bmin[axis]), (axis + 3, bmax[axis])):
            if np.isfinite(bound) and not (all_bits >> bit) & 1:
                planes.append((Vector(row * ((bound - m[axis, 3]) / length2)), normal))
    return planes


def split_by_plane(bm, faces, co, no, snap):
    side = {}
    for f in faces:
        for v in f.verts:
            if v not in side:
                d = (v.co - co).dot(no)
                side[v] = 0.0 if abs(d) <= snap else d
    for e in list(dict.fromkeys(e for f in faces for e in f.edges)):
        v1, v2 = e.verts
        d1, d2 = side[v1], side[v2]
        if d1 * d2 < 0.0:
            _edge, v_new = bmesh.utils.edge_split(e, v1, d1 / (d1 - d2))
            side[v_new] = 0.0
    out, hard = [], []
    for f in faces:
        out.append(f)
        dists = [side[v] for v in f.verts]
        if not min(dists) < 0.0 < max(dists):
            continue
        on_plane = [v for v, d in zip(f.verts, dists) if d == 0.0]
        piece = None
        if len(on_plane) == 2:
            try:
                piece = bmesh.utils.face_split(f, on_plane[0], on_plane[1])[0]
            except ValueError:
                piece = None
        if piece is not None:
            out.append(piece)
        else:
            hard.append(f)
    if hard:
        known = set(out)
        for f in hard:
            f.normal_update()
        geom = [*dict.fromkeys(v for f in hard for v in f.verts),
                *dict.fromkeys(e for f in hard for e in f.edges), *hard]
        res = bmesh.ops.bisect_plane(bm, geom=geom, dist=snap, plane_co=co, plane_no=no)
        out += [g for g in res["geom"] if isinstance(g, bmesh.types.BMFace) and g not in known]
    return out


def cut_faces(bm, faces, cutters, snap, view=None):
    for m, bmin, bmax, all_bits in cutters:
        for co, no in cutter_planes(m, bmin, bmax, all_bits):
            faces = split_by_plane(bm, faces, co, no, snap)
    for co, no in view or ():
        faces = split_by_plane(bm, faces, co, no, snap)
    limits = [(Matrix(m.tolist()), bmin.tolist(), bmax.tolist()) for m, bmin, bmax, _bits in cutters]
    outside = []
    for f in faces:
        center = f.calc_center_median()
        if limits:
            for m, bmin, bmax in limits:
                p = m @ center
                if all(bmin[a] - 1e-6 <= p[a] <= bmax[a] + 1e-6 for a in range(3)):
                    break
            else:
                outside.append(f)
                continue
        if view and any((center - co).dot(no) > snap for co, no in view):
            outside.append(f)
    return outside


def make_copy(ob, touch, cut=None):
    src = ob.data
    me = src.copy()
    normal_attr = None
    if src.has_custom_normals:
        n_loops = len(src.loops)
        normals = np.empty(n_loops * 3, dtype=np.float32)
        src.corner_normals.foreach_get("vector", normals)
        attr = me.attributes.new(NORMAL_ATTR, 'FLOAT_VECTOR', 'CORNER')
        attr.data.foreach_set("vector", normals)
        normal_attr = attr.name
    bm = bmesh.new()
    try:
        bm.from_mesh(me, face_normals=False, vertex_normals=False)
        bm.faces.ensure_lookup_table()
        n_before = len(bm.faces)
        drop = [bm.faces[i] for i in np.flatnonzero(~touch).tolist()]
        if cut is not None:
            border = [bm.faces[i] for i in np.flatnonzero(cut[0]).tolist()]
            if border:
                drop += cut_faces(bm, border, cut[1], cut[2], cut[3] if len(cut) > 3 else None)
        same = not drop and len(bm.faces) == n_before
        if drop:
            bmesh.ops.delete(bm, geom=drop, context='FACES')
        n_faces = len(bm.faces)
        if n_faces and not same:
            bm.to_mesh(me)
    finally:
        bm.free()
    if same or n_faces == 0:
        bpy.data.meshes.remove(me)
        return ob if same else None
    if normal_attr is not None:
        attr = me.attributes.get(normal_attr)
        if attr is not None and len(attr.data) == len(me.loops):
            kept = np.empty(len(me.loops) * 3, dtype=np.float32)
            attr.data.foreach_get("vector", kept)
            kept = kept.reshape(-1, 3)
            length = np.linalg.norm(kept, axis=1, keepdims=True)
            kept = np.where(length > 1e-6, kept / np.maximum(length, 1e-6), 0.0).astype(np.float32)
            me.attributes.remove(attr)
            me.normals_split_custom_set(kept.tolist())
            fix_broken_fans(me, kept)
        elif attr is not None:
            me.attributes.remove(attr)
    cp = ob.copy()
    cp.data = me
    cp.name = ob.name + ".crop"
    me.name = cp.name
    if SRC_PROP in cp:
        del cp[SRC_PROP]
    cp[COPY_PROP] = ob.name
    return cp


def remove_copies():
    copies = [o for o in bpy.data.objects if o.get(COPY_PROP)]
    if not copies:
        return 0
    meshes = {o.data.name: o.data for o in copies if o.data is not None}
    bpy.data.batch_remove(ids=copies)
    orphans = [me for me in meshes.values() if me.users == 0]
    if orphans:
        bpy.data.batch_remove(ids=orphans)
    return len(copies)


def apply_plan(scene, plan, progress=None):
    keep = get_coll(scene, "KEEP")
    off = get_coll(scene, "OFF")
    far = get_coll(scene, "FAR", create=bool(plan["far"]))
    ours = our_collections()
    index = collection_index(scene)
    set_exclude(scene, off, True)
    visible = 0
    for ob in plan["keep"]:
        move_to(scene, ob, keep, ours, index)
        visible += tri_count(ob.data)
    for ob in plan["far"]:
        move_to(scene, ob, far, ours, index)
        visible += tri_count(ob.data)
    for ob in plan["off"]:
        move_to(scene, ob, off, ours, index)
    copies = whole = 0
    for i, (ob, touch, cut) in enumerate(plan["copy"]):
        if progress is not None:
            progress(i)
        cp = make_copy(ob, touch, cut)
        if cp == ob:
            move_to(scene, ob, keep, ours, index)
            visible += tri_count(ob.data)
            whole += 1
            continue
        if cp is not None:
            keep.objects.link(cp)
            visible += tri_count(cp.data)
            copies += 1
        move_to(scene, ob, off, ours, index)
    if far is not None and len(far.all_objects) == 0:
        bpy.data.collections.remove(far)
    return visible, copies, whole


def light_copy(ob, mask):
    if mask.all():
        cp = ob.copy()
        cp.data = ob.data.copy()
        if SRC_PROP in cp:
            del cp[SRC_PROP]
    else:
        cp = make_copy(ob, mask)
    origin = str(ob.get(COPY_PROP) or ob.name)
    cp[COPY_PROP] = origin
    cp[LIGHT_PROP] = 1
    cp.name = origin + ".light"
    cp.data.name = cp.name
    cp.visible_camera = False
    cp.display_type = 'BOUNDS'
    if hasattr(cp, "cycles"):
        cp.cycles.use_camera_cull = cp.cycles.use_distance_cull = False
    return cp


def crop_by_sight(scene, camera, in_view, progress=None):
    keep, far = find_coll("KEEP"), find_coll("FAR")
    zone = [o for c in (keep, far) if c is not None for o in c.objects
            if o.type == 'MESH' and o.data is not None and len(o.data.polygons)
            and not o.hide_render and o.visible_camera]
    if not zone:
        return 0, 0
    gpu_start()
    tset = triangle_set(zone, scene.render.engine in CULLING_ENGINES)
    looked_at = np.array([str(o.get(COPY_PROP) or o.name) in in_view for o in zone])[tset["obj"]]
    limit = gpu.capabilities.max_texture_size_get()
    views = camera["views"]
    frames = [view_buffer(v, camera["fov_margin"], scene, camera["detail"], limit) for v in views]
    visible = sight(tset, frames, looked_at & ~tset["through"], looked_at, progress=progress)
    visible |= camera_near(tset, np.array([v[1] for v in views]), camera["margin"])
    far_keys = {o.as_pointer() for o in far.objects} if far is not None else set()
    backdrop = np.array([o.as_pointer() in far_keys for o in zone])
    mode = camera["light_mode"]
    if mode == 'ALL':
        light = ~visible
    elif mode == 'AWARE':
        light = light_polygons(tset, visible, ~backdrop, scene, camera, progress)
    else:
        light = np.zeros_like(visible)
    return sort_by_sight(scene, zone, tset["spans"], visible, light, backdrop, camera["cull_far"])


def sort_by_sight(scene, zone, spans, visible, light, backdrop, cull_far):
    keep, far = find_coll("KEEP"), find_coll("FAR")
    off = get_coll(scene, "OFF")
    lights = None
    ours = our_collections()
    index = collection_index(scene)
    doomed, renamed = [], []
    hidden = light_only = 0
    for ob, (start, n), is_backdrop in zip(zone, spans, backdrop):
        seen = visible[start:start + n]
        lit = light[start:start + n] & ~seen
        if seen.all():
            continue
        if lit.any() and lights is None:
            lights = get_coll(scene, "LIGHT")
            ours = our_collections()
        if is_backdrop:
            if seen.any() or not (lit.any() or cull_far):
                continue
            if lit.any():
                lights.objects.link(light_copy(ob, np.ones(n, dtype=bool)))
                light_only += 1
            else:
                hidden += 1
            move_to(scene, ob, off, ours, index)
            continue
        is_copy = bool(ob.get(COPY_PROP))
        if (not is_copy and seen.any()
                and not (n - int(seen.sum()) >= TRIM_MIN_DROP and seen.mean() < TRIM_MAX_INSIDE)):
            continue
        if seen.any():
            cp = make_copy(ob, seen)
            if is_copy:
                cp[COPY_PROP] = ob[COPY_PROP]
                renamed.append((cp, ob.name))
            keep.objects.link(cp)
        if lit.any():
            lights.objects.link(light_copy(ob, lit))
            light_only += 1
        elif not seen.any():
            hidden += 1
        if is_copy:
            doomed.append(ob)
        else:
            move_to(scene, ob, off, ours, index)
    if doomed:
        meshes = [o.data for o in doomed]
        bpy.data.batch_remove(ids=doomed)
        unused = [me for me in meshes if me.users == 0]
        if unused:
            bpy.data.batch_remove(ids=unused)
    for cp, name in renamed:
        cp.name = name
        cp.data.name = name
    if far is not None and len(far.all_objects) == 0:
        bpy.data.collections.remove(far)
    return hidden, light_only


def zone_counts():
    keep, far, lights = find_coll("KEEP"), find_coll("FAR"), find_coll("LIGHT")
    shown = [o for c in (keep, far) if c is not None for o in c.objects if o.type == 'MESH' and o.data is not None]
    lit = [o for o in lights.objects if o.type == 'MESH' and o.data is not None] if lights is not None else []
    return (sum(tri_count(o.data) for o in shown),
            len(keep.objects) if keep is not None else 0,
            sum(1 for o in keep.objects if o.get(COPY_PROP)) if keep is not None else 0,
            len(far.objects) if far is not None else 0,
            len(hidden_objects(find_coll("OFF"))),
            len(lit), sum(tri_count(o.data) for o in lit))


def hidden_objects(off):
    if off is None:
        return []
    return [ob for ob in off.objects if SRC_PROP in ob and ob.type == 'MESH' and ob.data is not None]


def split_deletable(off):
    hidden = hidden_objects(off)
    if not hidden:
        return [], []
    elsewhere = set()
    for coll in (*(s.collection for s in bpy.data.scenes), *bpy.data.collections):
        if coll != off:
            elsewhere.update(ob.as_pointer() for ob in coll.objects)
    doomed = {ob.as_pointer(): ob for ob in hidden if ob.as_pointer() not in elsewhere}
    users = bpy.data.user_map(subset=list(doomed.values()))
    refs = {}
    for key, ob in doomed.items():
        found = [u.as_pointer() for u in users.get(ob, ()) if not isinstance(u, IGNORED_USERS)]
        if found:
            refs[key] = found
    changed = True
    while changed:
        changed = False
        for key, found in refs.items():
            if key in doomed and any(u != key and u not in doomed for u in found):
                del doomed[key]
                changed = True
    return list(doomed.values()), [ob for ob in hidden if ob.as_pointer() not in doomed]


def remove_unused(materials):
    dead = [m for m in materials if m.users == 0]
    if not dead:
        return 0, 0
    groups, images = {}, {}
    stack = [m.node_tree for m in dead if m.node_tree is not None]
    while stack:
        for node in stack.pop().nodes:
            image = getattr(node, "image", None)
            if isinstance(image, bpy.types.Image):
                images[image.as_pointer()] = image
            tree = getattr(node, "node_tree", None)
            if isinstance(tree, bpy.types.NodeTree) and tree.as_pointer() not in groups:
                groups[tree.as_pointer()] = tree
                stack.append(tree)
    bpy.data.batch_remove(ids=dead)
    while True:
        unused = [key for key, tree in groups.items() if tree.users == 0]
        if not unused:
            break
        bpy.data.batch_remove(ids=[groups.pop(key) for key in unused])
    unused = [image for image in images.values() if image.users == 0]
    if unused:
        bpy.data.batch_remove(ids=unused)
    return len(dead), len(unused)


def commit_crop(off, doomed, fallback_src):
    doomed_keys = {ob.as_pointer() for ob in doomed}
    promote = []
    for cp in bpy.data.objects:
        name = cp.get(COPY_PROP)
        if not name:
            continue
        orig = bpy.data.objects.get(str(name))
        if orig is None:
            promote.append((cp, str(name), fallback_src, str(name)))
        elif orig.as_pointer() in doomed_keys:
            promote.append((cp, orig.name, str(orig.get(SRC_PROP, fallback_src)), orig.data.name))
    meshes, materials = {}, {}
    for ob in doomed:
        meshes[ob.data.as_pointer()] = ob.data
        for slot in ob.material_slots:
            if slot.material is not None:
                materials[slot.material.as_pointer()] = slot.material
    bpy.data.batch_remove(ids=doomed)
    unused = [me for me in meshes.values() if me.users == 0]
    if unused:
        bpy.data.batch_remove(ids=unused)
    removed = remove_unused(list(materials.values()))
    for cp, name, src, mesh_name in promote:
        del cp[COPY_PROP]
        cp[SRC_PROP] = src
        if cp.get(LIGHT_PROP):
            name, mesh_name = name + ".light", mesh_name + ".light"
        cp.name = name
        if cp.data is not None and bpy.data.meshes.get(mesh_name) is None:
            cp.data.name = mesh_name
    if len(off.all_objects) == 0 and len(off.children) == 0:
        bpy.data.collections.remove(off)
    return removed


def map_tris():
    return sum(tri_count(ob.data) for ob in bpy.data.objects
               if SRC_PROP in ob and ob.type == 'MESH' and ob.data is not None)


def fmt(n):
    return f"{n:,}"


def plural(n, word):
    return f"{fmt(n)} {word}" if n == 1 else f"{fmt(n)} {word}s"


def path_frames(scene, step):
    frames = list(range(scene.frame_start, scene.frame_end + 1, step))
    if frames[-1] != scene.frame_end:
        frames.append(scene.frame_end)
    return frames


def sample_camera(context, settings, frames):
    scene = context.scene
    if len(frames) == 1 and frames[0] == scene.frame_current:
        cam = settings.camera or scene.camera
        if cam is None:
            return None, "There is no camera. Set the scene camera or pick one in Map Crop"
        if cam.data.type == 'PANO':
            return None, "Panoramic cameras are not supported"
        return [camera_view(cam, context.evaluated_depsgraph_get(), scene)], None
    frame, subframe = scene.frame_current, scene.frame_subframe
    wm = context.window_manager
    views = []
    wm.progress_begin(0, len(frames))
    try:
        depsgraph = context.evaluated_depsgraph_get()
        for i, f in enumerate(frames):
            wm.progress_update(i)
            scene.frame_set(f)
            cam = settings.camera or scene.camera
            if cam is None:
                return None, f"There is no camera on frame {f}. Set the scene camera or pick one in Map Crop"
            if cam.data.type == 'PANO':
                return None, "Panoramic cameras are not supported"
            views.append(camera_view(cam, depsgraph, scene))
    finally:
        scene.frame_set(frame, subframe=subframe)
        wm.progress_end()
    return views, None


def map_objects_all(scene, settings):
    src = settings.source
    if src is not None and src.get(ROLE_PROP):
        src = None
    pool = list(src.all_objects) if src is not None else list(scene.collection.objects)
    for coll in our_collections():
        pool += list(coll.all_objects)
    seen, out = set(), []
    for ob in pool:
        key = ob.as_pointer()
        if key not in seen:
            seen.add(key)
            out.append(ob)
    return out


TUNE_PROPS = ("device", "samples", "use_adaptive_sampling", "adaptive_threshold", "max_bounces", "diffuse_bounces",
              "glossy_bounces", "transmission_bounces", "transparent_max_bounces", "caustics_reflective",
              "caustics_refractive")
RENDER_BACKUP = "mapcrop_render"
TEXTURE_BACKUP = "mapcrop_textures"

TUNE_STEPS = (
    ("caustics off", ({"caustics_reflective": False, "caustics_refractive": False},)),
    ("bounces", tuple({"max_bounces": v} for v in (8, 6, 4))),
    ("diffuse bounces", tuple({"diffuse_bounces": v} for v in (3, 2, 1))),
    ("glossy bounces", tuple({"glossy_bounces": v} for v in (3, 2, 1))),
    ("transmission bounces", tuple({"transmission_bounces": v} for v in (8, 6, 4, 2))),
    ("transparent bounces", tuple({"transparent_max_bounces": v} for v in (6, 4))),
    ("noise threshold", tuple({"adaptive_threshold": v} for v in (0.02, 0.03, 0.05))),
    ("samples", tuple({"samples": v} for v in (2048, 1024, 512, 256, 128))),
)


def cheaper(c, option):
    for key, value in option.items():
        now = getattr(c, key)
        if isinstance(value, bool):
            if not now or value:
                return False
        elif key == "adaptive_threshold":
            if not c.use_adaptive_sampling or value <= now:
                return False
        elif value >= now:
            return False
    return True


def test_render(scene, path, seed):
    scene.cycles.seed = seed
    t = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    seconds = time.perf_counter() - t
    img = bpy.data.images.load(path, check_existing=False)
    try:
        w, h = img.size
        px = np.empty(w * h * 4, dtype=np.float32)
        img.pixels.foreach_get(px)
    finally:
        bpy.data.images.remove(img)
    return px.reshape(h, w, 4)[:, :, :3], seconds


def image_difference(a, b, grid=(8, 6)):
    mean = float(b.mean()) + 1e-6
    pixel = float(np.sqrt(((a - b) ** 2).mean())) / mean
    h, w = b.shape[:2]
    ys = np.linspace(0, h, min(grid[1], h) + 1).astype(int)
    xs = np.linspace(0, w, min(grid[0], w) + 1).astype(int)
    light = 0.0
    for y0, y1 in zip(ys[:-1], ys[1:]):
        for x0, x1 in zip(xs[:-1], xs[1:]):
            d = np.abs(a[y0:y1, x0:x1].mean(axis=(0, 1)) - b[y0:y1, x0:x1].mean(axis=(0, 1)))
            light = max(light, float(d.max()))
    return pixel, light / mean


def render_state(scene):
    c = scene.cycles
    state = {key: getattr(c, key) for key in TUNE_PROPS}
    state["use_persistent_data"] = scene.render.use_persistent_data
    return state


def set_render_state(scene, state):
    for key, value in state.items():
        if key == "use_persistent_data":
            scene.render.use_persistent_data = value
        else:
            setattr(scene.cycles, key, value)


def tune_render(scene, size, tolerance, progress=None):
    c, r = scene.cycles, scene.render
    out = (r.filepath, r.resolution_percentage, r.image_settings.file_format, r.image_settings.color_depth,
           r.use_sequencer, c.seed, c.use_animated_seed)
    path = os.path.join(bpy.app.tempdir, "mapcrop_tune.exr")
    steps = list(TUNE_STEPS)
    prefs = bpy.context.preferences.addons.get("cycles")
    if c.device == 'CPU' and prefs is not None and prefs.preferences.has_active_device():
        steps.insert(0, ("GPU", ({"device": 'GPU'},)))
    changes = []
    try:
        r.filepath = path
        r.image_settings.file_format = 'OPEN_EXR'
        r.image_settings.color_depth = '32'
        r.use_sequencer = False
        r.resolution_percentage = max(1, round(r.resolution_percentage * size / 100))
        c.use_animated_seed = False
        reference, t_reference = test_render(scene, path, 1)
        twin, _seconds = test_render(scene, path, 2)
        noise, spread = image_difference(twin, reference)
        if spread > 2 * tolerance + 0.01:
            original = (c.samples, c.adaptive_threshold)
            c.samples = min(c.samples * 16, 1 << 16)
            c.adaptive_threshold = max(c.adaptive_threshold / 4, 0.0005)
            reference, _seconds = test_render(scene, path, 4)
            c.samples, c.adaptive_threshold = original
            noise, spread = image_difference(twin, reference)
            spread = max(spread, image_difference(test_render(scene, path, 5)[0], reference)[1])
        best = t_reference
        n = 0
        for label, options in steps:
            for option in options:
                if option.get("device") != 'GPU' and not cheaper(c, option):
                    continue
                n += 1
                if progress is not None:
                    progress(n)
                old = {key: getattr(c, key) for key in option}
                for key, value in option.items():
                    setattr(c, key, value)
                try:
                    img, seconds = test_render(scene, path, 3)
                    pixel, light = image_difference(img, reference)
                    keep = pixel <= noise * (1.0 + 5.0 * tolerance) + 1e-4 and light <= spread + tolerance
                    if "device" in option:
                        keep = keep and seconds < best * 0.9
                except RuntimeError:
                    keep = False
                if not keep:
                    for key, value in old.items():
                        setattr(c, key, value)
                    break
                best = min(best, seconds)
                changes.append((label, old, dict(option)))
    finally:
        (r.filepath, r.resolution_percentage, r.image_settings.file_format, r.image_settings.color_depth,
         r.use_sequencer, c.seed, c.use_animated_seed) = out
    return changes, t_reference, best


LIGHT_TEXTURE = 256
MIN_TEXTURE = 64
TEXTURE_PERCENTILE = 99.5
TEXTURE_VIEWS = 24
FIT_SUFFIX = ".fit"


def follow(socket):
    while socket.is_linked:
        link = socket.links[0]
        if link.from_node.bl_idname != 'NodeReroute':
            return link.from_socket
        socket = link.from_node.inputs[0]
    return None


def uv_source(node):
    if node.projection != 'FLAT':
        return None, None
    out = follow(node.inputs["Vector"])
    if out is None:
        return "", (1.0, 1.0)
    sx = sy = 1.0
    if out.node.bl_idname == 'ShaderNodeMapping':
        scale = out.node.inputs.get("Scale")
        if scale is None or scale.is_linked or out.node.vector_type not in {'POINT', 'TEXTURE', 'VECTOR'}:
            return None, None
        sx, sy = abs(scale.default_value[0]), abs(scale.default_value[1])
        if out.node.vector_type == 'TEXTURE':
            sx, sy = 1.0 / max(sx, 1e-9), 1.0 / max(sy, 1e-9)
        out = follow(out.node.inputs["Vector"])
        if out is None:
            return None, None
    if out.node.bl_idname == 'ShaderNodeUVMap':
        return out.node.uv_map, (sx, sy)
    if out.node.bl_idname == 'ShaderNodeTexCoord' and out.name == "UV":
        return "", (sx, sy)
    return None, None


def tree_images(tree, owner, found, done):
    done.add(tree.as_pointer())
    for node in tree.nodes:
        if node.bl_idname == 'ShaderNodeTexImage' and node.image is not None:
            found.setdefault(node.image, []).append((tree, node))
        sub = getattr(node, "node_tree", None)
        if isinstance(sub, bpy.types.NodeTree) and sub.as_pointer() not in done:
            tree_images(sub, owner, found, done)


def material_images(mat):
    found = {}
    if mat is not None and mat.node_tree is not None:
        tree_images(mat.node_tree, mat, found, set())
    return found


def screen_areas(world, tris, matrix, w, h):
    clip = np.hstack([world, np.ones((len(world), 1), dtype=np.float32)]) @ matrix.astype(np.float32).T
    front = clip[:, 3] > 1e-6
    wc = np.where(front, clip[:, 3], 1.0)
    px = (clip[:, 0] / wc * 0.5 + 0.5) * w
    py = (clip[:, 1] / wc * 0.5 + 0.5) * h
    x, y = px[tris], py[tris]
    area = 0.5 * np.abs((x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0]))
    shown = front[tris].all(axis=1) & (x.max(axis=1) >= 0) & (x.min(axis=1) <= w) & (y.max(axis=1) >= 0) & (y.min(axis=1) <= h)
    return np.where(shown, area, 0.0)


def uv_areas(me, name, tri_loops):
    layer = None
    for uv in me.uv_layers:
        if (name and uv.name == name) or (not name and uv.active_render):
            layer = uv
    if layer is None:
        return None
    uv = np.empty(len(me.loops) * 2, dtype=np.float32)
    layer.uv.foreach_get("vector", uv)
    uv = uv.reshape(-1, 2)[tri_loops]
    return 0.5 * np.abs((uv[:, 1, 0] - uv[:, 0, 0]) * (uv[:, 2, 1] - uv[:, 0, 1])
                        - (uv[:, 2, 0] - uv[:, 0, 0]) * (uv[:, 1, 1] - uv[:, 0, 1]))


def weighted_percentile(values, weights, q):
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    return float(values[order][min(np.searchsorted(cum, cum[-1] * q / 100.0), len(values) - 1)])


def texture_needs(objs, visible, frames, light_objs):
    samples = {}
    cache = {}
    for ob, seen in zip(objs, visible):
        me = ob.data
        uses = {}
        for k, slot in enumerate(ob.material_slots):
            for image, nodes in material_images(slot.material).items():
                for _tree, node in nodes:
                    uses.setdefault(k, []).append((image, *uv_source(node)))
        if not uses or not seen.any():
            continue
        m = np.array(world_matrix(ob, cache), dtype=np.float64)
        world = read_positions(me) @ m[:3, :3].T.astype(np.float32) + m[:3, 3].astype(np.float32)
        nt = len(me.loop_triangles)
        tris = np.empty(nt * 3, dtype=np.int32)
        me.loop_triangles.foreach_get("vertices", tris)
        tri_loops = np.empty(nt * 3, dtype=np.int32)
        me.loop_triangles.foreach_get("loops", tri_loops)
        tpoly = np.empty(nt, dtype=np.int32)
        me.loop_triangles.foreach_get("polygon_index", tpoly)
        tris, tri_loops = tris.reshape(-1, 3), tri_loops.reshape(-1, 3)
        slot = np.zeros(len(me.polygons), dtype=np.int32)
        attr = me.attributes.get("material_index")
        if attr is not None and attr.domain == 'FACE' and len(attr.data) == len(me.polygons):
            attr.data.foreach_get("value", slot)
        tslot = np.clip(slot, 0, max(len(ob.material_slots) - 1, 0))[tpoly]
        tseen = seen[tpoly]
        screen = np.max([screen_areas(world, tris, f[0], f[2], f[3]) for f in frames], axis=0)
        uv_cache = {}
        for k, images in uses.items():
            pick = tseen & (tslot == k) & (screen > 0.0)
            for image, uv_name, scale in images:
                if uv_name is None:
                    samples.setdefault(image, []).append((np.array([np.inf]), np.array([1.0])))
                    continue
                if uv_name not in uv_cache:
                    uv_cache[uv_name] = uv_areas(me, uv_name, tri_loops)
                area = uv_cache[uv_name]
                if area is None:
                    continue
                ok = pick & (area > 1e-12)
                need = np.sqrt(screen[ok] / (area[ok] * scale[0] * scale[1]))
                samples.setdefault(image, []).append((need, screen[ok]))
    needs = {}
    for image, parts in samples.items():
        values = np.concatenate([p[0] for p in parts])
        weights = np.concatenate([p[1] for p in parts])
        if not len(values) or weights.sum() <= 0.0:
            continue
        side = weighted_percentile(values, weights, TEXTURE_PERCENTILE)
        w, h = image.size
        needs[image] = side * math.sqrt(w / h)
    for ob in light_objs:
        for slot in ob.material_slots:
            for image in material_images(slot.material):
                needs[image] = max(needs.get(image, 0.0), float(LIGHT_TEXTURE))
    return needs


def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def shrink_image(image, halvings):
    w, h = image.size
    px = np.empty(w * h * 4, dtype=np.float32)
    image.pixels.foreach_get(px)
    px = px.reshape(h, w, 4)
    srgb = not image.is_float and "srgb" in image.colorspace_settings.name.lower()
    rgb, alpha = px[:, :, :3], px[:, :, 3:]
    if srgb:
        rgb = srgb_to_linear(rgb)
    straight = image.alpha_mode == 'STRAIGHT' and float(alpha.min()) < 1.0
    if straight:
        rgb = rgb * alpha
    px = np.concatenate([rgb, alpha], axis=2)
    for _ in range(halvings):
        if px.shape[0] % 2:
            px = np.concatenate([px, px[-1:]], axis=0)
        if px.shape[1] % 2:
            px = np.concatenate([px, px[:, -1:]], axis=1)
        px = (px[0::2, 0::2] + px[1::2, 0::2] + px[0::2, 1::2] + px[1::2, 1::2]) * 0.25
    rgb, alpha = px[:, :, :3], px[:, :, 3:]
    if straight:
        rgb = np.where(alpha > 1e-6, rgb / np.maximum(alpha, 1e-6), 0.0)
    if srgb:
        rgb = linear_to_srgb(rgb)
    px = np.concatenate([rgb, alpha], axis=2).astype(np.float32)
    small = bpy.data.images.new(f"{image.name}{FIT_SUFFIX}{px.shape[1]}", px.shape[1], px.shape[0],
                                alpha=True, float_buffer=image.is_float)
    small.colorspace_settings.name = image.colorspace_settings.name
    small.alpha_mode = image.alpha_mode
    small.pixels.foreach_set(px.ravel())
    small.pack()
    return small


def texture_bytes(image):
    w, h = image.size
    return w * h * (16 if image.is_float else 4)


def restore_textures(scene):
    records = json.loads(scene.get(TEXTURE_BACKUP, "[]"))
    copies = set()
    n = 0
    for kind, tree_name, node_name, original, copy, fake in records:
        owner = (bpy.data.materials if kind == 'MATERIAL' else bpy.data.node_groups).get(tree_name)
        tree = owner.node_tree if kind == 'MATERIAL' and owner is not None else owner
        image = bpy.data.images.get(original)
        node = tree.nodes.get(node_name) if tree is not None else None
        if node is not None and image is not None:
            node.image = image
            image.use_fake_user = fake
            n += 1
        copies.add(copy)
    unused = [bpy.data.images[c] for c in copies if c in bpy.data.images and bpy.data.images[c].users == 0]
    if unused:
        bpy.data.batch_remove(ids=unused)
    if TEXTURE_BACKUP in scene:
        del scene[TEXTURE_BACKUP]
    return n


def fit_textures(scene, settings, detail, progress=None):
    if scene.get(TEXTURE_BACKUP):
        restore_textures(scene)
    keep, far, lights, off = (find_coll(r) for r in ("KEEP", "FAR", "LIGHT", "OFF"))
    if keep is None and far is None:
        src = settings.source if settings.source is not None and not settings.source.get(ROLE_PROP) else None
        shown = [o for o in (src.all_objects if src is not None else scene.collection.objects)
                 if is_static_map_mesh(o) and o.visible_get()]
    else:
        shown = [o for c in (keep, far) if c is not None for o in c.objects]
    shown = [o for o in shown if o.type == 'MESH' and o.data is not None and len(o.data.polygons)
             and not o.hide_render and o.visible_camera]
    light_objs = [o for o in lights.objects if o.type == 'MESH'] if lights is not None else []
    measured = {o.as_pointer() for o in shown} | {o.as_pointer() for o in light_objs}
    hidden = {o.as_pointer() for o in off.objects} if off is not None else set()
    rendered = bpy.context.view_layer.objects
    foreign = set()
    for ob in bpy.data.objects:
        if ob.as_pointer() in measured or ob.as_pointer() in hidden or ob.hide_render or rendered.get(ob.name) != ob:
            continue
        for slot in ob.material_slots:
            foreign |= set(material_images(slot.material))
    for owner in (*bpy.data.worlds, *bpy.data.lights):
        if owner.node_tree is not None:
            found = {}
            tree_images(owner.node_tree, owner, found, set())
            foreign |= set(found)
    frames_src = path_frames(scene, settings.frame_step) if scene.frame_end > scene.frame_start else [scene.frame_current]
    views, error = sample_camera(bpy.context, settings, frames_src)
    if error:
        raise ValueError(error)
    views = distinct_views(views)
    views = views[::max(1, math.ceil(len(views) / TEXTURE_VIEWS))]
    r = scene.render
    width = r.resolution_x * r.resolution_percentage / 100 * detail
    height = r.resolution_y * r.resolution_percentage / 100 * detail
    frames = [(view_matrices(rot, loc, kind, *frame, near, far)[0], None, width, height)
              for rot, loc, kind, frame, near, far in views]
    try:
        gpu_start()
        tset = triangle_set(shown, scene.render.engine in CULLING_ENGINES)
        limit = gpu.capabilities.max_texture_size_get()
        seen = sight(tset, [view_buffer(v, settings.fov_margin, scene, 1.0, limit) for v in views],
                     ~tset["through"], np.ones(len(tset["tris"]), dtype=bool), progress=progress)
        visible = [seen[s:s + n] for s, n in tset["spans"]]
    except (SystemError, RuntimeError, ValueError):
        visible = [np.ones(len(o.data.polygons), dtype=bool) for o in shown]
    needs = texture_needs(shown, visible, frames, light_objs)
    users = {}
    for mat in bpy.data.materials:
        for image, nodes in material_images(mat).items():
            for tree, node in nodes:
                kind = 'MATERIAL' if tree == mat.node_tree else 'GROUP'
                name = mat.name if kind == 'MATERIAL' else tree.name
                users.setdefault(image, {})[(kind, name, node.name)] = node
    records, before, after = [], 0, 0
    for image, need in needs.items():
        w, h = image.size
        if image in foreign or image.source not in {'FILE', 'GENERATED'} or not w or not h or not math.isfinite(need):
            continue
        halvings = 0
        while w / 2 ** (halvings + 1) >= max(need, MIN_TEXTURE) and h / 2 ** (halvings + 1) >= MIN_TEXTURE:
            halvings += 1
        if not halvings or len(image.pixels) != w * h * image.channels or image.channels != 4:
            continue
        small = shrink_image(image, halvings)
        before += texture_bytes(image)
        after += texture_bytes(small)
        fake = image.use_fake_user
        image.use_fake_user = True
        for (kind, tree_name, node_name), node in users.get(image, {}).items():
            node.image = small
            records.append([kind, tree_name, node_name, image.name, small.name, fake])
    scene[TEXTURE_BACKUP] = json.dumps(records)
    return len({r[3] for r in records}), before, after


class MapCropSettings(bpy.types.PropertyGroup):
    source: PointerProperty(
        name="Map",
        type=bpy.types.Collection,
        description="Collection that holds the imported map",
        poll=lambda self, coll: not coll.get(ROLE_PROP),
    )
    box_size: FloatProperty(
        name="Box Size", default=60.0, min=1.0, soft_max=500.0,
        subtype='DISTANCE', unit='LENGTH',
        description="Side of a new box. After that move and scale it like any other object",
    )
    cut_mode: EnumProperty(
        name="Border",
        description="What to do with objects that cross the border of the zone",
        items=(
            ('EXACT', "Cut Exactly at Box",
             "Geometry is cut along the box faces: nothing sticks out of the box, nothing inside it is lost"),
            ('POLYS', "Whole Polygons",
             "Whole polygons that touch the zone stay. Faster, but the edge is ragged"),
            ('WHOLE', "Whole Objects",
             "An object stays whole if any part of it reaches into the zone"),
        ),
        default='EXACT',
    )
    margin: FloatProperty(
        name="Margin", default=2.0, min=0.0, soft_max=50.0,
        subtype='DISTANCE', unit='LENGTH',
        description="How much wider the zone is than the box itself. Not used by Cut Exactly at Box",
    )
    full_height: BoolProperty(
        name="Full Height", default=True,
        description="Do not cut vertically: everything above and below the box stays. "
                    "When off, the top and bottom of the box cut as well",
    )
    keep_far: BoolProperty(
        name="Keep Backdrop", default=True,
        description="Objects with these words in their names always stay whole "
                    "in the MAP_FAR collection and are never cut",
    )
    far_words: StringProperty(
        name="Backdrop Words", default="backdrop, cloud, water_harbor, sky",
        description="Comma separated",
    )
    camera: PointerProperty(
        name="Camera",
        type=bpy.types.Object,
        description="Camera whose view is kept. Empty: the scene camera, switched by camera markers too",
        poll=lambda self, ob: ob.type == 'CAMERA' and self.id_data.objects.get(ob.name) == ob,
    )
    fov_margin: FloatProperty(
        name="FOV Margin", default=math.radians(10.0), min=0.0, max=math.radians(90.0),
        subtype='ANGLE',
        description="Added to the field of view of the camera, across and up: "
                    "10° widens the view by 5° on each side",
    )
    view_margin: FloatProperty(
        name="Distance Margin", default=5.0, min=0.0, soft_max=50.0,
        subtype='DISTANCE', unit='LENGTH',
        description="Geometry this close to the view stays as well: next to the edges of the frame, "
                    "just behind the camera and past its clip end",
    )
    frame_step: IntProperty(
        name="Frame Step", default=1, min=1, soft_max=24,
        description="Test every n-th frame of the scene range. Frames that show almost the same "
                    "view are skipped in any case",
    )
    cull_far: BoolProperty(
        name="Cull Backdrop", default=True,
        description="Backdrop objects the camera never sees are hidden too. The ones it sees stay whole",
    )
    occlusion: BoolProperty(
        name="Hide Occluded", default=True,
        description="Also hides what the camera does not see because something is in front of it. "
                    "Every sampled frame is drawn on the GPU from the camera; glass and alpha "
                    "materials do not block the view. Lighting can change: hidden walls and roofs "
                    "no longer cast shadows",
    )
    occlusion_detail: IntProperty(
        name="Detail", default=100, min=10, max=400, subtype='PERCENTAGE',
        description="Resolution of the visibility test against the render resolution. "
                    "More keeps thinner and smaller details",
    )
    light_mode: EnumProperty(
        name="Lighting",
        description="What happens to the geometry the camera does not see",
        items=(
            ('AWARE', "Keep What Lights the Shot",
             "Hidden geometry that may shade what the camera sees (from the lamps, the sun of the HDRI "
             "or the sky), is close enough to bounce light onto it or glows stays as light-only "
             "geometry in MAP_LIGHT: invisible to the camera, drawn as boxes in the viewport"),
            ('ALL', "Keep All for Light",
             "Everything hidden stays as light-only geometry. The lighting is exactly the same, "
             "only the viewport gets lighter"),
            ('NONE', "Drop Hidden",
             "Hidden geometry goes. The lightest scene, but its shadows, sky light and bounces are lost"),
        ),
        default='AWARE',
    )
    light_margin: FloatProperty(
        name="Bounce Distance", default=3.0, min=0.0, soft_max=20.0,
        subtype='DISTANCE', unit='LENGTH',
        description="Hidden geometry this close to what the camera sees stays for the light it bounces",
    )
    sky_samples: IntProperty(
        name="Sky Directions", default=32, min=0, soft_max=128,
        description="Directions of the sky tested for shadows and blocked sky light. "
                    "Sun lamps and the brightest spots of the HDRI are always tested",
    )
    tune_size: IntProperty(
        name="Test Size", default=25, min=5, max=100, subtype='PERCENTAGE',
        description="Size of the test renders of Tune Render against the render size. "
                    "Smaller is quicker, larger judges finer detail",
    )
    tune_tolerance: FloatProperty(
        name="Tolerance", default=2.0, min=0.0, max=20.0, subtype='PERCENTAGE',
        description="How much farther from the original look a cheaper setting may go than two "
                    "renders of the original are from each other",
    )
    texture_detail: IntProperty(
        name="Texture Detail", default=100, min=25, max=400, subtype='PERCENTAGE',
        description="Texels per pixel Fit Textures keeps: 100% keeps one texel for every pixel "
                    "of the render, on the frames of the camera path",
    )
    render_stats: StringProperty()
    stats: StringProperty()


class MAPCROP_OT_add_zone(bpy.types.Operator):
    bl_idname = "mapcrop.add_zone"
    bl_label = "Add Zone Box"
    bl_description = "Puts a box at the center of the view. Move and scale it over the part of the map you need"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def execute(self, context):
        scene = context.scene
        settings = scene.mapcrop
        loc = scene.cursor.location.copy()
        space = context.space_data
        if space is not None and space.type == 'VIEW_3D' and space.region_3d is not None:
            loc = space.region_3d.view_location.copy()
        zone = bpy.data.objects.new("MapCrop_Zone", None)
        zone.empty_display_type = 'CUBE'
        zone.empty_display_size = settings.box_size * 0.5
        zone.location = loc
        zone.show_in_front = True
        zone[ZONE_PROP] = 1
        scene.collection.objects.link(zone)
        for ob in context.selected_objects:
            ob.select_set(False)
        zone.select_set(True)
        context.view_layer.objects.active = zone
        if settings.source is None:
            settings.source = guess_source(scene)
        return {'FINISHED'}


def crop(op, context, camera=None):
    scene = context.scene
    settings = scene.mapcrop
    context.view_layer.update()
    zones = scene_zones(scene)
    mode = settings.cut_mode
    boxes = None
    if zones:
        margin = 0.0 if mode == 'EXACT' else settings.margin
        boxes = zone_boxes(zones, margin, settings.full_height)
        if not boxes:
            op.report({'ERROR'}, "The box has zero size along one of its axes")
            return {'CANCELLED'}
    elif camera is None:
        op.report({'ERROR'}, "There is no zone box yet. Add a box first")
        return {'CANCELLED'}
    if settings.source is None:
        settings.source = guess_source(scene)
    objs = map_objects(scene, settings)
    if not objs:
        op.report({'ERROR'}, "There are no meshes in the map collection")
        return {'CANCELLED'}
    words = parse_words(settings.far_words) if settings.keep_far else []
    sight = camera is not None and camera["occlusion"]
    failed = None
    if sight:
        try:
            gpu_start()
        except SystemError as e:
            sight, failed = False, e
    wm = context.window_manager
    wm.progress_begin(0, len(objs))
    try:
        plan = build_plan(objs, boxes, mode, words, wm.progress_update, camera)
        if not plan["keep"] and not plan["copy"]:
            if camera is None:
                op.report({'ERROR'}, "The boxes do not touch the map, nothing was changed")
            elif boxes is None:
                op.report({'ERROR'}, "The camera does not see the map, nothing was changed")
            else:
                op.report({'ERROR'}, "The camera sees nothing inside the boxes, nothing was changed")
            return {'CANCELLED'}
        if sight:
            in_view = {o.name for o in plan["keep"]} | {o.name for o in plan["far"]}
            in_view |= {o.name for o, _touch, _cut in plan["copy"]}
            plan = build_plan(objs, boxes, mode, words, wm.progress_update)
        remove_copies()
        visible, copies, whole = apply_plan(scene, plan, wm.progress_update)
        sorted_out = None
        if sight:
            try:
                sorted_out = crop_by_sight(scene, camera, in_view, wm.progress_update)
            except (SystemError, RuntimeError, ValueError) as e:
                failed = e
    finally:
        wm.progress_end()
    if failed is not None:
        op.report({'WARNING'}, f"Hide Occluded was skipped: {failed}")
        if sight:
            camera["occlusion"] = False
            return crop(op, context, camera)
    in_zone = len(plan["keep"]) + copies + whole
    hidden = len(plan["off"]) + len(plan["copy"]) - whole
    backdrop = len(plan["far"])
    if sorted_out is not None:
        visible, in_zone, copies, backdrop, hidden, light_objs, light_tris = zone_counts()
    lines = [
        f"Visible: {fmt(visible)} of {fmt(plan['tris_total'])} tris",
        f"In zone: {plural(in_zone, 'object')} ({fmt(copies)} cut)",
        f"Backdrop: {fmt(backdrop)}, hidden: {fmt(hidden)}",
    ]
    if camera is not None:
        lines.append(camera["info"])
    if sorted_out is not None and camera["light_mode"] != 'NONE':
        lines.append(f"Light only: {plural(light_objs, 'object')}, {fmt(light_tris)} tris")
    settings.stats = "\n".join(lines)
    op.report({'INFO'}, lines[0])
    return {'FINISHED'}


class MAPCROP_OT_apply(bpy.types.Operator):
    bl_idname = "mapcrop.apply"
    bl_label = "Crop to Zone"
    bl_description = ("Leaves visible only what falls into the boxes, plus the backdrop. The rest goes "
                      "to the disabled MAP_OFF collection, nothing is deleted")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def execute(self, context):
        return crop(self, context)


class MAPCROP_OT_camera(bpy.types.Operator):
    bl_idname = "mapcrop.camera"
    bl_label = "Crop to Camera"
    bl_options = {'REGISTER', 'UNDO'}

    current_frame: BoolProperty(
        name="Current Frame", default=False, options={'SKIP_SAVE'},
        description="Only the current frame, cut exactly at the view",
    )

    @classmethod
    def description(cls, context, props):
        occluded = (" With Hide Occluded, what other geometry covers goes as well; what lights "
                    "the shot stays, invisible to the camera.")
        if props.current_frame:
            return ("Keeps what the camera sees on the current frame, inside the boxes. Geometry is cut "
                    "exactly at the edges of the view with the margins." + occluded + " Nothing is deleted")
        return ("Keeps what the camera sees over the frame range of the scene, inside the boxes. Whole "
                "polygons stay; a camera that does not move is cut exactly." + occluded + " Nothing is deleted")

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def execute(self, context):
        scene = context.scene
        settings = scene.mapcrop
        if self.current_frame:
            frames = [scene.frame_current]
        else:
            frames = path_frames(scene, settings.frame_step)
        views, error = sample_camera(context, settings, frames)
        if error:
            self.report({'ERROR'}, error)
            return {'CANCELLED'}
        camera = camera_zone(views, settings.fov_margin, settings.view_margin)
        camera.update(cull_far=settings.cull_far, occlusion=settings.occlusion,
                      detail=settings.occlusion_detail / 100, fov_margin=settings.fov_margin,
                      margin=settings.view_margin, views=distinct_views(views),
                      light_mode=settings.light_mode, light_margin=settings.light_margin,
                      sky_samples=settings.sky_samples)
        if self.current_frame:
            camera["info"] = f"Camera: frame {frames[0]}, cut exactly"
        elif camera["exact"]:
            camera["info"] = "Camera: does not move, cut exactly"
        else:
            camera["info"] = (f"Camera: {plural(len(camera['normals']), 'view')}, "
                              f"frames {frames[0]}-{frames[-1]}")
        return crop(self, context, camera)


class MAPCROP_OT_cycles_cull(bpy.types.Operator):
    bl_idname = "mapcrop.cycles_cull"
    bl_label = "Cycles Culling"
    bl_description = ("Turns on Simplify > Culling > Camera Culling and Distance Culling for Cycles and "
                      "enables both on every map object. On each frame Cycles then skips the objects "
                      "out of view; the ones closer than the culling distance stay. Light-only geometry "
                      "is never culled: it lights the shot from out of view")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        if not hasattr(context.scene, "cycles"):
            cls.poll_message_set("Cycles is not enabled")
            return False
        return True

    def execute(self, context):
        scene = context.scene
        settings = scene.mapcrop
        if settings.source is None:
            settings.source = guess_source(scene)
        scene.render.use_simplify = True
        scene.cycles.use_camera_cull = True
        scene.cycles.use_distance_cull = True
        n = 0
        for ob in map_objects_all(scene, settings):
            if ob.type in CULL_TYPES:
                cull = not ob.get(LIGHT_PROP)
                ob.cycles.use_camera_cull = cull
                ob.cycles.use_distance_cull = cull
                n += cull
        self.report({'INFO'}, f"Cycles culling is on for {plural(n, 'object')}")
        return {'FINISHED'}


class MAPCROP_OT_tune_render(bpy.types.Operator):
    bl_idname = "mapcrop.tune_render"
    bl_label = "Tune Render"
    bl_description = ("Finds cheaper Cycles settings that look the same. The current frame is rendered small; "
                      "then a GPU, caustics off, fewer bounces, a looser noise threshold and fewer samples are "
                      "tried in turn, each kept only if the render stays as close to the original as two "
                      "renders of the original are. For an animation Persistent Data is turned on too. "
                      "Every step is a render, so it takes a while")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        if context.scene.render.engine != 'CYCLES':
            cls.poll_message_set("Works with Cycles")
            return False
        if context.scene.camera is None:
            cls.poll_message_set("The scene has no camera")
            return False
        return True

    def execute(self, context):
        scene = context.scene
        settings = scene.mapcrop
        if RENDER_BACKUP not in scene:
            scene[RENDER_BACKUP] = json.dumps(render_state(scene))
        wm = context.window_manager
        wm.progress_begin(0, 32)
        try:
            changes, before, after = tune_render(scene, settings.tune_size, settings.tune_tolerance / 100,
                                                 wm.progress_update)
        finally:
            wm.progress_end()
        lines = [f"Test render: {before:.1f} s -> {after:.1f} s (x{before / max(after, 1e-6):.1f})"]
        for label, old, new in changes:
            values = ", ".join(f"{old[k]} -> {new[k]}" for k in new if not isinstance(new[k], bool))
            lines.append(f"  {label}" + (f": {values}" if values else ""))
        if scene.frame_end > scene.frame_start and not scene.render.use_persistent_data:
            scene.render.use_persistent_data = True
            lines.append("  Persistent Data on: the animation keeps the scene between frames")
        if len(lines) == 1:
            lines.append("  nothing cheaper looks the same")
        settings.render_stats = "\n".join(lines)
        self.report({'INFO'}, lines[0])
        return {'FINISHED'}


class MAPCROP_OT_restore_render(bpy.types.Operator):
    bl_idname = "mapcrop.restore_render"
    bl_label = "Restore Render Settings"
    bl_description = "Puts back the render settings from before Tune Render"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        if RENDER_BACKUP not in context.scene:
            cls.poll_message_set("Tune Render has not changed anything")
            return False
        return True

    def execute(self, context):
        scene = context.scene
        set_render_state(scene, json.loads(scene[RENDER_BACKUP]))
        del scene[RENDER_BACKUP]
        scene.mapcrop.render_stats = "Render settings restored"
        return {'FINISHED'}


class MAPCROP_OT_fit_textures(bpy.types.Operator):
    bl_idname = "mapcrop.fit_textures"
    bl_label = "Fit Textures"
    bl_description = ("Replaces textures bigger than any frame of the camera path needs by smaller copies: a "
                      "texel stays no bigger than a pixel of the render. Textures seen only through bounced "
                      "light get small. Textures also used by objects outside the map, the world or lights, "
                      "or not mapped by UVs, are left alone. The originals stay in the file")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and context.scene.camera is not None

    def execute(self, context):
        scene = context.scene
        settings = scene.mapcrop
        if settings.source is None:
            settings.source = guess_source(scene)
        wm = context.window_manager
        wm.progress_begin(0, 32)
        try:
            n, before, after = fit_textures(scene, settings, settings.texture_detail / 100, wm.progress_update)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        finally:
            wm.progress_end()
        mb = 1 << 20
        settings.render_stats = (f"Textures: {plural(n, 'image')} made smaller, "
                                 f"{fmt(before // mb)} MB -> {fmt(after // mb)} MB")
        self.report({'INFO'}, settings.render_stats)
        return {'FINISHED'}


class MAPCROP_OT_restore_textures(bpy.types.Operator):
    bl_idname = "mapcrop.restore_textures"
    bl_label = "Restore Textures"
    bl_description = "Puts the original textures back and removes the smaller copies"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        if not context.scene.get(TEXTURE_BACKUP):
            cls.poll_message_set("Fit Textures has not changed anything")
            return False
        return True

    def execute(self, context):
        n = restore_textures(context.scene)
        context.scene.mapcrop.render_stats = f"Textures restored in {plural(n, 'node')}"
        return {'FINISHED'}


class MAPCROP_OT_reset(bpy.types.Operator):
    bl_idname = "mapcrop.reset"
    bl_label = "Restore Full Map"
    bl_description = ("Puts the objects back into their original collections and removes the cut copies. "
                      "Geometry deleted with Confirm does not come back")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def execute(self, context):
        scene = context.scene
        ours = our_collections()
        remove_copies()
        index = collection_index(scene)
        n = 0
        for ob in list(bpy.data.objects):
            if SRC_PROP in ob:
                restore_object(scene, ob, ours, index)
                n += 1
        for coll in ours:
            if len(coll.all_objects) == 0:
                bpy.data.collections.remove(coll)
        scene.mapcrop.stats = ""
        self.report({'INFO'}, f"Objects restored: {fmt(n)}")
        return {'FINISHED'}


class MAPCROP_OT_commit(bpy.types.Operator):
    bl_idname = "mapcrop.commit"
    bl_label = "Confirm (Delete Polygons)"
    bl_description = ("Makes the crop permanent: deletes everything the crop has hidden, together with the "
                      "meshes, materials and images nothing else uses. Restore Full Map will not bring it back")
    bl_options = {'UNDO'}

    count = 0
    tris = 0

    @classmethod
    def poll(cls, context):
        if context.mode != 'OBJECT':
            return False
        off = find_coll("OFF")
        if off is None or len(off.objects) == 0:
            cls.poll_message_set("Nothing is hidden yet. Use Crop to Zone first")
            return False
        return True

    def invoke(self, context, _event):
        doomed, _spared = split_deletable(find_coll("OFF"))
        if not doomed:
            return self.execute(context)
        self.count = len(doomed)
        self.tris = sum(tri_count(ob.data) for ob in doomed)
        return context.window_manager.invoke_props_dialog(
            self, width=380, title="Delete hidden geometry?", confirm_text="Delete")

    def draw(self, _context):
        col = self.layout.column(align=True)
        col.label(text=f"{plural(self.count, 'hidden object')}, {fmt(self.tris)} tris", icon='TRASH')
        col.label(text="will be deleted from this file for good.")
        col.label(text="Restore Full Map will not bring them back.")
        col.label(text="Keep a copy of the full map file in case you need it.")

    def execute(self, context):
        settings = context.scene.mapcrop
        off = find_coll("OFF")
        doomed, spared = split_deletable(off)
        if not doomed:
            if spared:
                self.report({'ERROR'}, "Other objects still use the hidden ones, nothing was deleted")
            else:
                self.report({'ERROR'}, "Nothing is hidden. Use Crop to Zone first")
            return {'CANCELLED'}
        src = settings.source
        fallback = src.name if src is not None and not src.get(ROLE_PROP) else ROOT_TOKEN
        deleted = len(doomed)
        before = map_tris()
        n_materials, n_images = commit_crop(off, doomed, fallback)
        lines = [
            f"Deleted {plural(deleted, 'hidden object')}",
            f"Left: {fmt(map_tris())} of {fmt(before)} tris",
        ]
        if n_materials or n_images:
            lines.append(f"Removed {plural(n_materials, 'material')}, {plural(n_images, 'image')}")
        if spared:
            lines.append(f"Kept {plural(len(spared), 'hidden object')} still in use")
        settings.stats = "\n".join(lines)
        self.report({'INFO'}, lines[0])
        return {'FINISHED'}


class MAPCROP_PT_panel(bpy.types.Panel):
    bl_idname = "MAPCROP_PT_panel"
    bl_label = "Map Crop 1.6"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Map Crop"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        settings = scene.mapcrop
        if not bpy.data.filepath:
            box = layout.box()
            box.label(text="File is not saved", icon='ERROR')
            box.label(text="Save the scene before cropping")
        layout.prop(settings, "source")
        col = layout.column(align=True)
        col.operator("mapcrop.add_zone", icon='CUBE')
        col.prop(settings, "box_size")
        col = layout.column(align=True)
        col.prop(settings, "cut_mode", text="")
        row = col.row()
        row.enabled = settings.cut_mode != 'EXACT'
        row.prop(settings, "margin")
        col.prop(settings, "full_height")
        col.prop(settings, "keep_far")
        row = col.row()
        row.enabled = settings.keep_far
        row.prop(settings, "far_words", text="")
        col = layout.column(align=True)
        col.scale_y = 1.3
        col.operator("mapcrop.apply", icon='CHECKMARK')
        header, body = layout.panel("MAPCROP_camera")
        header.label(text="Camera", icon='CAMERA_DATA')
        if body is not None:
            body.prop(settings, "camera", placeholder=scene.camera.name if scene.camera else "")
            col = body.column(align=True)
            col.prop(settings, "fov_margin")
            col.prop(settings, "view_margin")
            col.prop(settings, "frame_step")
            row = body.row()
            row.enabled = settings.keep_far
            row.prop(settings, "cull_far")
            row = body.row(align=True)
            row.prop(settings, "occlusion")
            sub = row.row(align=True)
            sub.enabled = settings.occlusion
            sub.prop(settings, "occlusion_detail")
            col = body.column(align=True)
            col.enabled = settings.occlusion
            col.prop(settings, "light_mode", text="")
            sub = col.column(align=True)
            sub.enabled = settings.light_mode == 'AWARE'
            sub.prop(settings, "light_margin")
            sub.prop(settings, "sky_samples")
            col = body.column(align=True)
            row = col.row(align=True)
            row.scale_y = 1.3
            row.operator("mapcrop.camera", icon='VIEW_CAMERA').current_frame = False
            row.operator("mapcrop.camera", text="Current Frame", icon='TIME').current_frame = True
            col.label(text=f"Frames {scene.frame_start}-{scene.frame_end}, step {settings.frame_step}")
        header, body = layout.panel("MAPCROP_render", default_closed=True)
        header.label(text="Render", icon='RENDER_STILL')
        if body is not None:
            body.operator("mapcrop.cycles_cull", icon='SHADING_RENDERED')
            body.prop(scene.render, "use_persistent_data")
            col = body.column(align=True)
            row = col.row(align=True)
            row.prop(settings, "tune_size")
            row.prop(settings, "tune_tolerance")
            row = col.row(align=True)
            row.operator("mapcrop.tune_render", icon='PREFERENCES')
            row.operator("mapcrop.restore_render", text="", icon='LOOP_BACK')
            col = body.column(align=True)
            col.prop(settings, "texture_detail")
            row = col.row(align=True)
            row.operator("mapcrop.fit_textures", icon='TEXTURE')
            row.operator("mapcrop.restore_textures", text="", icon='LOOP_BACK')
            if settings.render_stats and settings.render_stats.isascii():
                box = body.box()
                for line in settings.render_stats.split("\n"):
                    box.label(text=line)
        layout.operator("mapcrop.reset", icon='LOOP_BACK')
        layout.operator("mapcrop.commit", icon='TRASH')
        if settings.stats and settings.stats.isascii():
            box = layout.box()
            for line in settings.stats.split("\n"):
                box.label(text=line)


classes = (
    MapCropSettings,
    MAPCROP_OT_add_zone,
    MAPCROP_OT_apply,
    MAPCROP_OT_camera,
    MAPCROP_OT_cycles_cull,
    MAPCROP_OT_tune_render,
    MAPCROP_OT_restore_render,
    MAPCROP_OT_fit_textures,
    MAPCROP_OT_restore_textures,
    MAPCROP_OT_reset,
    MAPCROP_OT_commit,
    MAPCROP_PT_panel,
)


def register():
    if hasattr(bpy.types.Scene, "mapcrop"):
        del bpy.types.Scene.mapcrop
    stale = [c for c in bpy.types.PropertyGroup.__subclasses__()
             if c.__name__ == MapCropSettings.__name__ and c.is_registered]
    stale += [c for c in classes if c.is_registered and c not in stale]
    for cls in stale:
        bpy.utils.unregister_class(cls)
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.mapcrop = PointerProperty(type=MapCropSettings)


def unregister():
    del bpy.types.Scene.mapcrop
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
