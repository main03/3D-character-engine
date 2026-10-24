"""Vectorised CPU triangle rasterisation shared by UV baking and depth rendering."""

import numpy as np


def rasterize(tri: np.ndarray, width: int, height: int, chunk: int = 4096):
    """Rasterise 2D triangles given in pixel coordinates (pixel centres at integers).

    tri: (F, 3, 2) array of (x, y). Returns (ys, xs, face_ids, bary) for every covered pixel,
    with bary of shape (N, 3). Pixels covered by several triangles appear once per triangle.
    """
    out = ([], [], [], [])
    for start in range(0, len(tri), chunk):
        t = tri[start : start + chunk]
        lo = np.floor(t.min(axis=1))
        hi = np.ceil(t.max(axis=1))
        lo = np.clip(lo, 0, [width - 1, height - 1]).astype(np.int64)
        hi = np.clip(hi, 0, [width - 1, height - 1]).astype(np.int64)
        w = hi[:, 0] - lo[:, 0] + 1
        h = hi[:, 1] - lo[:, 1] + 1
        counts = w * h
        fid = np.repeat(np.arange(len(t)), counts)
        local = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        xs = lo[fid, 0] + local % w[fid]
        ys = lo[fid, 1] + local // w[fid]

        a, b, c = t[fid, 0], t[fid, 1], t[fid, 2]
        v0, v1 = b - a, c - a
        v2 = np.stack([xs, ys], axis=1) - a
        den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        valid = np.abs(den) > 1e-12
        den = np.where(valid, den, 1.0)
        w1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
        w2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
        w0 = 1.0 - w1 - w2
        eps = -1e-4
        inside = valid & (w0 >= eps) & (w1 >= eps) & (w2 >= eps)

        out[0].append(ys[inside])
        out[1].append(xs[inside])
        out[2].append(fid[inside] + start)
        out[3].append(np.stack([w0, w1, w2], axis=1)[inside])
    return tuple(np.concatenate(o) for o in out)
