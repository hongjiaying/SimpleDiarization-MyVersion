# diarize/multiscale_min.py
from dataclasses import dataclass
import numpy as np

@dataclass
class Segment:
    start: float
    end: float

def make_sliding_segments(vad_regions, win, hop):
    """在 VAD 区间内按窗口/步长做滑窗切分"""
    segs = []
    for s, e in vad_regions:
        t = s
        while t + win <= e:
            segs.append(Segment(t, t + win))
            t += hop
    return segs

def _centers(segs):
    if not segs:
        return np.zeros((0,), dtype=float)
    return np.array([(x.start + x.end) * 0.5 for x in segs], dtype=float)

def align_to_base(base_segs, segs_per_scale):
    """
    对齐到基准0.5s：对每个base片段i，在每个尺度里找“中心时间最近”的片段索引。
    返回 pairs: 长度=N_base，元素=[(k, idx_k), ...]
    """
    base_ctr = _centers(base_segs)
    pairs = []
    for i in range(len(base_segs)):
        row = []
        for k, segs_k in enumerate(segs_per_scale):
            ctr_k = _centers(segs_k)
            if ctr_k.size == 0:
                row.append((k, -1))
            else:
                j = int(np.argmin(np.abs(ctr_k - base_ctr[i])))
                row.append((k, j))
        pairs.append(row)
    return pairs

def fuse_equal_embeddings(embeds_per_scale, pairs):
    """
    等权融合：对每个基准时间步，把各尺度对应的向量做平均。
    embeds_per_scale[k] : [N_k, D]
    返回 U : [N_base, D]
    """
    D = embeds_per_scale[0].shape[1]
    U = np.zeros((len(pairs), D), dtype=np.float32)
    for i, row in enumerate(pairs):
        vecs = []
        for (k, j) in row:
            if j >= 0:
                vecs.append(embeds_per_scale[k][j])
        U[i] = np.mean(vecs, axis=0) if vecs else 0.0
    return U
