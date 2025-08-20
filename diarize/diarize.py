import os
import sys
import numpy as np
import hashlib   # === NEW: 缓存需要

cur = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(cur))

from utils import write_rttm, read_vadfile
from vad import VADModule
from embeddings import EmbeddingModule
from cluster import ClusterModule


# =======================
# 辅助：多尺度切分/对齐/融合（零训练）
# =======================
def _sliding_segments_in_vad(vad_intervals, win, hop):
    starts, ends, ids = [], [], []
    seg_id = 0
    for s, e in vad_intervals:
        t = s
        # 常规整窗
        while t + win < e:
            starts.append(round(t, 3))
            ends.append(round(t + win, 3))
            ids.append(seg_id); seg_id += 1
            t += hop
        # 固定长度贴尾：最后一个窗口长度仍是 win，但对齐到末端
        if t < e:
            tail_start = max(s, max(t, e - win))   # ← 关键防护
            starts.append(round(tail_start, 3))
            ends.append(round(e, 3))
            ids.append(seg_id); seg_id += 1
    return starts, ends, ids


def _centers(starts, ends):
    if len(starts) == 0:
        return np.zeros((0,), dtype=float)
    s = np.asarray(starts, dtype=float)
    e = np.asarray(ends, dtype=float)
    return (s + e) * 0.5


def _align_to_base(base_st, base_en, multi_scales_segs):
    """
    将所有尺度的切片，对齐到 base 尺度的时间步：
    返回 pairs: len = N_base，每个元素是一行 [(k, idx_k), ...]
    其中 idx_k 是尺度 k 中与该 base 片段“中心最近”的片段索引（找不到则 -1）
    """
    base_ctr = _centers(base_st, base_en)
    pairs = []
    for i in range(len(base_st)):
        row = []
        for k, (st_k, en_k, _) in enumerate(multi_scales_segs):
            ctr_k = _centers(st_k, en_k)
            if ctr_k.size == 0:
                row.append((k, -1))
            else:
                j = int(np.argmin(np.abs(ctr_k - base_ctr[i])))
                row.append((k, j))
        pairs.append(row)
    return pairs


# L2 + 加权/等权融合
def _l2norm(x, axis=-1, eps=1e-9):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)

def _fuse_weighted(embeds_per_scale, pairs, weights=None):
    """
    - 每个尺度向量先 L2
    - 再加权平均（weights=None=等权）
    - 最后 fused 再 L2
    """
    K = len(embeds_per_scale)
    if weights is None:
        w = np.ones((K,), dtype=np.float32) / K
    else:
        w = np.asarray(weights, dtype=np.float32)
        w = w / (np.sum(w) + 1e-9)

    D = embeds_per_scale[0].shape[1]
    U = np.zeros((len(pairs), D), dtype=np.float32)

    for i, row in enumerate(pairs):
        vecs, ws = [], []
        for slot, (k, j) in enumerate(row):
            if j >= 0:
                v = embeds_per_scale[k][j]
                vecs.append(_l2norm(v[None, :])[0])
                w_slot = w[slot] if slot < len(w) else (1.0 / max(1, K))
                ws.append(w_slot)
        if vecs:
            V = np.stack(vecs, axis=0)
            W = np.asarray(ws, dtype=np.float32)
            W = W / (np.sum(W) + 1e-9)
            u = (W[:, None] * V).sum(axis=0)
            U[i] = _l2norm(u[None, :])[0]
        else:
            U[i] = 0.0
    return U


class DiarizationModule():

    def __init__(self, cfg):
        self.cfg = cfg

        self.vad_module = VADModule(cfg)
        self.embedding_module = EmbeddingModule(cfg)
        self.cluster_module = ClusterModule(cfg)

        # Create output_dir
        self.output_dir = self.cfg.misc.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self, wav_file, ref_rttm_file, vad_file=""):
        """
        兼容模式：
        - 若 cfg.embedding.win_length_list / hop_length_list 存在，则走“多尺度”路径（零训练融合）。
        - 否则保持原单尺度流程不变。
        """

        # =======================
        # 1) VAD（保持原逻辑）
        # =======================
        if vad_file == "" or (os.path.isfile(vad_file) == False):
            orig_vadresults = self.vad_module.get_pyannote_segments(wav_file)
        else:
            orig_vadresults_raw = read_vadfile(vad_file)
            orig_vadresults = []
            # filter by min_duration_on
            for st, en in orig_vadresults_raw:
                if (en - st) > self.cfg.vad.min_duration_on:
                    orig_vadresults.append([st, en])

        # 是否先合并 VAD 间隔
        if self.cfg.vad.merge_vad:
            merged_vadresults = orig_vadresults
        else:
            merged_vadresults = self.vad_module.merge_intervals(orig_vadresults)

        # =======================
        # 2) 分支：多尺度 or 单尺度
        # =======================
        wins = getattr(self.cfg.embedding, "win_length_list", None)
        hops = getattr(self.cfg.embedding, "hop_length_list", None)
        base_win = getattr(self.cfg.embedding, "base_win", 0.5)
        base_hop = getattr(self.cfg.embedding, "base_hop", 0.25)

        use_multiscale = (
            wins is not None
            and hops is not None
            and isinstance(wins, (list, tuple))
            and isinstance(hops, (list, tuple))
            and len(wins) == len(hops)
            and len(wins) >= 1
        )

        if use_multiscale:
            # =============== 多尺度路径（零训练） ===============
            # 2.1 为每个尺度做切分
            segs_per_scale = []
            for w, h in zip(wins, hops):
                st, en, ids = _sliding_segments_in_vad(merged_vadresults, w, h)
                segs_per_scale.append((st, en, ids))

            # 2.2 基准尺度（建议 0.5s / 0.25s）
            base_st, base_en, base_ids = _sliding_segments_in_vad(merged_vadresults, base_win, base_hop)

            # === NEW: 缓存机制（避免重复算 VAD→embedding→融合U）
            def _hash_vad(intervals):
                arr = np.array(intervals, dtype=np.float32)
                arr = np.round(arr, 3)  # 对齐到毫秒，避免浮点噪声
                return hashlib.md5(arr.tobytes()).hexdigest()

            def _cache_key(wav, wins, hops, base_win, base_hop, vad_hash):
                sig = f"{os.path.basename(wav)}|{wins}|{hops}|{base_win}|{base_hop}|{vad_hash}"
                return hashlib.md5(sig.encode()).hexdigest()

            cache_dir = os.path.join(self.output_dir, "_cache")
            os.makedirs(cache_dir, exist_ok=True)
            vad_hash = _hash_vad(merged_vadresults)
            ckey = _cache_key(wav_file, wins, hops, base_win, base_hop, vad_hash)
            cache_path = os.path.join(cache_dir, f"{ckey}.npz")

            U = None  # 先尝试读缓存
            if os.path.isfile(cache_path):
                data = np.load(cache_path, allow_pickle=False)
                U = data["U"]
                base_st = data["base_st"]
                base_en = data["base_en"]

            if U is None:
                # 2.3 抽取每个尺度的嵌入（复用你现有的接口）
                embeds_per_scale = []
                for (st, en, ids) in segs_per_scale:
                    vad_segments = list(zip(st, en, ids))
                    E = self.embedding_module.extract_embeddings(wav_file, vad_segments)
                    embeds_per_scale.append(E)

                # 2.4 对齐到基准尺度
                pairs = _align_to_base(base_st, base_en, segs_per_scale)

                # 2.5 融合 —— 等权（你当前最优）
                U = _fuse_weighted(embeds_per_scale, pairs, weights=None)

                # 写缓存（U / base_st / base_en）
                np.savez_compressed(
                    cache_path,
                    U=U.astype(np.float32),
                    base_st=np.asarray(base_st, dtype=np.float32),
                    base_en=np.asarray(base_en, dtype=np.float32),
                )
            # === NEW 结束

            # 2.6 聚类（复用你原来的）
            SEC_tuples = self.cluster_module.cluster(U, base_st, base_en)

            # 2.7 写 RTTM（用基准尺度的时间）
            sys_rttm = os.path.join(self.output_dir, ref_rttm_file.split('/')[-1])
            write_rttm(SEC_tuples, sys_rttm)
            return sys_rttm

        else:
            # =============== 单尺度旧路径（原样保留） ===============
            if self.cfg.vad.merge_vad:
                # 使用原始起止时间
                starts = [tup[0] for tup in orig_vadresults]
                ends = [tup[1] for tup in orig_vadresults]
                seg_ids = list(range(len(starts)))
                vad_segments = list(zip(starts, ends, seg_ids))
                embeddings = self.embedding_module.extract_embeddings(wav_file, vad_segments)
            else:
                # 用默认窗口做滑窗
                starts, ends, seg_ids = self.vad_module.sliding_window(merged_vadresults)
                vad_segments = list(zip(starts, ends, seg_ids))
                embeddings = self.embedding_module.extract_embeddings(wav_file, vad_segments)

            assert len(starts) == embeddings.shape[0]
            SEC_tuples = self.cluster_module.cluster(embeddings, starts, ends)

            sys_rttm = os.path.join(self.output_dir, ref_rttm_file.split('/')[-1])
            write_rttm(SEC_tuples, sys_rttm)
            return sys_rttm
