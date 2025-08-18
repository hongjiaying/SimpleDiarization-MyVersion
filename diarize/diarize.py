import os
import sys
import numpy as np

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
    """
    在 VAD 有声区间内做滑窗切分，返回 (starts, ends, ids)
    """
    starts, ends, ids = [], [], []
    seg_id = 0
    for s, e in vad_intervals:
        t = s
        # 注意用 <=，避免漏掉边界合法窗口
        while t + win <= e:
            starts.append(t)
            ends.append(t + win)
            ids.append(seg_id)
            seg_id += 1
            t += hop
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
    其中 idx_k 是尺度 k 中与该 base 片段“中心最近”的切片索引（找不到则 -1）
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


# >>> MOD: 新增 L2 归一化与“加权融合”（可退化为等权），更稳
def _l2norm(x, axis=-1, eps=1e-9):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)

def _fuse_weighted(embeds_per_scale, pairs, weights=None):
    """
    融合（改进版）：
    - 先对每个尺度取到的向量做 L2 归一化
    - 再做加权平均（weights=None 时=等权）
    - 最后对 fused 向量再做一次 L2 归一化
    embeds_per_scale[k] : np.ndarray [N_k, D]
    pairs : list of [(k, idx_k), ...]，按 base 步对齐
    weights : None 或 长度=K 的权重列表（会自动归一化）
    返回: U : [N_base, D]
    """
    K = len(embeds_per_scale)
    if weights is None:
        w = np.ones((K,), dtype=np.float32) / K
    else:
        w = np.asarray(weights, dtype=np.float32)
        # 允许传入与 K 不同长度（按 pairs 的顺序使用）；这里先整体归一化
        w = w / (np.sum(w) + 1e-9)

    D = embeds_per_scale[0].shape[1]
    U = np.zeros((len(pairs), D), dtype=np.float32)

    for i, row in enumerate(pairs):
        vecs = []
        ws = []
        for slot, (k, j) in enumerate(row):
            if j >= 0:
                v = embeds_per_scale[k][j]
                vecs.append(_l2norm(v[None, :])[0])  # 先 L2 归一化
                # 若提供的 weights 长度 < 实际尺度个数，这里安全取值
                w_slot = w[slot] if slot < len(w) else (1.0 / max(1, K))
                ws.append(w_slot)
        if vecs:
            V = np.stack(vecs, axis=0)             # [m, D]
            W = np.asarray(ws, dtype=np.float32)   # [m]
            W = W / (np.sum(W) + 1e-9)
            u = (W[:, None] * V).sum(axis=0)       # 加权平均
            U[i] = _l2norm(u[None, :])[0]          # 再 L2
        else:
            U[i] = 0.0
    return U
# <<< MOD


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
            merged_vadresults = orig_vadresults  # 你原注释写“需用原始起止时间”，这里按你原意保留
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

            # 2.3 抽取每个尺度的嵌入（复用你现有的接口）
            embeds_per_scale = []
            for (st, en, ids) in segs_per_scale:
                # 组装成你 EmbeddingModule 期望的三元组 (start, end, seg_id)
                vad_segments = list(zip(st, en, ids))
                E = self.embedding_module.extract_embeddings(wav_file, vad_segments)
                embeds_per_scale.append(E)

            # 2.4 对齐到基准尺度
            pairs = _align_to_base(base_st, base_en, segs_per_scale)

            # >>> MOD: 2.5 融合 —— 使用“L2+加权”版本（更稳）
            # 若你的尺度顺序是 wins=[1.5, 1.0, 0.5]，建议给 0.5s 稍小权重：
            # 可先试 [0.4, 0.4, 0.2]；如果你 wins 顺序不同，请同步调整
            # 如果你想保持完全等权，直接把 weights=None 即可。
            try:
                # 简单按窗口长短给权重：长窗更稳，短窗略小
                order = np.argsort(-np.array(wins))  # 从大到小
                # 基于 wins 构造一个与 pairs 顺序一致的权重向量（与 segs 顺序相同）
                # 这里假设 segs_per_scale 的顺序就是 wins 的顺序
                # 默认：三尺度给 [0.4,0.4,0.2]；两尺度给 [0.6,0.4]；其它情形等权
                if len(wins) == 3:
                    weights = [0.4, 0.4, 0.2]
                elif len(wins) == 2:
                    weights = [0.6, 0.4]
                else:
                    weights = None  # 等权
            except Exception:
                weights = None

            U = _fuse_weighted(embeds_per_scale, pairs, weights=weights)  # [N_base, D]
            # <<< MOD

            # 2.6 聚类（复用你原来的）
            SEC_tuples = self.cluster_module.cluster(U, base_st, base_en)

            # 2.7 写 RTTM（用基准尺度的时间）
            sys_rttm = os.path.join(self.output_dir, ref_rttm_file.split('/')[-1])
            # 提醒：utils.write_rttm 建议用我给你的带 min_dur 版本，避免 0.000 报错
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
