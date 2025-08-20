from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering

class ClusterModule():

    def __init__(self, cfg):
        self.cfg = cfg

    def cluster(self, embeddings, starts, ends):
        # =========================
        # 读取后处理配置（可选项）
        # post:
        #   mode_filter: "safer" 或 "majority"
        #   mode_k: 3            # 仅当 mode_filter=majority 时使用
        #   min_turn: 0.10
        #   merge_tol: 0.001
        # =========================
        post = getattr(self.cfg, "post", None)
        mode_filter = getattr(post, "mode_filter", "safer")  # "safer" / "majority"
        mode_k = getattr(post, "mode_k", 3)
        min_turn = getattr(post, "min_turn", 0.10)
        tol = getattr(post, "merge_tol", 1e-3)

        # 1) 聚类前可选归一化
        if self.cfg.cluster.normalize:
            embeddings = normalize(embeddings, axis=1, norm='l2')

        # 2) 自适应/固定聚类数
        num_cluster = self.cfg.cluster.num_cluster
        if num_cluster is None or (isinstance(num_cluster, str) and str(num_cluster).strip().lower() == 'none'):
            cluster_labels = AgglomerativeClustering(
                n_clusters=None,
                metric='cosine',
                linkage='average',
                distance_threshold=self.cfg.cluster.threshold
            ).fit_predict(embeddings)
        else:
            cluster_labels = AgglomerativeClustering(
                n_clusters=int(num_cluster),
                linkage='average',
                metric='cosine'
            ).fit_predict(embeddings)

        # 3) 标签小平滑（可配置）
        cluster_labels = list(cluster_labels)
        if mode_filter == "majority":
            cluster_labels = self._mode_filter(cluster_labels, k=mode_k)  # 多数票 k=mode_k
        else:
            cluster_labels = self._mode_filter_safer(cluster_labels)  # 只消 A,B,A 孤点

        # 4) 组段
        SEC_tuples = [(s, e, c) for s, e, c in zip(starts, ends, cluster_labels)]

        # 5) 自适应合并（同说话人相邻/重叠合并）
        if not self.cfg.vad.merge_vad:
            SEC_tuples = self.merge_speakers(SEC_tuples, tol=tol)

        # 6) 最短话段平滑（把很短的小段并给相邻）
        #    建议：min_turn=0.10~0.12，tol=1e-3
        # 调试统计（可注释）
        # print("#segs before smooth:", len(SEC_tuples))
        SEC_tuples = self.smooth_short_turns(SEC_tuples, min_turn=min_turn, tol=tol)
        # print("#segs after  smooth:", len(SEC_tuples))

        return SEC_tuples

    #原本1e-6
    def merge_speakers(self, SEC_tuples, tol=1e-3):
        """
        自适应合并：
        - 如果相邻片段属于同一说话人，并且时间上相连或重叠，就合并
        - 说话人不同就断开
        - 不依赖 win/hop，不做“切半”
        """
        output = []
        if not SEC_tuples:
            return output

        SEC_tuples = sorted(SEC_tuples, key=lambda x: (x[0], x[1]))  # 按时间排序
        prev_s, prev_e, prev_c = SEC_tuples[0]

        for s, e, c in SEC_tuples[1:]:
            if c == prev_c and s <= prev_e + tol:  # 同说话人且有重叠/相邻
                prev_e = max(prev_e, e)  # 延长区间
            else:
                output.append((prev_s, prev_e, prev_c))
                prev_s, prev_e, prev_c = s, e, c

        output.append((prev_s, prev_e, prev_c))
        return [(round(a, 3), round(b, 3), c) for a, b, c in output]

    #有两个旋钮可以改，min_turn和tol
    #min_turn是只有小于0.1s的片段，会被合并，
    def smooth_short_turns(self, SEC_tuples, min_turn=0.10, tol=1e-3):
        """
        最短话段平滑：将时长 < min_turn 的小片段并入相邻片段，优先并给同说话人。
        规则（按优先级）：
          1) 若左邻同说话人 → 并到左边（延长左段的 end）
          2) 否则若右邻同说话人 → 并到右边（提前右段的 start）
          3) 否则并给“更长”的一侧（左/右二选一；若一侧不存在，就并给另一侧）
        注意：输入应为按时间排序后的 (start, end, label) 列表。
        """
        if not SEC_tuples:
            return []

        # 按时间排序，避免乱序
        segs = sorted(SEC_tuples, key=lambda x: (x[0], x[1]))
        n = len(segs)
        i = 0
        out = []

        while i < n:
            s, e, c = segs[i]
            dur = max(0.0, e - s)

            # 正常长度，直接入栈，并与上一个同人片段做一次自适应合并
            if dur + tol >= min_turn or n == 1:
                if not out:
                    out.append([s, e, c])
                else:
                    ps, pe, pc = out[-1]
                    if pc == c and s <= pe + tol:
                        out[-1][1] = max(pe, e)
                    else:
                        out.append([s, e, c])
                i += 1
                continue

            # dur < min_turn：尝试并入邻居
            left_idx = len(out) - 1
            right_idx = i + 1 if (i + 1) < n else None

            # 1) 左邻同人：合到左边
            if left_idx >= 0 and out[left_idx][2] == c:
                out[left_idx][1] = max(out[left_idx][1], e)
                i += 1
                continue

            # 2) 右邻同人：合到右边（把右段起点向左扩展）
            if right_idx is not None and segs[right_idx][2] == c:
                rs, re, rc = segs[right_idx]
                segs[right_idx] = (min(s, rs), re, rc)
                i += 1
                continue

            # 3) 左右都不同人：并给更长的一侧
            left_dur = (out[left_idx][1] - out[left_idx][0]) if left_idx >= 0 else -1
            right_dur = (segs[right_idx][1] - segs[right_idx][0]) if right_idx is not None else -1

            if left_dur >= right_dur and left_idx >= 0:
                out[left_idx][1] = max(out[left_idx][1], e)
            elif right_idx is not None:
                rs, re, rc = segs[right_idx]
                segs[right_idx] = (min(s, rs), re, rc)
            # else：左右都不存在，忽略这个极短段
            i += 1

        # 收尾再合并一次，保证相邻同人拼起来
        out.sort(key=lambda x: (x[0], x[1]))
        merged = []
        for s, e, c in out:
            if not merged:
                merged.append([s, e, c])
                continue
            ps, pe, pc = merged[-1]
            if c == pc and s <= pe + tol:
                merged[-1][1] = max(pe, e)
            else:
                merged.append([s, e, c])

        return [(round(s, 3), round(e, 3), c) for s, e, c in merged]

    def _mode_filter(self, labels, k=3):
        """简单多数票平滑：窗口 k（奇数），把中心点改成窗口众数。"""
        import numpy as np
        from collections import Counter
        if k <= 1 or len(labels) < k:
            return labels
        r = k // 2
        out = labels.copy()
        for i in range(r, len(labels) - r):
            win = labels[i - r: i + r + 1]
            out[i] = Counter(win).most_common(1)[0][0]
        return out
    #这个是多数票平滑，aba
    def _mode_filter_safer(self, labels):
        out = labels.copy()
        for i in range(1, len(labels) - 1):
            if labels[i - 1] == labels[i + 1] and labels[i] != labels[i - 1]:
                out[i] = labels[i - 1]
        return out




