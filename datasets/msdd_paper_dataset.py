# 文件：datasets/msdd_paper_dataset.py
import os, numpy as np, torch, yaml
from torch.utils.data import Dataset
from collections import namedtuple, defaultdict

from diarize.vad import VADModule
from diarize.embeddings import EmbeddingModule
from diarize.utils import Dict2ObjParser, read_inputlist

from sklearn.cluster import AgglomerativeClustering

Sample = namedtuple("Sample", ["U_seq","V1","V2","Y","pair","wav"])

def _centers(starts, ends):
    if len(starts)==0: return np.zeros((0,), dtype=float)
    s, e = np.asarray(starts), np.asarray(ends)
    return (s+e)*0.5

def _sliding_segments_in_vad(vad, win, hop):
    starts, ends, ids = [], [], []
    gid=0
    for s,e in vad:
        t=s
        while t+win<e:
            starts.append(round(t,3)); ends.append(round(t+win,3)); ids.append(gid); gid+=1
            t+=hop
        if t<e:
            tail=max(s, max(t, e-win))
            starts.append(round(tail,3)); ends.append(round(e,3)); ids.append(gid); gid+=1
    return starts, ends, ids

def _align_to_base(base_st, base_en, segs_per_scale):
    base_ctr=_centers(base_st, base_en)
    pairs=[]
    for i in range(len(base_st)):
        row=[]
        for k,(st,en,_) in enumerate(segs_per_scale):
            ctr=_centers(st,en)
            if ctr.size==0: row.append((k,-1))
            else: row.append((k, int(np.argmin(np.abs(ctr-base_ctr[i])))))
        pairs.append(row)
    return pairs

def _l2n(x, axis=-1, eps=1e-9):
    n=np.linalg.norm(x, axis=axis, keepdims=True)
    return x/(n+eps)

def _fuse_equal(embeds_per_scale, pairs):
    D=embeds_per_scale[0].shape[1]
    U=np.zeros((len(pairs),D),dtype=np.float32)
    for i,row in enumerate(pairs):
        ve=[]
        for (k,j) in row:
            if j>=0: ve.append(_l2n(embeds_per_scale[k][j][None,:])[0])
        U[i]=_l2n(np.mean(ve,axis=0,keepdims=True))[0] if ve else 0.0
    return U

def _read_rttm(rttm_path, file_id=None):
    spk2ivs=defaultdict(list)
    if file_id is None: file_id=os.path.splitext(os.path.basename(rttm_path))[0]
    with open(rttm_path,'r',encoding='utf-8') as f:
        for line in f:
            if not line.strip() or line.startswith(';'): continue
            p=line.strip().split()
            if len(p)<9 or p[0]!="SPEAKER": continue
            if p[1]!=file_id: continue
            st=float(p[3]); du=float(p[4]); spk=p[7]
            spk2ivs[spk].append((st, st+du))
    for spk,ivs in spk2ivs.items():
        ivs=sorted(ivs); merged=[]
        for s,e in ivs:
            if not merged or s>merged[-1][1]: merged.append([s,e])
            else: merged[-1][1]=max(merged[-1][1],e)
        spk2ivs[spk]=[(round(a,3),round(b,3)) for a,b in merged]
    return spk2ivs

def _ov(a1,a2,b1,b2): return max(0.0, min(a2,b2)-max(a1,b1))

def _Y_from_true(base_st, base_en, spk_ivs_dict, spkA, spkB, thr=0.5):
    T=len(base_st); Y=np.zeros((T,2),dtype=np.int64)
    A=spk_ivs_dict.get(spkA,[]); B=spk_ivs_dict.get(spkB,[])
    for i in range(T):
        s,e=base_st[i], base_en[i]; dur=max(1e-6, e-s)
        ovA=sum(_ov(s,e,x1,x2) for x1,x2 in A)/dur if A else 0.0
        ovB=sum(_ov(s,e,x1,x2) for x1,x2 in B)/dur if B else 0.0
        Y[i,0]=1 if ovA>thr else 0
        Y[i,1]=1 if ovB>thr else 0
    return Y

class MSDDPaperDataset(Dataset):
    """
    论文做法的数据集：
      - 多尺度切片 & 提 embedding
      - 用等权融合U做一次聚类，得到簇数S与逐步簇标签
      - 计算每尺度簇均值 v_k^{(s)}，用于“参考向量”
      - 为会话内每个簇对 (s,q) 生成一个训练样本：
           U_seq[T,K,Ne], V1=vs, V2=vq, Y[T,2] 来自 RTTM 的 (spk_s, spk_q)
        （这里用“簇→最相近真说话人”做一个简单映射来取标签）
    """
    def __init__(self, cfg):
        self.cfg=cfg
        self.wav_list=read_inputlist(cfg.misc.input_list)
        self.vad=VADModule(cfg)
        self.emb=EmbeddingModule(cfg)
        self.wins=list(cfg.embedding.win_length_list)
        self.hops=list(cfg.embedding.hop_length_list)
        self.K=len(self.wins)
        self.base_win=float(cfg.embedding.base_win)
        self.base_hop=float(cfg.embedding.base_hop)
        self.label_thr=float(getattr(cfg.msdd,'base_overlap_thr',0.5))

    def __len__(self): return len(self.wav_list)

    def __getitem__(self, idx):
        wav=self.wav_list[idx]
        # 1) VAD
        vad = (self.vad.get_pyannote_segments(wav) if not self.cfg.vad.ref_vad
               else [tuple(map(float, line.strip().split()[:2]))
                    for line in open(wav.replace('.wav', self.cfg.vad.ref_suffix))])
        if self.cfg.vad.merge_vad: vad_m=vad
        else: vad_m=self.vad.merge_intervals([list(x) for x in vad])

        # 2) 切片
        segs_per_scale=[]
        for w,h in zip(self.wins,self.hops):
            st,en,ids=_sliding_segments_in_vad(vad_m,w,h); segs_per_scale.append((st,en,ids))
        base_st,base_en,base_ids=_sliding_segments_in_vad(vad_m,self.base_win,self.base_hop)
        pairs=_align_to_base(base_st,base_en,segs_per_scale)

        # 3) 提各尺度 embedding
        embeds_per_scale=[]
        for (st,en,ids) in segs_per_scale:
            E=self.emb.extract_embeddings(wav, list(zip(st,en,ids))).astype(np.float32)
            embeds_per_scale.append(E)
        Ne=embeds_per_scale[0].shape[1]

        # 4) 构造 U_seq[T,K,Ne]
        T=len(pairs)
        U_seq=np.zeros((T,self.K,Ne),dtype=np.float32)
        for i,row in enumerate(pairs):
            for slot,(k,j) in enumerate(row):
                U_seq[i,slot]=_l2n(embeds_per_scale[k][j][None,:])[0] if j>=0 else 0.0

        # 5) 用等权融合的 U 做聚类初始化（得到 S 和 逐步簇标签）
        U_fused=_fuse_equal(embeds_per_scale,pairs)   # [T,Ne]
        # 自适应簇数：使用 cfg.cluster.threshold 作为 distance_threshold
        if str(self.cfg.cluster.num_cluster).lower()=="none":
            ac=AgglomerativeClustering(n_clusters=None, metric='cosine', linkage='average',
                                       distance_threshold=self.cfg.cluster.threshold)
        else:
            ac=AgglomerativeClustering(n_clusters=int(self.cfg.cluster.num_cluster),
                                       metric='cosine', linkage='average')
        labels=ac.fit_predict(U_fused)  # [T]
        clusters=sorted(set(labels)); S=len(clusters)

        # 6) 每尺度簇均值 v_k^{(s)}
        #    对每个簇 s、尺度 slot，收集 i属于簇s 的 U_seq[i,slot] 做均值
        V_all=np.zeros((S,self.K,Ne),dtype=np.float32)
        for si,s in enumerate(clusters):
            for slot in range(self.K):
                xs=[U_seq[i,slot] for i in range(T) if labels[i]==s]
                V_all[si,slot]=_l2n(np.mean(xs,axis=0,keepdims=True))[0] if xs else 0.0

        # 7) 生成训练样本：对每对簇 (s,q)
        #    标签来自 RTTM：需要把簇映射到“最相近的真实说话人”
        ref=wav.replace('.wav','.rttm'); file_id=os.path.splitext(os.path.basename(ref))[0]
        spk2ivs=_read_rttm(ref, file_id)
        # 先计算 base 步的真实“活跃说话人”集合
        # 为计算“簇→真说话人”的映射，用重叠计数
        spk_list=list(spk2ivs.keys())
        # 逐步确定哪个说话人活跃
        active_mat=np.zeros((len(spk_list),T),dtype=np.int32)
        for si,spk in enumerate(spk_list):
            for i in range(T):
                s,e=base_st[i], base_en[i]
                if any(_ov(s,e,x1,x2)>(e-s)*0.5 for x1,x2 in spk2ivs[spk]):  # >50%算活跃
                    active_mat[si,i]=1

        # 簇->真说话人：按“在该簇内活跃次数最多”的真说话人
        cluster2spk={}
        for si,s in enumerate(clusters):
            idx=np.where(labels==s)[0]
            if idx.size==0 or len(spk_list)==0:
                cluster2spk[si]=None
            else:
                scores=active_mat[:,idx].sum(axis=1)  # 每个真说话人在该簇的活跃计数
                cluster2spk[si]=spk_list[int(np.argmax(scores))] if scores.sum()>0 else None

        # 产出所有 (s,q) 对作为一个 batch=1 的样本列表（这里 __getitem__ 仅返回第一个对；
        # 你可以在 DataLoader 外层循环一遍会话，或把 dataset 设计成展开式。为简单，这里仅返回第一对。）
        pairs_idx=[(i,j) for i in range(S) for j in range(i+1,S)]
        if not pairs_idx:
            # 单簇无法训练两说话人模型，返回一个空样本
            return None

        si,qi=pairs_idx[0]
        spkA=cluster2spk[si]; spkB=cluster2spk[qi]
        # 若映射失败，则先用“谁活跃多就当A/B”，避免中断
        if spkA is None or spkB is None or spkA==spkB:
            # 找两个活跃时长最长的真说话人
            items=[(spk, sum(e-s for s,e in ivs)) for spk,ivs in spk2ivs.items()]
            items.sort(key=lambda x: x[1], reverse=True)
            if len(items)>=2: spkA,spkB=items[0][0],items[1][0]
            elif len(items)==1: spkA,spkB=items[0][0], items[0][0]
            else: spkA=spkB=None

        V1=V_all[si]  # [K,Ne]
        V2=V_all[qi]  # [K,Ne]
        if spkA is not None and spkB is not None:
            Y=_Y_from_true(base_st,base_en,spk2ivs,spkA,spkB,thr=self.label_thr)  # [T,2]
        else:
            Y=np.zeros((T,2),dtype=np.int64)

        # 转 torch
        U_seq=torch.from_numpy(U_seq).float().unsqueeze(0)  # [1,T,K,Ne]
        V1=torch.from_numpy(V1).float().unsqueeze(0)        # [1,K,Ne]
        V2=torch.from_numpy(V2).float().unsqueeze(0)        # [1,K,Ne]
        Y =torch.from_numpy(Y ).long().unsqueeze(0)         # [1,T,2]
        return Sample(U_seq,V1,V2,Y,pair=(int(si),int(qi)),wav=wav)

    @staticmethod
    def collate(batch):
        batch=[b for b in batch if b is not None]
        if not batch: return None
        b0=batch[0]
        return b0.U_seq, b0.V1, b0.V2, b0.Y
