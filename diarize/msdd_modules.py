# 文件：diarize/msdd_modules.py
import torch, torch.nn as nn, torch.nn.functional as F

def l2n(x, dim=-1, eps=1e-8): return x/(x.norm(p=2,dim=dim,keepdim=True)+eps)
def cosine_sim(a,b): return (l2n(a,dim=-1)*l2n(b,dim=-1)).sum(-1)

class ScaleWeightCNN(nn.Module):
    def __init__(self, K, Ne, conv_channels=16, kernel_size=1, fc_hidden=256):
        super().__init__()
        Cin=3*K
        self.conv1=nn.Conv1d(Cin,conv_channels,kernel_size=kernel_size,padding=0,bias=False)
        self.bn1=nn.BatchNorm1d(conv_channels)
        self.conv2=nn.Conv1d(conv_channels,conv_channels,kernel_size=kernel_size,padding=0,bias=False)
        self.bn2=nn.BatchNorm1d(conv_channels)
        self.fc1=nn.Linear(conv_channels*Ne, fc_hidden)
        self.fc2=nn.Linear(fc_hidden, K)
    def forward(self, D):              # D: B x (3K) x Ne
        x=F.relu(self.bn1(self.conv1(D)))
        x=F.relu(self.bn2(self.conv2(x)))
        x=torch.flatten(x,1)
        return F.softmax(self.fc2(F.relu(self.fc1(x))), dim=-1)  # B x K

class MSDDStep(nn.Module):
    def __init__(self, K, Ne):
        super().__init__()
        self.cnn=ScaleWeightCNN(K,Ne)
    def forward(self, ui, v1, v2):     # ui,v1,v2: B x K x Ne
        D=torch.cat([ui,v1,v2],dim=1)           # B x (3K) x Ne
        w=self.cnn(D)                            # B x K
        sim1=cosine_sim(ui,v1); sim2=cosine_sim(ui,v2) # B x K
        C=torch.cat([w*sim1, w*sim2], dim=-1)   # B x (2K)
        return C

class MSDDDecoder(nn.Module):
    def __init__(self, K, Ne, lstm_hidden=256, lstm_layers=2, bidir=True):
        super().__init__()
        self.step=MSDDStep(K,Ne)
        self.lstm=nn.LSTM(input_size=2*K, hidden_size=lstm_hidden,
                          num_layers=lstm_layers, batch_first=True, bidirectional=bidir)
        outdim=2*lstm_hidden if bidir else lstm_hidden
        self.out=nn.Linear(outdim,2)
    def forward(self, U_seq, V1, V2):  # U_seq: B x T x K x Ne
        B,T,K,Ne=U_seq.shape
        Cs=[]
        for t in range(T):
            Cs.append(self.step(U_seq[:,t], V1, V2))  # B x (2K)
        C=torch.stack(Cs, dim=1)                      # B x T x (2K)
        H,_=self.lstm(C)
        return torch.sigmoid(self.out(H))             # B x T x 2
