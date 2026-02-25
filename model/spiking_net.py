import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

# --- Model (모델) ---
# SNN 아키텍처 정의 (npx_module.py 역할)
# Linear -> LIF -> Linear -> LIF 구조
class SpikingNet(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, beta):
        super().__init__()
        
        # 레이어 정의
        # fc1: 입력 특징 -> 은닉층
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        # lif1: 첫 번째 LIF 뉴런 층 (Spike Gradient로 fast_sigmoid 사용)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        # fc2: 은닉층 -> 출력층
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        # lif2: 출력 LIF 뉴런 층 (최종 스파이크 출력)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(), output=True)
        
    def forward(self, x):
        # x shape: [Time, Batch, Features]
        # (DataLoader에서 [Batch, Time, Feat]로 나오지만, 학습 루프에서 permute하여 입력됨)
        
        # 뉴런의 초기 상태(전압) 초기화
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        # 시간 스텝별 출력을 저장할 리스트
        spk2_rec = []
        mem2_rec = []
        
        # 시간(Time Step) 반복 루프 (BPTT 핵심)
        # npx_trainer의 forward_pass와 동일한 로직
        for step in range(x.size(0)):
            cur1 = self.fc1(x[step])        # 1. 입력 -> FC1
            spk1, mem1 = self.lif1(cur1, mem1) # 2. FC1 출력 -> LIF1 (스파이크 발생 여부 결정)
            cur2 = self.fc2(spk1)           # 3. LIF1 스파이크 -> FC2
            spk2, mem2 = self.lif2(cur2, mem2) # 4. FC2 출력 -> LIF2 (최종 클래스 스파이크)
            
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)
            
        # 모든 시간 스텝의 출력을 스택으로 쌓아서 반환
        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)
