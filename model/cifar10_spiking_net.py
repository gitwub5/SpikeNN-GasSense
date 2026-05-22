import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

class CIFAR10SpikingNet(nn.Module):
    def __init__(self, beta=0.9, spike_grad=None):
        super().__init__()
        
        if spike_grad is None:
            spike_grad = surrogate.fast_sigmoid()
            
        # Block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.mp1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        # Block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.mp2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        # Block 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.mp3 = nn.MaxPool2d(2)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        # Fully connected block
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(1024, 10)
        self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad, output=True)
        
    def forward(self, x, num_steps):
        """
        x: [Batch, Channels, Height, Width] (static image)
        num_steps: Time steps to simulate
        """
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()
        
        # Record final layer spikes and membrane potentials
        spk5_rec = []
        mem5_rec = []
        
        for step in range(num_steps):
            # Block 1
            cur1 = self.mp1(self.conv1(x))
            spk1, mem1 = self.lif1(cur1, mem1)
            
            # Block 2
            cur2 = self.mp2(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)
            
            # Block 3
            cur3 = self.mp3(self.conv3(spk2))
            spk3, mem3 = self.lif3(cur3, mem3)
            
            # FC Block
            cur4 = self.fc1(self.flatten(spk3))
            spk4, mem4 = self.lif4(cur4, mem4)
            
            cur5 = self.fc2(spk4)
            spk5, mem5 = self.lif5(cur5, mem5)
            
            spk5_rec.append(spk5)
            mem5_rec.append(mem5)
            
        return torch.stack(spk5_rec, dim=0), torch.stack(mem5_rec, dim=0)
