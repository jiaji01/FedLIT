import torch
from torch import nn
from torch.nn import functional as F


def sample_noise(num, dim, device=None) -> torch.Tensor:
    return torch.randn(num, dim, device=device)

def _weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.)

'''
VAE1 input: 56
'''
class Encoder1(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.h1_nchan = 64
        self.conv12 = nn.Sequential(   # 64*14*14
                nn.Conv2d(1, self.h1_nchan, kernel_size=4, stride=4, padding=0),
                nn.LeakyReLU(.1, inplace=True)
        )
        self.h2_nchan = 128
        self.conv1 = nn.Sequential(   # 128*7*7
                nn.Conv2d(self.h1_nchan, self.h2_nchan, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(self.h2_nchan),
                nn.LeakyReLU(.1, inplace=True)
        )
      
        # linearize
        self.h4_dim = 1024
        self.fc2 = nn.Sequential(
                nn.Linear(7 * 7 * self.h2_nchan, self.h4_dim),
                nn.BatchNorm1d(self.h4_dim),
                nn.LeakyReLU(.1, inplace=True)
        )
        self.fc1 = nn.Sequential(
                nn.Linear(self.h4_dim, latent_dim),
                nn.BatchNorm1d(latent_dim),
                nn.LeakyReLU(.1, inplace=True)
        )

    def forward(self, x):
        x = self.conv12(x)
        x = self.conv1(x).view(-1, 7 * 7 * self.h2_nchan)
        x = self.fc2(x)
        z = self.fc1(x)
        return z


    

class Decoder1(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

        self.h1_dim = 1024
        self.fc1 = nn.Sequential(
            nn.Linear(self.latent_dim, self.h1_dim),
            nn.BatchNorm1d(self.h1_dim),
            nn.ReLU(inplace=True)
        )

        
        self.h2_nchan = 128
        h2_dim = 7 * 7 * self.h2_nchan
        self.fc2 = nn.Sequential(
            nn.Linear(self.h1_dim, h2_dim),
            nn.BatchNorm1d(h2_dim),
            nn.ReLU(inplace=True)
        )
        self.h3_nchan = 64
        self.conv1 = nn.Sequential(  # 64*14*14
            nn.ConvTranspose2d(self.h2_nchan, self.h3_nchan,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.h3_nchan),
            nn.ReLU(inplace=True)
        )

        self.conv12 = nn.Sequential(  # 1*56*56
            nn.ConvTranspose2d(self.h3_nchan, 1, kernel_size=4, stride=4, padding=0),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x).view(-1, self.h2_nchan, 7, 7)
        x = self.conv1(x)
        x = self.conv12(x)
        return x
    

class VAE1(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

        self.enc = Encoder1(self.latent_dim)
        self.dec = Decoder1(self.latent_dim)

        self.apply(_weights_init)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, data):
        z = self.enc(data)
        recon = self.dec(z)
        return z, recon
    
    


'''
VAE2 input: 28
'''

class Encoder2(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.h1_nchan = 64
        self.conv22 = nn.Sequential(   # 64*14*14
                nn.Conv2d(1, self.h1_nchan, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(.1, inplace=True)
        )
        self.h2_nchan = 128
        self.conv1 = nn.Sequential(   # 128*7*7
                nn.Conv2d(self.h1_nchan, self.h2_nchan, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(self.h2_nchan),
                nn.LeakyReLU(.1, inplace=True)
        )


        # linearize
        self.h4_dim = 1024
        self.fc2 = nn.Sequential(
                nn.Linear(7 * 7 * self.h2_nchan, self.h4_dim),
                nn.BatchNorm1d(self.h4_dim),
                nn.LeakyReLU(.1, inplace=True)
        )

        self.fc1 = nn.Sequential(
                nn.Linear(self.h4_dim, latent_dim),
                nn.BatchNorm1d(latent_dim),
                nn.LeakyReLU(.1, inplace=True)
        )

    def forward(self, x):
        x = self.conv22(x)
        x = self.conv1(x).view(-1, 7 * 7 * self.h2_nchan)
        x = self.fc2(x)
        z = self.fc1(x)
        return z


    

class Decoder2(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.h1_dim = 1024
        self.fc1 = nn.Sequential(
            nn.Linear(self.latent_dim, self.h1_dim),
            nn.BatchNorm1d(self.h1_dim),
            nn.ReLU(inplace=True)
        )
        
        self.h2_nchan = 128
        h2_dim = 7 * 7 * self.h2_nchan
        self.fc2 = nn.Sequential(
            nn.Linear(self.h1_dim, h2_dim),
            nn.BatchNorm1d(h2_dim),
            nn.ReLU(inplace=True)
        )
        self.h3_nchan = 64
        self.conv1 = nn.Sequential(  # 64*14*14
            nn.ConvTranspose2d(self.h2_nchan, self.h3_nchan,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.h3_nchan),
            nn.ReLU(inplace=True)
        )
        self.conv22 = nn.Sequential(  # 1*28*28
            nn.ConvTranspose2d(self.h3_nchan, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x).view(-1, self.h2_nchan, 7, 7)
        x = self.conv1(x)
        x = self.conv22(x)
        return x
    

class VAE2(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

        self.enc = Encoder2(self.latent_dim)
        self.dec = Decoder2(self.latent_dim)

        self.apply(_weights_init)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, data):
        z = self.enc(data)
        recon = self.dec(z)
        return z, recon
    
    


'''
VAE3 input: 42
'''

class Encoder3(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.h1_nchan = 64
        self.conv32 = nn.Sequential(   # 64*14*14
                nn.Conv2d(1, self.h1_nchan, kernel_size=3, stride=3, padding=0),
                nn.LeakyReLU(.1, inplace=True)
        )
        self.h2_nchan = 128
        self.conv1 = nn.Sequential(   # 128*7*7
                nn.Conv2d(self.h1_nchan, self.h2_nchan, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(self.h2_nchan),
                nn.LeakyReLU(.1, inplace=True)
        )


        # linearize
        self.h3_dim = 1024
        self.fc2 = nn.Sequential(
                nn.Linear(7 * 7 * self.h2_nchan, self.h3_dim),
                nn.BatchNorm1d(self.h3_dim),
                nn.LeakyReLU(.1, inplace=True)
        )

        self.fc1 = nn.Sequential(
                nn.Linear(self.h3_dim, latent_dim),
                nn.BatchNorm1d(latent_dim),
                nn.LeakyReLU(.1, inplace=True)
        )

    def forward(self, x):
        x = self.conv32(x)
        x = self.conv1(x).view(-1, 7 * 7 * self.h2_nchan)
        x = self.fc2(x)
        z = self.fc1(x)
        return z


    

class Decoder3(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.h1_dim = 1024
        self.fc1 = nn.Sequential(
            nn.Linear(self.latent_dim, self.h1_dim),
            nn.BatchNorm1d(self.h1_dim),
            nn.ReLU(inplace=True)
        )

        
        self.h2_nchan = 128
        h2_dim = 7 * 7 * self.h2_nchan
        self.fc2 = nn.Sequential(
            nn.Linear(self.h1_dim, h2_dim),
            nn.BatchNorm1d(h2_dim),
            nn.ReLU(inplace=True)
        )
        self.h3_nchan = 64
        self.conv1 = nn.Sequential(  # 64*14*14
            nn.ConvTranspose2d(self.h2_nchan, self.h3_nchan,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.h3_nchan),
            nn.ReLU(inplace=True)
        )
        self.conv32 = nn.Sequential(  # 1*42*42
            nn.ConvTranspose2d(self.h3_nchan, 1, kernel_size=3, stride=3, padding=0),
            nn.Sigmoid()
        )
        

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x).view(-1, self.h2_nchan, 7, 7)
        x = self.conv1(x)
        x = self.conv32(x)
        return x
    

class VAE3(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

        self.enc = Encoder3(self.latent_dim)
        self.dec = Decoder3(self.latent_dim)

        self.apply(_weights_init)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, data):
        z = self.enc(data)
        recon = self.dec(z)
        return z, recon
    
    


















