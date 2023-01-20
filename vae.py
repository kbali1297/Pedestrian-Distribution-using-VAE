import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 

class Encoder(nn.Module):
    def __init__(self, input_dims, hidden_dims, latent_dims, num_layers=1) -> None:
        super().__init__()

        self.num_layers = num_layers
        for i in range(num_layers):
            if i==0:
                self.add_module(f'Linear{i+1}', nn.Linear(input_dims, hidden_dims))
            else:
                self.add_module(f'Linear{i+1}', nn.Linear(hidden_dims, hidden_dims)) 

        self.linear_mu = nn.Linear(hidden_dims, latent_dims)
        self.linear_sigma = nn.Linear(hidden_dims, latent_dims)

    def forward(self, x):
        x = x.view(-1, x.shape[1])
        
        for i in range(self.num_layers):
            x = F.relu(getattr(self, f'Linear{i+1}')(x))

        mu = self.linear_mu(x)
        sigma = torch.exp(self.linear_sigma(x))

        return mu, sigma

class Decoder(nn.Module):
    def __init__(self, input_dims, hidden_dims, latent_dims, num_layers=1) -> None:
        super().__init__()

        self.num_layers = num_layers
        for i in range(num_layers):
            if i==0:
                self.add_module(f'Linear{i+1}', nn.Linear(latent_dims, hidden_dims)) 
            else:
                self.add_module(f'Linear{i+1}', nn.Linear(hidden_dims, hidden_dims)) 

        self.linear_mu = nn.Linear(hidden_dims, input_dims)
        self.sigma = nn.parameter.Parameter(torch.ones(1))

    def forward(self, x):
        #print(f"Inside Decoder: {x.shape}")
        x = x.view(-1, x.shape[1])

        for i in range(self.num_layers):
            x = F.relu(getattr(self, f'Linear{i+1}')(x))
        
        mu = torch.tanh(self.linear_mu(x))    
        #mu = self.linear_mu(x)
        return mu

class VAE(nn.Module):
    def __init__(self, input_dims=784, hidden_dims=256, latent_dims=2, encoder_layers=1, decoder_layers=1) -> None:
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.latent_dims = latent_dims

        ##Encoder
        self.encoder = Encoder(input_dims, hidden_dims, latent_dims, num_layers=encoder_layers).to(device)

        ##Decoder 
        self.decoder = Decoder(input_dims, hidden_dims, latent_dims, num_layers=decoder_layers).to(device)
        self.N = torch.distributions.Normal(0,1)

    def forward(self, x):
        mu_enc, sigma_enc = self.encoder(x)

        # Reparametrization z = mu + sigma * epsilon, *: element wise product and epsilon ~ N(0,1)
        z = mu_enc + sigma_enc * self.N.sample(mu_enc.shape).to(device)
        # Output the mean of p(x|z)
        mu_dec = self.decoder(z)
        return mu_dec, self.decoder.sigma * torch.ones_like(mu_dec), mu_enc, sigma_enc