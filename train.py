import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 


def train(num_epochs, train_loader, test_loader, optimizer, vae, verbose=True, print_every_epochs=10):
    """
    Training Boiler Plate code. Returns the trained vae and list of loss values for each epoch.
    """
    loss_values = []
    for epoch in range(num_epochs):
        train_loss_value = 0
        val_loss_val = 0
        kl_divergence_val = 0
        for i, data in enumerate(train_loader):
            inputs, _ = data
            input_dims = inputs.shape[-1] * inputs.shape[-2]
            inputs = inputs.view(-1, input_dims).to(device)
            optimizer.zero_grad()
            mu_dec, sigma_dec, mu_enc, sigma_enc = vae(inputs)
            p_x = torch.distributions.Normal(mu_dec, sigma_dec**2)
            log_likelihood = p_x.log_prob(inputs).sum(-1).mean()
            kl_divergence = torch.distributions.kl_divergence(torch.distributions.Normal(mu_enc, sigma_enc), torch.distributions.Normal(0,1)).sum(-1).mean()
            
            loss = -log_likelihood + kl_divergence #-elbo
            
            loss = loss.to(device)
            loss.backward()
            optimizer.step()
            train_loss_value += loss.item()
        # Validate model on test loader
        for j, data_val in enumerate(test_loader):
            with torch.no_grad():
                inputs_val, _ = data_val
                input_dims = inputs_val.shape[-1] * inputs_val.shape[-2]
                inputs_val = inputs_val.view(-1, input_dims).to(device)

                mu_dec, sigma_dec, mu_enc, sigma_enc = vae(inputs)
                p_x = torch.distributions.Normal(mu_dec, sigma_dec**2)
                log_likelihood = p_x.log_prob(inputs).sum(-1).mean()

                kl_divergence = torch.distributions.kl_divergence(torch.distributions.Normal(mu_enc, sigma_enc), torch.distributions.Normal(0,1)).sum(-1).mean()
                loss_val = -log_likelihood + kl_divergence
                val_loss_val += loss_val.item()
                kl_divergence_val += kl_divergence.item()
        if verbose:
            if (epoch+1)%print_every_epochs==0:
                print(f"Epoch : {epoch+1}, Train Loss Value: {train_loss_value}, Val Loss Value:{val_loss_val}, KL Div value:{kl_divergence_val}")   
        loss_values.append(val_loss_val)

    return vae, loss_values