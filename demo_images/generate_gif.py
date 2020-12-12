import os
import subprocess

import matplotlib.pyplot as plt
import stheno.torch as stheno
import torch
import wbml.plot

from convcnp.architectures import SimpleConv
from convcnp.set_conv import ConvCNP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def to_numpy(x):
    """Convert a PyTorch tensor to NumPy."""
    return x.squeeze().detach().cpu().numpy()


convcnp = ConvCNP(learn_length_scale=True,
                  points_per_unit=64,
                  architecture=SimpleConv())
convcnp.to(device)
load_dict = torch.load('./saved_models/convcnp-matern/model_best.pth.tar')
convcnp.load_state_dict(load_dict['state_dict'])
convcnp.eval()

# Construct GP.
kernel = stheno.Matern52().stretch(0.25)
gp = stheno.GP(kernel)

# Sample function from GP and random permutation of data.
num_points = 200
rand_indices = torch.randperm(num_points)
x_all = torch.linspace(-2., 2., num_points)
y_all = gp(x_all).sample()

# Generate frames of GIF.
for context_size in range(1, 16):
    plt.figure(figsize=(8, 4))

    x_context = x_all[rand_indices][:context_size][None, :, None].to(device)
    y_context = y_all[rand_indices][:context_size][None, ...].to(device)

    # Make predictions with model.
    with torch.no_grad():
        y_mean, y_std = convcnp(x_context, y_context,
                                x_all[None, :, None].to(device))

    # Make predictions with oracle GP
    post = gp.measure | (gp(to_numpy(x_context)), to_numpy(y_context))
    gp_mean, gp_lower, gp_upper = post(gp(to_numpy(x_all))).marginals()

    # Plot context set.
    plt.scatter(to_numpy(x_context), to_numpy(y_context),
                label='Context Set', color='black')
    plt.plot(to_numpy(x_all), to_numpy(y_all), '--',
             label='Sampled function', color='gray', alpha=0.9)

    # Plot GP predictions.
    plt.plot(to_numpy(x_all), gp_mean, color='black', label='Oracle GP')
    plt.plot(to_numpy(x_all), gp_lower, color='black', alpha=0.4)
    plt.plot(to_numpy(x_all), gp_upper, color='black', alpha=0.4)

    # Plot model predictions.
    plt.plot(to_numpy(x_all), to_numpy(y_mean),
             label='ConvCNP', color='blue')
    plt.fill_between(to_numpy(x_all),
                     to_numpy(y_mean + 2 * y_std),
                     to_numpy(y_mean - 2 * y_std),
                     facecolor='tab:blue', alpha=0.2)
    plt.ylim(-3., 3)
    plt.axis('off')
    wbml.plot.tweak()
    plt.tight_layout()
    plt.savefig(f'frame{context_size:02d}.png', dpi=100)

# Build GIF.
subprocess.call(['convert',
                 '-delay', '40',
                 '-loop', '0',
                 'frame*.png', 'demo_images/convcnp.gif'])

# Clean up files.
for context_size in range(1, 16):
    os.remove(f'frame{context_size:02d}.png')
