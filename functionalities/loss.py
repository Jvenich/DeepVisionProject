import torch


def l1_loss(x, y):
    return torch.mean(torch.abs(x-y))


def l2_loss(x, y):
    return torch.mean((x-y)**2)


def feat_loss(x, y, feat_model):
    return torch.mean(torch.abs(feat_model(x) - feat_model(y)))


def MMD_multiscale(x, y, device):
    x, y = x.to(device), y.to(device)
    xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2.*xx
    dyy = ry.t() + ry - 2.*yy
    dxy = rx.t() + ry - 2.*zz

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    for a in [0.2, 0.5, 0.9, 1.3]:
        XX += a**2 * (a**2 + dxx)**-1
        YY += a**2 * (a**2 + dyy)**-1
        XY += a**2 * (a**2 + dxy)**-1

    return torch.mean(XX + YY - 2.*XY)


def loss_max_likelihood(x, y, model, num_classses, sigma):
    jac = model.jacobian(run_forward=False)

    neg_log_like = ( 0.5 / sigma**2 * torch.sum((x[:, :num_classses] - y[:, :num_classses])**2, 1)
                     + 0.5 * torch.sum(x[:, num_classses:]**2, 1) - jac)

    return torch.mean(neg_log_like)


def loss_forward_mmd(out, y, num_classes):
    # Shorten output, and remove gradients wrt y, for latent loss
    output_block_grad = torch.cat((out[:, :num_classes],
                                   out[:, num_classes:].data), dim=1)
    y_short = torch.cat((y[:, :num_classes], y[:, num_classes:]), dim=1)

    l_forw_fit = 1. * l2_loss(out[:, :num_classes], y[:, :num_classes])
    l_forw_mmd = 50.  * torch.mean(MMD_multiscale(output_block_grad, y_short))

    return l_forw_fit, l_forw_mmd

def loss_backward_mmd(x, y, model):
    x_samples = model(y, rev=True)
    MMD = MMD_multiscale(x, x_samples)
    #if c.mmd_back_weighted:
     #   MMD *= torch.exp(- 0.5 / c.y_uncertainty_sigma**2 * losses.l2_dist_matrix(y, y))
    return 500. * torch.mean(MMD)

def loss_reconstruction(out_y, y, x, model, num_classes, add_z_noise, add_y_noise, ndim_pad_zy=False):
    cat_inputs = [out_y[:, num_classes:] + add_z_noise * noise_batch(len(out_y) - num_classes)]
   # if ndim_pad_zy:
   #     cat_inputs.append(out_y[:, c.ndim_z:-c.ndim_y] + c.add_pad_noise * noise_batch(c.ndim_pad_zy))
    cat_inputs.append(out_y[:, :num_classes] + add_y_noise * noise_batch(num_classes))

    x_reconstructed = model(torch.cat(cat_inputs, 1), rev=True)
    return 1. * l2_loss(x_reconstructed, x)


def noise_batch(batch_size, ndim, device):
    return torch.randn(batch_size, ndim).to(device)
