from torch.nn.functional import mse_loss

def completion_network_loss(input, output, mask):
    loss = 0.5 * ((input - output)**2 * mask).sum() / mask.sum()
    return loss
