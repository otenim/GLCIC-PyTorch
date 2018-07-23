from torch.nn.functional import mse_loss

def completion_network_loss(input, output, mask):
    return mse_loss(output * mask, input * mask)
