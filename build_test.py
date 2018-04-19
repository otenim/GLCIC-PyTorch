from models import CompletionNetwork, LocalDiscriminator
from torch.autograd import Variable
import torch

model_c = CompletionNetwork(input_shape=(3, 256, 256))
model_ld = LocalDiscriminator(input_shape=(3, 128, 128))
print(model_c)
print(model_ld)

x_c = Variable(torch.rand(1, 3, 256, 256))
x_ld = Variable(torch.rand(1, 3, 128, 128))
outs_c = model_c(x_c)
outs_ld = model_ld(x_ld)
print(outs_c.shape)
print(outs_ld.shape)
