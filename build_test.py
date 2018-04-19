from models import CompletionNetwork
from torch.autograd import Variable
import torch

model_c = CompletionNetwork(input_shape=(3, 256, 256))
print(model_c)

x = Variable(torch.rand(1, 3, 256, 256))
outs = model_c(x)
print(outs.shape)
