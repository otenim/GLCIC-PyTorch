from models import CompletionNetwork, LocalDiscriminator, GlobalDiscriminator, ContextDiscriminator
import torch

model_c = CompletionNetwork(input_shape=(3, 256, 256))
model_ld = LocalDiscriminator(input_shape=(3, 128, 128))
model_gd = GlobalDiscriminator(input_shape=(3, 256, 256))
model_d = ContextDiscriminator(local_input_shape=(3, 128, 128), global_input_shape=(3, 256, 256))
print(model_c)
print(model_ld)
print(model_gd)
print(model_d)

x_c = torch.rand(1, 3, 256, 256)
x_ld = torch.rand(1, 3, 128, 128)
x_gd = torch.rand(1, 3, 256, 256)
outs_c = model_c(x_c)
outs_ld = model_ld(x_ld)
outs_gd = model_gd(x_gd)
outs_d = model_d([x_ld, x_gd])
print(outs_c.shape)
print(outs_ld.shape)
print(outs_gd.shape)
print(outs_d.shape)
