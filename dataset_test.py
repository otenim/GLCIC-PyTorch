from datasets import ImageDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

train_dataset = ImageDataset('./datasets/img_align_celeba/train', transform=transform)
valid_dataset = ImageDataset('./datasets/img_align_celeba/valid', transform=transform)

print('Training samples: %d' % len(train_dataset))
print('Validation samples: %d' % len(valid_dataset))
print(train_dataset[0])
print(train_dataset[0].shape)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
cnt = 5
for batch in train_loader:
    print(batch)
    cnt -= 1
    if cnt <= 0:
        break
