from datasets import ImageDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import sample_random_batch
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

train_dataset = ImageDataset('./datasets/img_align_celeba/train', transform=transform)
valid_dataset = ImageDataset('./datasets/img_align_celeba/valid', transform=transform)

print('Training samples: %d' % len(train_dataset))
print('Validation samples: %d' % len(valid_dataset))

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

pbar = tqdm(total=len(train_loader))
for batch in enumerate(train_loader):
    pbar.update()
pbar.close()
