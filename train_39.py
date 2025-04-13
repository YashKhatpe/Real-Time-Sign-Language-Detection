import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torchvision import transforms
import videotransforms
import numpy as np
from configs import Config
from pytorch_i3d import InceptionI3d
from dataset.nslt_dataset import NSLT as Dataset

# Set up environment
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Configuration
parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-save_model', type=str)
parser.add_argument('-root', type=str)
parser.add_argument('--num_class', type=int)
args = parser.parse_args()

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def run(configs, mode='rgb', root=None, train_split='', save_model='', weights=None):
    print(configs)

    # Dataset setup
    train_transforms = transforms.Compose([
        videotransforms.RandomCrop(224),
        videotransforms.RandomHorizontalFlip(),
    ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(train_split, 'train', root, mode, train_transforms)
    val_dataset = Dataset(train_split, 'test', root, mode, test_transforms)

    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=configs.batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=configs.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=False
    )
    dataloaders = {'train': dataloader, 'test': val_dataloader}

    # Model setup
    i3d = InceptionI3d(2000, in_channels=3, dropout_keep_prob=0.8)  # Start with 2000 class model
    
    if weights:
        print(f'Loading pretrained weights from {weights}')
        i3d.load_state_dict(torch.load(weights))
    
    # Replace final layer for 39 classes
    num_classes = dataset.num_classes
    i3d.replace_logits(num_classes)
    
    # Freeze layers if needed (optional)
    # for param in i3d.parameters():
    #     param.requires_grad = False
    # for param in i3d.logits.parameters():
    #     param.requires_grad = True

    i3d = nn.DataParallel(i3d.cuda())

    # Optimizer setup
    optimizer = optim.Adam(
        i3d.parameters(),
        lr=configs.init_lr,
        weight_decay=configs.adam_weight_decay
    )
    
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 'min', patience=3, factor=0.5
    # )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=10
    )

    # Training loop
    best_val_score = 0
    steps = 0
    num_steps_per_update = configs.update_per_step
    
    for epoch in range(configs.max_epochs):
        print(f'Epoch {epoch+1}/{configs.max_epochs}')
        
        for phase in ['train', 'test']:
            i3d.train(phase == 'train')
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, labels, _) in enumerate(dataloaders[phase]):
                if inputs is None:  # Skip invalid data
                    continue
                
                inputs = inputs.cuda()
                labels = labels.max(dim=2)[0].argmax(dim=1).cuda()  # Convert to class indices
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = i3d(inputs)
                    outputs = outputs.mean(dim=2)  # Temporal average pooling
                    loss = F.cross_entropy(outputs, labels)
                
                # Backward pass
                if phase == 'train':
                    loss.backward()
                    if (batch_idx + 1) % num_steps_per_update == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                        steps += 1
                
                # Statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                if batch_idx % 10 == 0:
                    print(f'{phase} Batch {batch_idx}/{len(dataloaders[phase])} Loss: {loss.item():.4f}')
            
            # Epoch statistics
            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = 100. * correct / total
            
            print(f'{phase} Epoch {epoch} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%')
            
            # Validation and checkpointing
            if phase == 'test':
                scheduler.step(epoch_loss)
                
                if epoch_acc > best_val_score:
                    best_val_score = epoch_acc
                    torch.save({
                        'epoch': epoch,
                        'state_dict': i3d.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, f'{save_model}/best_39class_checkpoint.pth')
                    
                # Save periodic checkpoints
                if epoch % 5 == 0:
                    torch.save({
                        'epoch': epoch,
                        'state_dict': i3d.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, f'{save_model}/checkpoint_{epoch}.pth')

if __name__ == '__main__':
    # Path configuration
    config = {
        'mode': 'rgb',
        'root': {'word': 'C:/Users/Yash Khatpe/Desktop/Yash/Mini Project/I3d_model/I3d_model/WLASL/I3D/data/videos1_preprocessed'},
        'save_model': 'checkpoints/',
        'train_split': 'preprocess/nslt_39.json',
        'weights': 'archived/asl2000/FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt',
        'config_file': 'configfiles/asl39.ini'
    }
    print(config)
    
    # Load config
    configs = Config(config['config_file'])
    
    # Add max_epochs to configs if not present
    if not hasattr(configs, 'max_epochs'):
        configs.max_epochs = 50  # Default value
    
    run(configs=configs, mode=config['mode'],root=config['root'], train_split=config['train_split'], save_model=config['save_model'], weights=config['weights']
    )