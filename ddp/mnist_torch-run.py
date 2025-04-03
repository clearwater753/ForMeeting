# # 单机4卡
# torchrun --nproc_per_node=4 script.py --epochs 2

# 多机多卡（例如2台机器，每台4卡）
# 机器1（主节点）:
# torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr="10.57.23.164" script.py --epochs 2
# # 机器2:
# torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr="10.57.23.164" script.py --epochs 2

import os
from datetime import datetime
import argparse
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
    
def train():
    # 从环境变量获取rank信息（torchrun自动设置）
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    # 初始化分布式环境（torchrun会自动处理）
    dist.init_process_group(backend='nccl')
    
    torch.manual_seed(0)
    model = ConvNet()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    
    batch_size = 100
    criterion = nn.CrossEntropyLoss().cuda(local_rank)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model,
                                              device_ids=[local_rank],
                                              output_device=local_rank)
    
    # Data loading
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                             train=True,
                                             transform=transforms.ToTensor(),
                                             download=True)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=local_rank
    )
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=0,
                                             pin_memory=True,
                                             sampler=train_sampler)

    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)  # 重要：设置epoch保证shuffle正确
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 100 == 0 and local_rank == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, 
                    args.epochs, 
                    i + 1, 
                    total_step,
                    loss.item())
                )
    
    if local_rank == 0:
        print("Training complete in: " + str(datetime.now() - start))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=2, type=int, 
                      help='number of total epochs to run')
    global args
    args = parser.parse_args()
    
    train()  # 直接调用train函数

if __name__ == '__main__':
    main()