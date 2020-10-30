"""The PyTorch benchmark script for custom models

The benchmark script to compare custom models for research purposes. 
It is easy to add the modified models into the library "custom_models" and 
modify the "__init__.py" file. 

The benchmark script's adventure is allowed to check time-consuming in any part 
of the custom model by adding "self.start_time" and "self.end_time." 
Please check the examples in "custom_models" for more details for the implementation. 

The current benchmark script support only testing inference time on both CPU and GPU

Author: Weikun Han <weikunhan@gmail.com>

Reference: 
- https://github.com/JunhongXu/pytorch-benchmark-volta/blob/master/benchmark_models.py

Please install: 
- pip install pandas
- pip install psutil
"""

import argparse
import json
import os
import time
import torch 
import pandas as pd
import platform
import psutil
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import custom_models

CONFIG = json.loads(open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 
                                      'config_benchmark_custom_models.json')).read())

class RandomDataset(Dataset):

    def __init__(self, ndata, imsize):
        self.len = ndata
        self.images = torch.randn(ndata, 3, imsize[0], imsize[1])

    def __getitem__(self, index):

        return self.images[index]

    def __len__(self):

        return self.len


def inference(random_loader, model, device, warm_up):
    durations = []
    model.eval()

    with torch.no_grad():
        for i , (images) in enumerate(random_loader):
            images = images.to(device)
            model(images)

            if i >= warm_up:
                durations.append((model.end_time - model.start_time) * 1000)

    return durations

def record_environment(): 
    system_configs = str(platform.uname()) + '\n'
    system_configs += str(psutil.cpu_freq()) + '\n'
    system_configs += str(psutil.virtual_memory())
    
    if torch.cuda.is_available():
        gpu_configs = 'gpu_name: ' + str(torch.cuda.get_device_name(0)) + '\n'
        gpu_configs += 'cuda_version: ' + str(torch.version.cuda) + '\n'
        gpu_configs += 'cudnn_version: ' + str(torch.backends.cudnn.version())
    else:
        gpu_configs = 'warning: not available, no GPUs detected'

    with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 
                           'benchmarking_environment_info.txt'), "w") as f:
        f.writelines('System environment for benchmarking models "{}"\n'.format(CONFIG['model_list']))
        f.writelines('\n')
        f.writelines('the system config info: \n')
        f.writelines(system_configs)
        f.writelines('\n')
        f.writelines('\n')
        f.writelines('the GPUs config info: \n')
        f.writelines(gpu_configs)
        f.writelines('\n')

def main():
    benchmark_inference_time_dict = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Start benchmarking custom models: {}'.format(CONFIG['model_list']))
    print('the config info: {}'.format(CONFIG['image_size']))

    record_environment()
    
    random_dataset = RandomDataset(
        args.batch_size * (args.warm_up + args.test_size), CONFIG['image_size'])
    random_loader = DataLoader(dataset=random_dataset, batch_size=args.batch_size, 
                               shuffle=False)
    cudnn.benchmark = True

    for model_name in CONFIG['model_list']:
        model = custom_models.__dict__[model_name]()

        if args.gpu_size > 1:
            model = nn.DataParallel(model, device_ids=range(args.gpu_size))

        model = model.to(device)
        durations = inference(random_loader, model, device,  args.warm_up)
        benchmark_inference_time_dict[model_name] = durations
        del model

   
    print('The number of {} tests run for each model:'.format(args.test_size))

    for key, value in benchmark_inference_time_dict.items():
        print('=> The model name: {}'.format(key))
        print('=> Average time consuming: {}ms'.format(sum(value) / args.test_size))
        print('=> FPS: {}'.format(args.test_size / (sum(value) / 1000)))
        print('\n')

    inference_result_df = pd.DataFrame(benchmark_inference_time_dict)
    inference_result_df.to_csv(
        os.path.join(os.path.abspath(os.path.dirname(__file__)), 
                     'benchmarking_inference_time_res.csv'), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch benchmark custom models')
    parser.add_argument('--warm-up', type=int, default=5,
                        help="number of tests run for warning up")
    parser.add_argument('--test-size', type=int, default=100,
                        help="number of benchmark tests run")
    parser.add_argument('--batch-size', type=int, default=1, 
                        help='number of batch size for benchmarking')
    parser.add_argument('--gpu-size', type=int, default=1, 
                        help='number of GPUs to use')
    args = parser.parse_args()

    main()