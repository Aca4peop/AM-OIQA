import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn

import torch.backends.cudnn as cudnn
from tqdm import tqdm

import dataset_vp
import subprocess
from AMOIQA import AMOIQA,VASIQA_without_VAR
from utils import performance_fit, seed_torch, getlogger,PLCCLoss
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def parse_args():

    """Parse input arguments. """
    parser = argparse.ArgumentParser(description="360 Image Quality Assessment")
    parser.add_argument('--num_epochs', help='Maximum number of training epochs.', default=200, type=int)
    parser.add_argument('--batch_size', help='Batch size.', default=8, type=int)
    parser.add_argument('--lr', type=float, default=1e-4)#1e-4
    parser.add_argument('--snapshot', help='Path of model snapshot.', default='checkpoint', type=str)
    parser.add_argument('--train_database', default='CVIQ', type=str, help='CVIQ|OIQA|IQA-ODI|MVAQD|OIQ10K')  
    parser.add_argument('--model', default='AMOIQA', type=str, help='Model name for experiment, default is AMIQA.')
    args = parser.parse_args()

    return args


def get_free_gpu_memory():
    """Return the amount of available GPU memory in bytes"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
                                stdout=subprocess.PIPE)
        output = result.stdout.decode('utf-8').strip()
        return int(output)
    except Exception as e:
        print(f"Error getting GPU free memory: {e}")
        return None


def wait_for_gpu_memory(min_free_memory_gb=12):
    """Wait until the GPU has at least `min_free_memory_gb` GB of free memory."""
    min_free_memory_bytes = min_free_memory_gb * 1024 
    print("Checking GPU memory...")
    while True:
        free_memory = get_free_gpu_memory()
        if free_memory is not None and free_memory > min_free_memory_bytes:
            print(f"GPU free memory is {free_memory}.")
            print("GPU memory is sufficient. Continuing with the code...")
            break 
        else:
            time.sleep(50)  




if __name__ == '__main__':
    
    args = parse_args()

    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    lr = args.lr

    snapshot = args.snapshot
    model_name = args.model

    # default is database 'CVIQ','OIQ10K','OIQA', select one for train and then test on other databases.
    train_database = args.train_database
    test_databases =['CVIQ','OIQ10K','OIQA']

    test_databases.remove(train_database)
    csv_train = './data/' + train_database + '/'
    model_save_path = os.path.join(snapshot, model_name,f"{train_database}_crosstrain")  
    wait_for_gpu_memory(12)


    if not os.path.exists(os.path.join(model_save_path)):
        os.makedirs(os.path.join(model_save_path))
    logger = getlogger(os.path.join(model_save_path))

    seed_torch(2024)
    
    model = AMOIQA(pretrained=True).to("cuda")
    filename_train = csv_train + train_database + '.csv'
    #cmp dataset
    train_dataset = dataset_vp.Dataset(csv_path=filename_train, database=train_database, train=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                                num_workers=8)
    criterion1 = PLCCLoss().to("cuda")
    criterion2 = nn.MSELoss().to("cuda")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(0.9, 0.95), weight_decay=5e-4)

    best_all = np.zeros([len(test_databases), 4])
    for epoch in range(num_epochs):
        lr = optimizer.param_groups[0]['lr']
        # train
        model.train()
        batch_losses = []
        session_start_time = time.time()
        pbar = tqdm(train_loader)

        for step, (vp_list, mos) in enumerate(pbar):
            for idx in range(len(vp_list)):
                vp_list[idx] = vp_list[idx].to("cuda")
            mos = mos[:, np.newaxis]
            mos = mos.to("cuda").view(-1)
            mos_output = model(vp_list)
            loss = criterion1(mos_output.view(-1), mos)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            pbar.set_description(
                'train Epoch [%d/%d], loss: %4f, lr: %6f'
                % (epoch + 1, num_epochs, loss.data.item(), lr))
        #sheduler.step()
        avg_loss = sum(batch_losses) / len(train_loader)
        logger.info(
            f'Train epoch: {epoch + 1}\t'
            f'loss {avg_loss:.4f}\t\n'
            f'learning rate: {lr}')
        print('Epoch %d averaged training loss: %.4f ' % (epoch + 1, avg_loss),end='\n')

        # Test
        model.eval()
        for n,db in enumerate(test_databases):
            csv_test = f'./data/{db}/{db}.csv' 
            test_dataset = dataset_vp.Dataset(csv_path=csv_test, database=db, train=False)   
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=8, shuffle=False, num_workers=8, persistent_workers=False)
            with torch.no_grad():
                y_output = []
                y_test = []
                for step1, (vp_list, mos) in enumerate(tqdm(test_loader)):
                    for idx in range(len(vp_list)):
                        vp_list[idx] = vp_list[idx].to("cuda")
                    y_test.append(mos.view(-1).numpy())
        
                    outputs = model(vp_list)
                    y_output.append(outputs.cpu().view(-1).numpy())
                y_test = np.concatenate(y_test)
                y_output = np.concatenate(y_output)
                test_PLCC, test_SRCC, test_KRCC, test_RMSE = performance_fit(y_test, y_output)
                print("Test results on {}: SROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(db,test_SRCC, test_PLCC, test_RMSE))
                logger.info("Test results on {}: SROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}\n".format(db,test_SRCC, test_PLCC, test_RMSE,
                                                                                                  test_PLCC, test_RMSE))
                
                if abs(test_SRCC)+abs(test_PLCC)  > best_all[n,0]+best_all[n,1]:
                    print("Update best model using best_val_criterion in epoch {}, cross {}".format(epoch + 1,db))
                    torch.save(model.state_dict(), os.path.join(model_save_path, f'cross_{db}_best_model.pth'))
                            
                    best_all[n,:] = [abs(test_SRCC), abs(test_PLCC), abs(test_KRCC), test_RMSE]
                    logger.info(f"\t\tUpdata best test on {db}: SROCC: {test_SRCC:.4f} \t\tPLCC: {test_PLCC:.4f} \t\tKROCC: {test_KRCC:.4f} \t\tRMSE: {test_RMSE:.4f}\n")
            test_loader = None  # 释放测试集的DataLoader内存
    logger.info("Best output: SROCC \tPLCC \tKROCC \tRMSE \n")
    print("Best results:")
    for n in range(len(test_databases)):
        logger.info("{},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(test_databases[n],best_all[n][0], best_all[n][1], best_all[n][2], best_all[n][3]))
        print("{},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(test_databases[n],best_all[n][0], best_all[n][1], best_all[n][2], best_all[n][3]))
