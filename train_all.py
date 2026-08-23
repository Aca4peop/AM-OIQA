import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from dataset_vp import get_dataset
import subprocess
from AMOIQA import AMOIQA
from utils import performance_fit, seed_torch, getlogger,PLCCLoss
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="360 Image Quality Assessment")
    parser.add_argument('--num_epochs', help='Maximum number of training epochs.', default=200, type=int)
    parser.add_argument('--batch_size', help='Batch size.', default=8, type=int)
    parser.add_argument('--lr', help='Learning rate.',type=float, default=1e-5)#1e-4
    parser.add_argument('--snapshot',   help='Path of model snapshot.', default='checkpoint', type=str)
    parser.add_argument('--cross_validation_index', dest='cross_validation_index',
                        help='The index of cross validation.', default=1, type=int)
    parser.add_argument('--model', default='TAGIQA', type=str,
                        help='Model name for experiment, default is TAGIQA')
    parser.add_argument('--database', default='CVIQ', help = 'The training database of the model:CVIQ|OIQA|IQA-ODI|MVAQD|OIQ10K' ,type=str)  
    args = parser.parse_args()
    return args
def trainer(num_epochs:int, train_loader:DataLoader, test_loader:DataLoader, model, criterion, optimizer, logger, model_save_path:str, cross, best:np.ndarray) -> np.ndarray:
    for epoch in range(num_epochs):

        # train
        lr = optimizer.param_groups[0]['lr']
        model.train()
        batch_losses = []
        session_start_time = time.time()
        pbar = tqdm(train_loader)
        for step, (vp_list, mos) in enumerate(pbar):

            # move data to GPU
            for idx in range(len(vp_list)):
                vp_list[idx] = vp_list[idx].to("cuda")
            mos = mos[:, np.newaxis]
            mos = mos.to("cuda").view(-1)

            # inference and backward the loss
            mos_output = model(vp_list)
            loss = criterion(mos_output.view(-1), mos)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            pbar.set_description(
                '%d th train Epoch [%d/%d], loss: %4f, lr: %6f'
                % (i + 1, epoch + 1, num_epochs, loss.data.item(), lr))
            
        #sheduler.step()
        avg_loss = sum(batch_losses) / len(train_loader)
        logger.info(
            f'Train epoch: {epoch + 1}\t'
            f'loss {avg_loss:.4f}\t\n'
            f'learning rate: {lr}')
        print('Epoch %d averaged training loss: %.4f ' % (epoch + 1, avg_loss),end=' ')

        # Test
        model.eval()
        y_output = []
        y_test = []

        with torch.no_grad():
            for step1, (vp_list, mos) in enumerate(test_loader):

                # move data to GPU
                for idx in range(len(vp_list)):
                    vp_list[idx] = vp_list[idx].to("cuda")
                y_test.append(mos.view(-1).numpy())

                # inference
                outputs = model(vp_list)
                y_output.append(outputs.cpu().view(-1).numpy())
            y_test = np.concatenate(y_test)
            y_output = np.concatenate(y_output)
            test_PLCC, test_SRCC, test_KRCC, test_RMSE = performance_fit(y_test, y_output)
            print("Test results: SROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(test_SRCC, test_PLCC, test_RMSE, test_PLCC, test_RMSE))

            # check if the current model is the best model
            if test_SRCC+test_PLCC > best[0]+best[2]:
                print("Update best model using best_val_criterion in epoch {}".format(epoch + 1))
                torch.save(model.state_dict(), os.path.join(model_save_path, str(cross), 'best_model.pth'))
                
                # update the best results
                best[0:4] = [test_SRCC, test_KRCC, test_PLCC, test_RMSE]

            print("Best results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(best[0], best[1],
                                                                                                best[2], best[3]))
            logger.info(
                f"\nepoch: {epoch + 1}\t Best SROCC :{best[0]:.4f} \tBest PLCC :{best[2]:.4f}\tBest KROCC :{best[1]:.4f}\tBest RMSE :{best[3]:.4f}\n "
                f"\t\ttest : SROCC: {test_SRCC:.4f} \t\tPLCC: {test_PLCC:.4f} \t\tKROCC: {test_KRCC:.4f} \t\tRMSE: {test_RMSE:.4f}\n")
    return best



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
    database = args.database
    cross = args.cross_validation_index



    best_all = np.zeros([5, 4])
    csv_path = './data/' + database + '/'
    model_save_path = os.path.join(snapshot, args.model, database, 'batch=' + str(batch_size))  
    #wait_for_gpu_memory()


    if not os.path.exists(os.path.join(model_save_path, str(cross))):
        os.makedirs(os.path.join(model_save_path, str(cross)))
    logger = getlogger(os.path.join(model_save_path, str(cross)))
    for i in range(0,5):

        seed_torch(2023)
        cross = i+1 
        # create model save path and log directory
        if not os.path.exists(os.path.join(model_save_path, str(cross))):
            os.makedirs(os.path.join(model_save_path, str(cross)))
       
        # create model
        model = AMOIQA(backbone="swinv2_tiny_patch4_window8_256",pretrained=True).to("cuda")
        filename_train = csv_path + database + '_trainc_' + str(i + 1) + '.csv'
        filename_test = csv_path + database + '_testc_' + str(i + 1) + '.csv'

        # prepare dataset and dataloader
        train_dataset = get_dataset(csv_path=filename_train, database=database, train=True)
        test_dataset = get_dataset(csv_path=filename_test, database=database, train=False)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,num_workers=8)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=4, shuffle=False, num_workers=8)
        # define loss function and optimizer
        criterion = PLCCLoss().to("cuda")
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(0.9, 0.95), weight_decay=1e-3)

        # print the number of trainable parameters
        param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Number of trainable parameters: {:.4f}M".format(param_num/1e6))
        print("Ready to train network")

        # train and evaluate the model, and save the best model to the disk
        best = np.zeros(4)
        best = trainer(num_epochs,train_loader,test_loader,model,criterion, optimizer,logger,model_save_path,cross,best)
        best_all[i, :] = best
        print("The best Val results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(best[0], best[1], 
                                                                                                  best[2], best[3]))
    best_mean = np.mean(best_all, 0)
    print("The mean val results: {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(best_mean[0], best_mean[1],
                                                                        best_mean[2], best_mean[3]))
                                                                                              
    logger.info("Best output: SROCC \tPLCC \tKROCC \tRMSE \n")
    for i in range(0,5):
        logger.info("Fold {:d}:{:.4f}\t {:.4f}\t {:.4f}\t {:.4f}\t".format(i,
                                                                best_all[i][0], best_all[i][1], 
                                                                best_all[i][2], best_all[i][3]))

     