import torch as t
import csv
from Data_Loader import train_loader, train_eval_loader, test_loader, val_loader
from Data_Config import data_config
from LR_Scheduler import CyclicLR, find_learning_rate, CosineAnnealingWarmRestarts
from Ensemble import ModelsAVG
from Pretrained_Models import MyPretrainModels
from Trainer import Trainer
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import matplotlib.pyplot as plt
plt.interactive(False)
t.backends.cudnn.benchmark = True


def average_main():
    avg_models = []
    device = t.device("cuda:0")
    my_models = MyPretrainModels(2019, device=t.device("cuda:0" if t.cuda.is_available() else "cpu"))
    # first model
    model_path = './ensemble/densenet161c_unfreeze_steplr_20190511/densenet161_freeze_train_2_8766.pkl'
    model = my_models.get_densenet161(file_path=model_path )
    avg_models.append(model)
    # second model
    model2_path = './ensemble/densenet161c_unfreeze_steplr_20190511/densenet161_freeze_train_4_8756.pkl'
    model2 = my_models.get_densenet161(file_path=model2_path)
    avg_models.append(model2) # 0.8867756315007429
    # third model
    model3_path = './ensemble/densenet161cycliclr_unfreeze_20190510/densenet161_freeze_train_2_613_865.pkl'
    model3 = my_models.get_densenet161(file_path=model3_path)
    avg_models.append(model3) # 0.8867756315007429
    # 4th model
    model4_path = './ensemble/resnet50_8_587_8624.pkl'
    model4 = my_models.get_resnet50(file_path=model4_path)
    avg_models.append(model4) # 0.9089648340762754
    # 5th model
    model5_path = './ensemble/resnet101_Epoch4_Iter0_616_872.pkl'
    model5 = my_models.get_resnet101(file_path=model5_path)
    avg_models.append(model5) # 0.9074789499752353
    # 6th model
    model6_path = './ensemble/resnet101_Epoch3_Iter0_614_866.pkl'
    model6 = my_models.get_resnet101(file_path=model6_path)
    avg_models.append(model6)
    # 7th model
    model7_path = './ensemble/resnet101_Epoch7_Iter0_607_860.pkl'
    model7 = my_models.get_resnet101(file_path=model7_path)
    avg_models.append(model7) # 0.9049034175334324
    # 8th model
    model8_path = './ensemble/desnet161_Epoch7_Iter0_587_852.pkl'
    model8 = my_models.get_densenet161(file_path=model8_path)
    #avg_models.append(model8) # 0.9049034175334324
    # 9th model
    model9_path = './ensemble/resnet_101_Epoch3_Iter0_597_854.pkl'
    model9 = my_models.get_resnet101(file_path=model9_path)
    #avg_models.append(model9)
    # 10th model
    model10_path = './ensemble/resnet50_5_597_8625.pkl'
    model10 = my_models.get_resnet50(file_path=model10_path)
    # avg_models.append(model10)
    # 11th model
    model11_path = './ensemble/resnet50_2_59_854.pkl'
    model11 = my_models.get_resnet50(file_path=model11_path)
    #avg_models.append(model11)
    # 12th model
    model12_path = './ensemble/resnet161_Epoch2_Iter0_595_857.pkl'
    model12 = my_models.get_densenet161(model12_path)
    #avg_models.append(model12)
    # 13 th model
    model_path = './ensemble/resnet101_Epoch6_Iter0_594_849.pkl'
    model13 = my_models.get_densenet161(model_path)
    # avg_models.append(model13)
    '''
    0. 1234 0.89404297
    1. 123456 0.90117187
    2. 1234567 0.90263672
    3. 12345  0.89882812
    4. 234567 0.90263672
    5. 23567 0.89609375 -> 4 is important 
    6. 23456 0.89404297 -> 7 is important
    7. 12345678 0.90322266   # 0.08144
    8. 123456789 0.9046875  
    9. 123456789(10) 0.90458984
    10. 1234567891011 0.90400391
    11 12345678911  0.90585938  # 0.08045
    12 1234567891112  0.90585938 
    13 1234567911 0.90576172 
    14 1234567891112 0.90595703
    15 123456789101112 0.90419922
    16 12345678910111213 0.90478516
    '''
    avg_ensemble = ModelsAVG(avg_models, val_loader, device=device)
    # print(avg_ensemble.run())
    avg_ensemble.test(test_loader, 'test10.csv')


def main():
    my_models = MyPretrainModels(2019, device=t.device("cuda:0" if t.cuda.is_available() else "cpu"))
    path = None
    model = my_models.get_se_resnext50_32x4d()
    # print(model)
    lr = [0.005, 0.03]
    bias, weights = [], []
    for name, weight in model.named_parameters():
        if 'bias' in name or 'bn' in name:
            bias.append(weight)
        else:
            weights.append(weight)
    optimizer = t.optim.SGD(model.parameters(), lr=lr[1])
    # find_learning_rate(model, train_loader, optimizer)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=4, eta_min=lr[0], T_mult=2)
    scheduler = CyclicLR(optimizer, base_lr=lr[0],max_lr=lr[1],step_size=45000, mode='triangular2')
    # scheduler = MultiStepLR(optimizer, milestones=[5, 15], gamma=0.1)
    trainer = Trainer(model, train_loader, lr=lr,
                      train_eval_loader=train_eval_loader,
                      val_loader=val_loader,
                      epoches=10,
                      lr_scheduler=scheduler,
                      optimizer=optimizer,
                      log_interval=1000)
    trainer.train()


if __name__ == '__main__':
    main()

