import argparse
import os
import shutil
from datetime import datetime

import numpy as np
import torch
import torchsummary
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.nn import CrossEntropyLoss
from torch.optim import Adam,SGD

from data.folder import FolderVideoData
from datasets.dataload import get_dataloader

# from models.resnet3d_transformer import generate_model
# from models.resnet_3d_fragment import generate_model
from models.resnet3d_ import generate_model
# from models.resnet3d_all import generate_model
# from models.SlowFast import resnet50,resnet101
# from models.C3D import C3D

from utils.create_log import logger_get
from utils.draw_utils import *
from utils.report_utils import df_generator, torch2onnx,loss_df_generator
from utils.save_args import save


# 导入参数
def get_arg():
    parser = argparse.ArgumentParser(description='classification parameter configuration(train)')
    parser.add_argument(
        '-t',
        type=str,
        default='class_action_resnet18(init)_16',
        help='This is your task theme name'
    )
    parser.add_argument(
        '-framep',
        type=str,
        default=r'D:\split_video\class_action_16',
        help="frame image's directory"
    )
    parser.add_argument(
        '-csvp',
        type=str,
        default=r'D:\PythonCode\Pytorch-VideoUnderstand-master\data\refer_16.csv',
        help="DIF(dataset information file)'s path"
    )
    parser.add_argument(
        '-clsp',
        type=str,
        default=r'D:\PythonCode\Pytorch-VideoUnderstand-master\data\classes.txt',
        help="classes.txt's path"
    )
    parser.add_argument(
        '-tp',
        type=float,
        default=0.7,
        help="train dataset's percent"
    )
    parser.add_argument(
        '-bs',
        type=int,
        default=32,
        help="train dataset's batch size"
    )
    parser.add_argument(
        '-rs',
        type=tuple,
        default=(112, 112),
        help='resized shape of input tensor'
    )
    parser.add_argument(
        '-cn',
        type=int,
        default=4,
        help='the number of classes'
    )
    parser.add_argument(
        '-e',
        type=int,
        default=150,
        help='epoch'
    )
    parser.add_argument(
        '-lr',
        type=float,
        default=0.0001,
        help='learning rate'
    )
    parser.add_argument(
        '-ld',
        type=str,
        default=r'D:\PythonCode\Pytorch-VideoUnderstand-master\work_dir\class_action\init',
        help="the training log's save directory"
    )
    parser.add_argument(
        '-nw',
        type=int,
        default=10,
        help="number of workers"
    )
    parser.add_argument(
        '-sf',
        type=str,
        default=16,
        help="the frames per video"
    )
    return parser.parse_args()


# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # 任务开始前的准备工作
    args = get_arg()  # 得到参数Namespace
    nowtime = datetime.now()  # 获取任务开始时间
    log_dir = "{}/exp_{}_{}_{}_{}_{}".format(args.ld, nowtime.month, nowtime.day, nowtime.hour, nowtime.minute, args.t)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir,
                            "exp_{}_{}_{}_{}_{}.log".format(nowtime.month, nowtime.day, nowtime.hour, nowtime.minute,
                                                            args.t))
    file_logger = logger_get(log_path)  # 获取logger
    source_data = FolderVideoData(args.framep, args.clsp, args.csvp)
    print("Generating classes.txt and DIF file.......")
    try:
        source_data.generate()
        print("Done.")
    except:
        file_logger.error("Generating DIF and classes.txt failure!")
    # 训练设备信息
    device_table = ""
    if torch.cuda.is_available():
        device_table = PrettyTable(['number of gpu', 'applied gpu index', 'applied gpu name'], min_table_width=80)
        gpu_num = torch.cuda.device_count()
        gpu_index = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name()
        device_table.add_row([str(gpu_num), str(gpu_index), str(gpu_name)])
        file_logger.info('Training device information:\n{}\n'.format(device_table))
    else:
        file_logger.warning("Using cpu......")
        device_table = 'CPU'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ----------------------------------------------------------------------------------------------------------------------
    # 数据集信息
    file_logger.info(
        "Use dataset information file:{}\nLoading dataset from path: {}......".format(args.csvp, args.framep))
    train_dl, valid_dl, samples_num, train_num, valid_num = get_dataloader(args.framep, args.csvp, args.rs, args.bs
                                                                           , args.nw, args.tp)
    dataset_table = PrettyTable(['number of samples', 'train number', 'valid number', 'percent'], min_table_width=80)
    dataset_table.add_row([samples_num, train_num, valid_num, args.tp])
    file_logger.info("dataset information:\n{}\n".format(dataset_table))
    # 绘制数据分布图
    dataset_distribution(args.csvp, args.clsp, log_dir)
    # ----------------------------------------------------------------------------------------------------------------------
    # 类别信息
    classes = source_data.cls_list
    classes_table = PrettyTable(classes, min_table_width=80)
    classes_table.add_row(range(len(classes)))
    file_logger.info("Classes information:\n{}\n".format(classes_table))
    # ----------------------------------------------------------------------------------------------------------------------
    # 训练组件配置
    # model = torch.load(r'D:\PythonCode\Pytorch-VideoUnderstand-master\work_dir\class_action\transformer\exp_9_10_15_44_class_action_resnet18(trans_early_encoder)_16\checkpoints\last.pth').to(device)
    model = generate_model(model_depth=18, n_classes=args.cn).to(device)
    # model = C3D(num_classes=args.cn).to(device)  ##################################################
    # model = resnet50(class_num=args.cn).to(device)
    # torchsummary.summary(model=model, input_size=(3, args.sf, *args.rs), batch_size=args.bs)
    optimizer = Adam(params=model.parameters(), lr=args.lr, weight_decay=1e-5)  ##################################################
    loss_fn = CrossEntropyLoss()  ##################################################
    train_table = PrettyTable(['theme', 'resize', 'batch size', 'epoch', 'learning rate', 'directory of log'],
                              min_table_width=120)
    train_table.add_row([args.t, args.rs, args.bs, args.e, args.lr, args.ld])
    # 保存训练参数表
    save(args,os.path.join(log_dir, 'args.yaml'))
    file_logger.info('Train information:\n{}\n'.format(train_table))
    # ----------------------------------------------------------------------------------------------------------------------
    # 开始训练
    file_logger.info("Train begins......")
    losses = []
    val_losses = []
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    aucs = []
    best_checkpoint = 0.
    shapeuse = 0
    preduse = 0
    labeluse = 0
    st = datetime.now()
    for epoch in range(args.e):

        prediction = []
        label = []
        score = []

        model.train()
        train_bar = tqdm(iter(train_dl), ncols=100, colour='blue')
        train_loss = 0.
        i = 0
        for train_data in train_bar:
            x_train, y_train = train_data
            shapeuse = x_train.shape
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            output = model(x_train)
            loss = loss_fn(output, y_train)
            optimizer.zero_grad()
            # clone().detach()：可以仅仅复制一个tensor的数值而不影响tensor# 原内存和计算图
            train_loss += loss.clone().detach().cpu().numpy()
            loss.backward()
            optimizer.step()
            # 显示每一批次的loss
            train_bar.set_description("Epoch:{}/{} Step:{}/{}".format(epoch + 1, args.e, i + 1, len(train_dl)))
            train_bar.set_postfix({"train loss": "%.3f" % loss.data})
            i += 1
        train_loss = train_loss / i
        file_logger.info("Epoch loss:{}".format(train_loss))
        # 最后得到的i是一次迭代中的样本数批数
        losses.append(train_loss)

        model.eval()
        valid_bar = tqdm(iter(valid_dl), ncols=150, colour='blue')
        valid_acc = 0.
        valid_pre = 0.
        valid_recall = 0.
        valid_f1 = 0.
        valid_auc = 0.
        valid_loss = 0.
        i = 0
        with torch.no_grad():
            for valid_data in valid_bar:
                x_valid, y_valid = valid_data
                x_valid = x_valid.to(device)
                y_valid_ = y_valid.clone().detach().numpy().tolist()  # y_valid就不必放到gpu上训练了
                output = model(x_valid)  # shape:(N*cls_n)
                loss = loss_fn(output, y_valid.to(device))
                valid_loss += loss.clone().detach().cpu().numpy()
                output_ = output.clone().detach().cpu()
                _, pred = torch.max(output_, 1)  # 输出每一行(样本)的最大概率的下标
                pred_ = pred.clone().detach().numpy().tolist()
                output_ = output_.numpy().tolist()
                # 显示每一批次的acc/precision/recall/f1
                valid_bar.set_description("Epoch:{}/{} Step:{}/{}".format(epoch + 1, args.e, i + 1, len(valid_dl)))
                prediction = prediction + pred_
                label = label + y_valid_
                score = score + output_
                i += 1
        # 最后得到的i是一次迭代中的样本数批数,每一次epoch计算一次indicators
        valid_acc = accuracy_score(y_true=label, y_pred=prediction)
        valid_pre = precision_score(y_true=label, y_pred=prediction, average='weighted')
        valid_recall = recall_score(y_true=label, y_pred=prediction, average='weighted')
        valid_f1 = f1_score(y_true=label, y_pred=prediction, average='weighted')
        valid_auc = roc_auc_score(y_true=np.eye(args.cn)[np.array(label)], y_score=np.eye(args.cn)[np.array(prediction)], average='weighted', multi_class='ovr')    # numpy-onehot化
        valid_loss /= i

        preduse = prediction
        labeluse = label
        # 验证阶段信息输出
        indicator_table = PrettyTable(['Accuracy', 'Precision', 'Recall', 'F1', 'AUC'], )
        indicator_table.add_row([valid_acc, valid_pre, valid_recall, valid_f1, valid_auc])
        file_logger.info('\n{}\n'.format(indicator_table))
        # indicator保存
        accuracies.append(valid_acc)
        precisions.append(valid_pre)
        recalls.append(valid_recall)
        f1s.append(valid_f1)
        aucs.append(valid_auc)
        val_losses.append(valid_loss)
        # 保存最好的f1指标的checkpoint
        if valid_acc >= max(accuracies):  # 如果本次epoch的acc大于了存储f1列表的最大值，那么最好的checkpoint赋值为model
            best_checkpoint = model
        # 保存每次的checkpoint，从而实现断点继训
        os.makedirs("../checkpoints", exist_ok=True)  # 项目根路径下的checkpoints目录下保存临时checkpoint
        if not os.path.exists("../checkpoints/train_info.txt"):
            with open("../checkpoints/info.txt", 'w') as f:
                content = "{}\n{}\n{}\n{}y\n{}\n".format(dataset_table, classes_table, device_table, train_table,
                                                         optimizer)
                f.write(content)
        torch.save(model, "../checkpoints/{}.pth".format(epoch))
        # indicators的df记录文件生成
        df = df_generator(epoch+1, [accuracies, precisions, recalls, f1s, aucs],
                          os.path.join(log_dir, 'indicators.csv'))
        # loss的df记录文件生成
        df_loss = loss_df_generator([losses,val_losses], os.path.join(log_dir,'losses.csv'))
        # 绘制loss和indicators变化曲线
        log_plot(df, log_dir)
        # 绘制loss的变化曲线
        loss_plot(df_loss, log_dir)
    et = datetime.now()
    # ----------------------------------------------------------------------------------------------------------------------
    # 完成训练后的断电续训的临时文件的删除、日志保存(程序结束后自动保存)以及绘图等后续工作
    # 耗时记录
    file_logger.info("Training time:{}".format(et - st))
    # 训练完，记得把model和优化器也加入到日志中（训练完加入以防训练前对model或者优化器产生影响）
    file_logger.info("optimizer:\n{}\nmodel:\n{}\n".format(str(optimizer), str(model)))

    # 权重生成(onnx/pth + bestacc/last)
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(model, os.path.join(checkpoint_dir, 'last.pth'))  # 最后一次的checkpoint
    try:
        torch2onnx(os.path.join(checkpoint_dir, 'last.pth'), os.path.join(checkpoint_dir, 'last.onnx'), shapeuse)
        file_logger.info(
            "Last model transforms successfully and path is {}.".format(os.path.join(checkpoint_dir, 'last.onnx')))
    except:
        file_logger.warning("Last model transforms failed.")

    torch.save(best_checkpoint, os.path.join(checkpoint_dir, 'best_acc.pth'))  # 最好的f1的checkpoint
    try:
        torch2onnx(os.path.join(checkpoint_dir, 'best_acc.pth'), os.path.join(checkpoint_dir, 'best_acc.onnx'), shapeuse)
        file_logger.info(
            "Best model transforms successfully and path is {}.".format(os.path.join(checkpoint_dir, 'best_f1.onnx')))
    except:
        file_logger.warning("Best model transforms failed.")
    # 绘图（当然也可以选择使用提供的函数在训练后绘制，一些参数可以在宝库函数中自行调整）
    # 绘制最后一次的热力图——当然可以根据自己改预测和标签
    get_confusion_matrix(y_pred=preduse, y_label=labeluse, cls_num=args.cn, fig_save_dir=log_dir)
    # 删除临时checkpoints文件以及临时信息文件——到这儿任务基本完成了，就说没没有出现断点，那么就删除
    shutil.rmtree("../checkpoints")

