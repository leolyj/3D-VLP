import warnings
import os
import sys
import json
import h5py
import argparse
import importlib
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

# HACK ignore warnings
warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.append(os.path.join(os.getcwd()))  # HACK add the root folder

from torch.utils.data import DataLoader
from datetime import datetime
from copy import deepcopy
from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.joint.dataset import ScannetReferenceDataset
from lib.joint.solver_pretrain import Solver
from lib.configs.config_pretrain import CONF
from models.network.network_pretrain import JointNet
from scripts.utils.AdamW import AdamW
from scripts.utils.script_utils import set_params_lr_dict


# extracted ScanNet object rotations from Scan2CAD
# NOTE some scenes are missing in this annotation!!!
SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))
SCAN2CAD_ROTATION = json.load(open(os.path.join(CONF.PATH.SCAN2CAD, "scannet_instance_rotations.json")))

# constants
DC = ScannetDatasetConfig()


def get_dataloader(args, scanrefer, scanrefer_new, all_scene_list, split, config, augment, shuffle=True,
                   scan2cad_rotation=None):
    dataset = ScannetReferenceDataset(
        scanrefer=scanrefer,
        scanrefer_new=scanrefer_new,
        scanrefer_all_scene=all_scene_list,
        split=split,
        name=args.dataset,
        num_points=args.num_points,
        use_height=(not args.no_height),
        use_color=args.use_color,
        use_normal=args.use_normal,
        use_multiview=args.use_multiview,
        lang_num_max=args.lang_num_max,
        augment=augment,
        shuffle=shuffle,
        scan2cad_rotation=scan2cad_rotation
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    return dataset, dataloader


def get_model(args, dataset, device):
    # initiate model
    print('multiview', args.use_multiview)
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(
        not args.no_height)
    print('input channels', input_channels)
    model = JointNet(
        num_class=DC.num_class,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr,
        input_feature_dim=input_channels,
        num_proposal=args.num_proposals,
        use_lang_classifier=(not args.no_lang_cls),
        use_bidir=args.use_bidir,
        dataset_config=DC
    )

    if args.use_pretrained:
        print("loading pretrained VoteNet...")
        pretrained_path = os.path.join(CONF.PATH.BASE, args.use_pretrained, "model_last.pth")
        print("pretrained_path", pretrained_path, flush=True)
        pretrained_model = JointNet(
            num_class=DC.num_class,
            num_heading_bin=DC.num_heading_bin,
            num_size_cluster=DC.num_size_cluster,
            mean_size_arr=DC.mean_size_arr,
            num_proposal=args.num_proposals,
            input_feature_dim=input_channels
        )
        pretrained_model.load_state_dict(torch.load(pretrained_path), strict=False)
        # mount pretrained VoteNet
        model.backbone_net = pretrained_model.backbone_net
        model.vgen = pretrained_model.vgen
        model.proposal = pretrained_model.proposal

    # mount

    if args.no_detection:
        # freeze pointnet++ backbone
        for param in model.backbone_net.parameters():
            param.requires_grad = False

        # freeze voting
        for param in model.vgen.parameters():
            param.requires_grad = False

        # freeze detector
        for param in model.proposal.parameters():
            param.requires_grad = False

        # freeze relation
        # for param in model.relation.parameters():
        #    param.requires_grad = False

    model.to(device)

    return model


def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params


def get_solver(args, dataset, dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(args, dataset["train"], device)
    # TODO

    # weight_dict = {
    #     'backbone_net': {'lr': 0.0001},
    #     'vgen': {'lr': 0.0001},
    #     'proposal': {'lr': 0.0001},
    #     'lang': {'lr': 0.0005},
    #     'relation': {'lr': 0.0005},
    #     'match': {'lr': 0.0005},
    #     'caption': {'lr': 0.0005},
    # }

    # original
    weight_dict = {
        'lang': {'lr': 0.0005},
        'relation': {'lr': 0.0005},
        'match': {'lr': 0.0005},
    }

    params = set_params_lr_dict(model, base_lr=args.lr, weight_decay=args.wd, weight_dict=weight_dict)
    # params = model.parameters()
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.wd, amsgrad=args.amsgrad)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    checkpoint_best = None

    if args.use_checkpoint:
        print("loading checkpoint {}...".format(args.use_checkpoint))
        stamp = args.use_checkpoint
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        checkpoint = torch.load(os.path.join(CONF.PATH.OUTPUT, args.use_checkpoint, "checkpoint.tar"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        checkpoint_best = checkpoint["best"]
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.tag: stamp += "_" + args.tag.upper()
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        os.makedirs(root, exist_ok=True)

    # scheduler parameters for training solely the detection pipeline
    # LR_DECAY_STEP = [80, 120, 160] if args.no_caption else None
    LR_DECAY_STEP = None
    if args.coslr:
        LR_DECAY_STEP = {
            'type': 'cosine',
            'T_max': args.epoch,
            'eta_min': 1e-5,
        }
    # LR_DECAY_RATE = 0.1 if args.no_caption else None
    # BN_DECAY_STEP = 20 if args.no_caption else None
    # BN_DECAY_RATE = 0.5 if args.no_caption else None
    LR_DECAY_RATE = None
    BN_DECAY_STEP = None
    BN_DECAY_RATE = None

    print('LR&BN_DECAY', LR_DECAY_STEP, LR_DECAY_RATE, BN_DECAY_STEP, BN_DECAY_RATE, flush=True)
    print("criterion", args.criterion, flush=True)
    solver = Solver(
        model=model,
        device=device,
        config=DC,
        dataset=dataset,
        dataloader=dataloader,
        optimizer=optimizer,
        stamp=stamp,
        val_step=args.val_step,
        num_epoch=args.num_epoch,
        detection=not args.no_detection,
        orientation=args.use_orientation,
        distance=args.use_distance,
        use_tf=args.use_tf,
        reference=not args.no_reference,
        use_lang_classifier=not args.no_lang_cls,
        lr_decay_step=LR_DECAY_STEP,
        lr_decay_rate=LR_DECAY_RATE,
        bn_decay_step=BN_DECAY_STEP,
        bn_decay_rate=BN_DECAY_RATE,
        criterion=args.criterion,
        checkpoint_best=checkpoint_best
    )
    num_params = get_num_params(model)

    return solver, num_params, root


def save_info(args, root, num_params, dataset):
    info = {}
    for key, value in vars(args).items():
        info[key] = value

    info["num_train"] = len(dataset["train"])
    info["num_eval_train"] = len(dataset["eval"]["train"])
    info["num_eval_val"] = len(dataset["eval"]["val"])
    info["num_train_scenes"] = len(dataset["train"].scene_list)
    info["num_eval_train_scenes"] = len(dataset["eval"]["train"].scene_list)
    info["num_eval_val_scenes"] = len(dataset["eval"]["val"].scene_list)
    info["num_params"] = num_params

    with open(os.path.join(root, "info.json"), "w") as f:
        json.dump(info, f, indent=4)


def get_scannet_scene_list(split):
    scene_list = sorted(
        [line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])

    return scene_list


def get_scanrefer(args):
    if args.dataset == "ScanRefer":
        scanrefer_train = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
        scanrefer_eval_train = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
        scanrefer_eval_val = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))
    elif args.dataset == "ReferIt3D":
        scanrefer_train = json.load(open(os.path.join(CONF.PATH.DATA, "nr3d_train.json")))
        scanrefer_eval_train = json.load(open(os.path.join(CONF.PATH.DATA, "nr3d_train.json")))
        scanrefer_eval_val = json.load(open(os.path.join(CONF.PATH.DATA, "nr3d_val.json")))
    else:
        raise ValueError("Invalid dataset.")

    if args.debug:
        scanrefer_train = [SCANREFER_TRAIN[0]]
        scanrefer_eval_train = [SCANREFER_TRAIN[0]]
        scanrefer_eval_val = [SCANREFER_TRAIN[0]]

    # get initial scene list
    train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train])))
    val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_eval_val])))

    # filter data in chosen scenes
    new_scanrefer_train = []
    scanrefer_train_new = []
    scanrefer_train_new_scene = []
    scene_id = ""
    for data in scanrefer_train:
        if data["scene_id"] in train_scene_list:
            new_scanrefer_train.append(data)
            if scene_id != data["scene_id"]:
                scene_id = data["scene_id"]
                if len(scanrefer_train_new_scene) > 0:
                    scanrefer_train_new.append(scanrefer_train_new_scene)
                scanrefer_train_new_scene = []
            if len(scanrefer_train_new_scene) >= args.lang_num_max:
                scanrefer_train_new.append(scanrefer_train_new_scene)
                scanrefer_train_new_scene = []
            scanrefer_train_new_scene.append(data)
    scanrefer_train_new.append(scanrefer_train_new_scene)

    # 注意：new_scanrefer_eval_train实际上没用
    # eval on train
    new_scanrefer_eval_train = []
    scanrefer_eval_train_new = []
    for scene_id in train_scene_list:
        data = deepcopy(SCANREFER_TRAIN[0])
        data["scene_id"] = scene_id
        new_scanrefer_eval_train.append(data)
        scanrefer_eval_train_new_scene = []
        for i in range(args.lang_num_max):
            scanrefer_eval_train_new_scene.append(data)
        scanrefer_eval_train_new.append(scanrefer_eval_train_new_scene)

    new_scanrefer_eval_val = scanrefer_eval_val
    scanrefer_eval_val_new = []
    scanrefer_eval_val_new_scene = []
    scene_id = ""
    for data in scanrefer_eval_val:
        # if data["scene_id"] not in scanrefer_val_new:
        # scanrefer_val_new[data["scene_id"]] = []
        # scanrefer_val_new[data["scene_id"]].append(data)
        if scene_id != data["scene_id"]:
            scene_id = data["scene_id"]
            if len(scanrefer_eval_val_new_scene) > 0:
                scanrefer_eval_val_new_scene.append(scanrefer_eval_val_new_scene)
            scanrefer_eval_val_new_scene = []
        if len(scanrefer_eval_val_new_scene) >= args.lang_num_max:
            scanrefer_eval_val_new.append(scanrefer_eval_val_new_scene)
            scanrefer_eval_val_new_scene = []
        scanrefer_eval_val_new_scene.append(data)
    scanrefer_eval_val_new.append(scanrefer_eval_val_new_scene)

    new_scanrefer_eval_val2 = []
    scanrefer_eval_val_new2 = []
    for scene_id in val_scene_list:
        data = deepcopy(SCANREFER_VAL[0])
        data["scene_id"] = scene_id
        new_scanrefer_eval_val2.append(data)
        scanrefer_eval_val_new_scene2 = []
        for i in range(args.lang_num_max):
            scanrefer_eval_val_new_scene2.append(data)
        scanrefer_eval_val_new2.append(scanrefer_eval_val_new_scene2)

    print("scanrefer_train_new", len(scanrefer_train_new), len(scanrefer_train_new[0]))
    print("scanrefer_eval_new", len(scanrefer_eval_train_new), len(scanrefer_eval_val_new))
    sum = 0
    for i in range(len(scanrefer_train_new)):
        sum += len(scanrefer_train_new[i])
        # print(len(scanrefer_train_new[i]))
    # for i in range(len(scanrefer_val_new)):
    #    print(len(scanrefer_val_new[i]))
    print("sum", sum)  # 1418 363

    # all scanrefer scene
    all_scene_list = train_scene_list + val_scene_list

    print("using {} dataset".format(args.dataset))
    print("train on {} samples from {} scenes".format(len(new_scanrefer_train), len(train_scene_list)))
    print("eval on {} scenes from train and {} scenes from val".format(len(new_scanrefer_eval_train),
                                                                       len(new_scanrefer_eval_val)))

    return new_scanrefer_train, new_scanrefer_eval_train, new_scanrefer_eval_val, new_scanrefer_eval_val2, all_scene_list, scanrefer_train_new, scanrefer_eval_train_new, scanrefer_eval_val_new, scanrefer_eval_val_new2


def train(args):
    # init training dataset
    print("preparing data...")
    scanrefer_train, scanrefer_eval_train, scanrefer_eval_val, scanrefer_eval_val2, all_scene_list, scanrefer_train_new, scanrefer_eval_train_new, scanrefer_eval_val_new, scanrefer_eval_val_new2 = get_scanrefer(
        args)

    # 注意：eval_train_dataset实际上没用
    # dataloader
    train_dataset, train_dataloader = get_dataloader(args, scanrefer_train, scanrefer_train_new, all_scene_list,
                                                     "train", DC, True, SCAN2CAD_ROTATION)
    eval_train_dataset, eval_train_dataloader = get_dataloader(args, scanrefer_eval_train, scanrefer_eval_train_new,
                                                               all_scene_list, "val", DC, False, shuffle=False)
    eval_val_dataset, eval_val_dataloader = get_dataloader(args, scanrefer_eval_val, scanrefer_eval_val_new,
                                                           all_scene_list, "val", DC, False, shuffle=False)
    eval_val_dataset2, eval_val_dataloader2 = get_dataloader(args, scanrefer_eval_val2, scanrefer_eval_val_new2,
                                                             all_scene_list, "val", DC, False, shuffle=False)

    dataset = {
        "train": train_dataset,
        "eval": {
            "train": eval_train_dataset,
            "val": eval_val_dataset,
            "val_scene": eval_val_dataset2
        }
    }
    dataloader = {
        "train": train_dataloader,
        "eval": {
            "train": eval_train_dataloader,
            "val": eval_val_dataloader,
            "val_scene": eval_val_dataloader2
        }
    }

    print("initializing...")
    solver, num_params, root = get_solver(args, dataset, dataloader)

    print("Start training...\n")
    save_info(args, root, num_params, dataset)
    solver(args.epoch, args.verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, help="tag for the training, e.g. cuda_wl", default="pretrain")
    parser.add_argument("--dataset", type=str, help="Choose a dataset: ScanRefer or ReferIt3D", default="ScanRefer")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    parser.add_argument("--batch_size", type=int, help="batch size", default=10)
    parser.add_argument("--epoch", type=int, help="number of epochs", default=150)
    parser.add_argument("--verbose", type=int, help="iterations of showing verbose", default=50)
    parser.add_argument("--val_step", type=int, help="iterations of validating", default=1000)
    parser.add_argument("--lr", type=float, help="learning rate", default=0.002)
    parser.add_argument("--wd", type=float, help="weight decay", default=1e-3)
    parser.add_argument("--amsgrad", action='store_true', help="optimizer with amsgrad")

    parser.add_argument("--lang_num_max", type=int, help="lang num max", default=8)
    parser.add_argument("--num_points", type=int, default=40000, help="Point Number [default: 40000]")
    parser.add_argument("--num_proposals", type=int, default=256, help="Proposal number [default: 256]")
    parser.add_argument("--num_locals", type=int, default=20, help="Number of local objects [default: -1]")
    parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")
    parser.add_argument("--num_graph_steps", type=int, default=0, help="Number of graph conv layer [default: 0]")
    parser.add_argument("--num_epoch", type=int, default=150, help="Number of pretrain epoch [default: 150]")

    parser.add_argument("--criterion", type=str, default="sum", \
                        help="criterion for selecting the best model [choices: bleu-1, bleu-2, bleu-3, bleu-4, cider, rouge, meteor, sum]")
    parser.add_argument("--query_mode", type=str, default="center",
                        help="Mode for querying the local context, [choices: center, corner]")
    parser.add_argument("--graph_mode", type=str, default="edge_conv",
                        help="Mode for querying the local context, [choices: graph_conv, edge_conv]")
    parser.add_argument("--graph_aggr", type=str, default="add",
                        help="Mode for aggregating features, [choices: add, mean, max]")

    parser.add_argument("--coslr", type=bool, default=True, help="cosine learning rate")
    parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_augment", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_detection", action="store_true", help="Do NOT train the detection module.")
    parser.add_argument("--no_lang_cls", action="store_true", help="Do NOT use language classifier.")
    parser.add_argument("--no_reference", action="store_true", help="Do NOT train the localization module.")

    parser.add_argument("--use_tf", action="store_true", help="enable teacher forcing in inference.")
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_multiview", action="store_true", help="Use multiview images.")
    parser.add_argument("--use_relation", action="store_true", help="Use object-to-object relation in graph.")
    parser.add_argument("--use_new", action="store_true", help="Use new Top-down module.")
    parser.add_argument("--use_orientation", action="store_true",
                        help="Use object-to-object orientation loss in graph.")
    parser.add_argument("--use_distance", action="store_true", help="Use object-to-object distance loss in graph.")
    parser.add_argument("--use_bidir", action="store_true", help="Use bi-directional GRU.")
    parser.add_argument("--use_pretrained", type=str, default='',
                        help="Specify the folder name containing the pretrained detection module.")
    parser.add_argument("--use_checkpoint", type=str, help="Specify the checkpoint root", default="")

    parser.add_argument("--debug", action="store_true", help="Debug mode.")
    args = parser.parse_args()

    # # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    train(args)

