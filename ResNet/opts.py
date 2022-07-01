import argparse 
from pathlib import Path


# Training settings
def parse_opts():
    parser = argparse.ArgumentParser(description='Bone Strength Project')
    parser.add_argument('--data_path',
                        default='/gpfs_projects/ran.yan/Project_Bone/DeepBone/data',
                        type=Path,
                        help='Data directory path')
    parser.add_argument('--result_path',
                        default='/gpfs_projects/ran.yan/Project_Bone/DeepBone/ResNet/output',
                        type=Path,
                        help='Result directory path')
    parser.add_argument('--batch_size',
                        default=5,
                        type=int,
                        help='Batch Size')
    parser.add_argument('--inference_batch_size',
                        default=0,
                        type=int,
                        help='Batch Size for inference. 0 means this is the same as batch_size.')
    parser.add_argument('--n_epochs',
                        default=20,
                        type=int,
                        help='Number of total epochs to run')
    parser.add_argument('--learning_rate',
                        default=0.01,
                        type=float,
                        help=('Initial learning rate'
                              '(divided by 10 while training by lr scheduler)'))
    parser.add_argument('--momentum',
                        default=0.9,
                        type=float,
                        help='Momentum')
    parser.add_argument('--dampening',
                        default=0.0,
                        type=float,
                        help='dampening of SGD')
    parser.add_argument('--weight_decay',
                        default=1e-3,
                        type=float,
                        help='Weight Decay')
    parser.add_argument('--nesterov',
                        action='store_true',
                        help='Nesterov momentum')
    parser.add_argument('--optimizer',
                        default='sgd',
                        type=str,
                        help='Currently only support SGD')
    parser.add_argument('--lr_scheduler',
                        default='plateau',
                        type=str,
                        help='Type of LR scheduler (multistep | plateau)')
    parser.add_argument('--multistep_milestones',
                        default=[50, 100, 150],
                        type=int,
                        nargs='+',
                        help='Milestones of LR scheduler. See documentation of MultistepLR.')
    parser.add_argument('--plateau_patience',
                        default=10,
                        type=int,
                        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    parser.add_argument('--checkpoint',
                        default=10,
                        type=int,
                        help='Trained model is saved at every this epochs.')
    parser.add_argument('--no_train',
                        action='store_true',
                        help='If true, training is not performed.')
    parser.add_argument('--no_val',
                        action='store_true',
                        help='If true, validation is not performed.')
    parser.add_argument('--inference',
                        action='store_true',
                        help='If true, inference is performed.')
    parser.add_argument('--inference_subset',
                        default='val',
                        type=str,
                        help='Used subset in inference (train | val | test)')
    parser.add_argument('--no_cuda',
                        action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--n_threads',
                        default=4,
                        type=int,
                        help='Number of threads for multi-thread loading')
    parser.add_argument('--manual_seed',
                        default=1,
                        type=int,
                        help='Manually set random seed')
    parser.add_argument('--model',
                        default='densenet',
                        type=str,
                        help='(resnet | resnet2p1d | wideresnet | resnext | preresnet | densenet')
    parser.add_argument('--model_depth',
                        default=10,
                        type=int,
                        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--resnet_shortcut',
                        default='B',
                        type=str,
                        help='Shortcut type of resnet (A | B)')
    parser.add_argument('--conv1_t_size',
                        default=7,
                        type=int,
                        help='Kernel size in t dim of conv1.')
    parser.add_argument('--conv1_t_stride',
                        default=1,
                        type=int,
                        help='Stride in t dim of conv1.')
    parser.add_argument('--no_max_pool',
                        action='store_true',
                        help='If true, the max pooling after conv1 is removed.')
    parser.add_argument('--resnet_widen_factor',
                        default=1.0,
                        type=float,
                        help='The number of feature maps of resnet is multiplied by this value')
    parser.add_argument('--wide_resnet_k',
                        default=2,
                        type=int,
                        help='Wide resnet k')
    parser.add_argument('--resnext_cardinality',
                        default=32,
                        type=int,
                        help='ResNeXt cardinality')
    parser.add_argument('--distributed',
                        action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs.')
    parser.add_argument('--tensorboard',
                        action='store_false',
                        help='If true, output tensorboard log file.')
    args = parser.parse_args()


    return args
