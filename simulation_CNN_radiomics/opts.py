import argparse 
from pathlib import Path

from traitlets import default


# Training settings
def parse_opts():
    parser = argparse.ArgumentParser(description='Bone Strength Project')
    parser.add_argument('--data_path',
                        default='/gpfs_projects/ran.yan/Project_Bone/DeepBone/data',
                        type=Path,
                        help='Data directory path')
    parser.add_argument('--result_path',
                        default='/gpfs_projects/ran.yan/Project_Bone/DeepBone/simulation_CNN_radiomics/output/TEST_29',
                        type=Path,
                        help='Result directory path')
    parser.add_argument('--resume_path',
                        # default=None,
                        default='/gpfs_projects/ran.yan/Project_Bone/DeepBone/simulation_CNN_radiomics/output/save_150.pth',
                        type=Path,
                        help='Save data (.pth) of previous training')
    parser.add_argument('--is_DL',
                        action='store_true',
                        help='If true, deep learning  is performed.')
    parser.add_argument('--is_RF',
                        action='store_true',
                        help='If true, random forest is performed.')
    parser.add_argument('--no_train',
                        action='store_true',
                        help='If true, training is not performed.')
    parser.add_argument('--no_val',
                        action='store_false',
                        help='If true, validation is not performed.')
    parser.add_argument('--inference',
                        action='store_false',
                        help='If true, inference is performed.')
    parser.add_argument('--inference_subset',
                        default='train',
                        type=str,
                        help='Used subset in inference (train | val | test)')
    parser.add_argument('--batch_size',
                        default=6,
                        type=int,
                        help='Batch Size')
    parser.add_argument('--inference_batch_size',
                        default=50,
                        type=int,
                        help='Batch Size for inference. 0 means this is the same as batch_size.')
    parser.add_argument('--no_sim',
                        action='store_true',
                        help='If true, image simulation is not performed.')
    parser.add_argument('--noise_scales',
                        default=1,
                        type=float,
                        help='Noise scaling.')
    parser.add_argument('--resolution_scales',
                        default=1,
                        type=float,
                        help='Resolution scaling.')
    parser.add_argument('--voxel_size_simulated',
                        default=(0.156,0.156,0.2),
                        type=tuple,
                        help='Voxel size of the simulated CT.')
    parser.add_argument('--n_epochs',
                        default=150,
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
                        default='multistep',
                        type=str,
                        help='Type of LR scheduler (multistep | plateau)')
    parser.add_argument('--multistep_milestones',
                        default=[50, 100, 150],
                        type=int,
                        nargs='+',
                        help='Milestones of LR scheduler. See documentation of MultistepLR.')
    parser.add_argument('--overwrite_milestones',
                        action='store_true',
                        help='If true, overwriting multistep_milestones when resuming training.')
    parser.add_argument('--plateau_patience',
                        default=10,
                        type=int,
                        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    parser.add_argument('--checkpoint',
                        default=10,
                        type=int,
                        help='Trained model is saved at every this epochs.')
    parser.add_argument('--no_cuda',
                        action='store_false',
                        help='If true, cuda is not used.')
    parser.add_argument('--n_threads',
                        default=4,
                        type=int,
                        help='Number of threads for multi-thread loading')
    parser.add_argument('--manual_seed',
                        default=100,
                        type=int,
                        help='Manually set random seed')
    parser.add_argument('--model',
                        default='resnet2p1d',
                        type=str,
                        help='(resnet | resnet2p1d | wideresnet | resnext | preresnet | densenet')
    parser.add_argument('--model_depth',
                        default=18,
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
    # parser.add_argument('--dist_url',
    #                     default='tcp://127.0.0.1:23456',
    #                     type=str,
    #                     help='url used to set up distributed training')
    # parser.add_argument('--world_size',
    #                     default=-1,
    #                     type=int,
    #                     help='number of nodes for distributed training')
    args = parser.parse_args()

    return args
    