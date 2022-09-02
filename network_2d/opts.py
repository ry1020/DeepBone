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
                        default='/gpfs_projects/ran.yan/Project_Bone/DeepBone/network_2d/output/TEST_36',
                        type=Path,
                        help='Result directory path')
    parser.add_argument('--resume_path',
                        default=None,
                        # default='/gpfs_projects/ran.yan/Project_Bone/DeepBone/network_2d/output/TEST_36/save_120.pth',
                        type=Path,
                        help='Save data (.pth) of previous training')
    parser.add_argument('--no_train',
                        action='store_true',
                        help='If true, training is not performed.')
    parser.add_argument('--train_subset',
                        default='train',
                        type=str,
                        help='Used subset in inference (train | trainVal)')
    parser.add_argument('--no_val',
                        action='store_true',
                        help='If true, validation is not performed.')
    parser.add_argument('--inference',
                        action='store_false',
                        help='If true, inference is performed.')
    parser.add_argument('--inference_subset',
                        default='test',
                        type=str,
                        help='Used subset in inference (train | trainVal | val | test | L1)')
    parser.add_argument('--batch_size',
                        default=64,
                        type=int,
                        help='Batch Size')
    parser.add_argument('--inference_batch_size',
                        default = 32,
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
                        default=200,
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
    parser.add_argument('--multistep_gamma',
                        default=0.3,
                        type=float,
                        help='Gamma of LR scheduler. See documentation of MultistepLR.')
    parser.add_argument('--multistep_milestones',
                        default= [125,160, 180, 200],
                        type=int,
                        nargs='+',
                        help='Milestones of LR scheduler. See documentation of MultistepLR.')
    parser.add_argument('--overwrite_milestones',
                        action='store_false',
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
                        action='store_true',
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
                        default='resnet18',
                        type=str,
                        help='(resnet18 | resnet34 | resnet50 | resnet101 | resnet152 | resnext50_32x4d | resnext101_32x8d | resnext101_64x4d | wide_resnet50_2 | wide_resnet101_2)')
    parser.add_argument('--zero_init_residual',
                        action='store_false',
                        help='If true, zero-initialize the last BN in each residual branch.')
    parser.add_argument('--replace_stride_with_dilation',
                        default= [False, False, False],
                        type=bool,
                        nargs='+',
                        help='Each element in the tuple indicates if we should replace the 2x2 stride with a dilated convolution instead')
    parser.add_argument('--distributed',
                        action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs.')
    parser.add_argument('--tensorboard',
                        action='store_false',
                        help='If true, output tensorboard log file.')
    args = parser.parse_args()

    return args
    