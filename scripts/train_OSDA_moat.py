import os
import torch
torch.backends.cudnn.benchmark=True
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from distutils.version import LooseVersion

import sys
sys.path.append(os.path.abspath('.'))

from datasets.synthia_Dataset_ADA import SYNTHIA_Dataset
from datasets.gta5_Dataset_ADA import GTA5_Dataset
from datasets.cityscapes_Dataset_ADA import City_Dataset

from utils.losses import feat_reg_ST_loss, IW_MaxSquareloss
from utils.train_helper import get_model
from utils.utils import *

from train_source import Trainer, str2bool, argparse, add_train_args, init_args, datasets_path


DEBUG = False


def memory_check(log_string):
    torch.cuda.synchronize()
    if DEBUG:
        print(log_string)
        print(' peak:', '{:.3f}'.format(torch.cuda.max_memory_allocated() / 1024 ** 3), 'GB')
        print(' current', '{:.3f}'.format(torch.cuda.memory_allocated() / 1024 ** 3), 'GB')


class UDATrainer(Trainer):

    def __init__(self, args, cuda=None, train_id="None", logger=None):
        super().__init__(args, cuda, train_id, logger)


        ### DATASETS ###
        self.logger.info('Adaptation {} -> {}'.format(self.args.source_dataset, self.args.target_dataset))

        source_data_kwargs = {'data_root_path':args.source_data_path,
                              'list_path':args.source_list_path,
                              'base_size':args.base_size,
                              'crop_size':args.crop_size}
        target_data_kwargs = {'data_root_path': args.data_root_path,
                              'list_path': args.list_path,
                              'base_size': args.target_base_size,
                              'crop_size': args.target_crop_size}
        dataloader_kwargs = {'batch_size':self.args.batch_size,
                             'num_workers':self.args.data_loader_workers,
                             'pin_memory':self.args.pin_memory,
                             'drop_last':True}

        if self.args.source_dataset == 'synthia':
            source_data_kwargs['class_16'] = target_data_kwargs['class_16'] = args.class_16

        source_data_gen = SYNTHIA_Dataset if self.args.source_dataset == 'synthia' else GTA5_Dataset

        if DEBUG: print('DEBUG: Loading training dataset (source)')
        self.source_dataloader = data.DataLoader(source_data_gen(args, split='train', **source_data_kwargs), shuffle=True, **dataloader_kwargs)
        if DEBUG: print('DEBUG: Loading validation dataset (source)')
        self.source_val_dataloader = data.DataLoader(source_data_gen(args, split='val', **source_data_kwargs), shuffle=False, **dataloader_kwargs)

        if DEBUG: print('DEBUG: Loading training dataset (target)')
        self.target_dataloader = data.DataLoader(City_Dataset(args, split='train', **target_data_kwargs), shuffle=True, **dataloader_kwargs)
        if DEBUG: print('DEBUG: Loading validation dataset (target)')
        target_data_set = City_Dataset(args, split='val', **target_data_kwargs)
        self.target_val_dataloader = data.DataLoader(target_data_set, shuffle=False, **dataloader_kwargs)

        self.dataloader.val_loader = self.target_val_dataloader
        self.dataloader.valid_iterations = (len(target_data_set) + self.args.batch_size) // self.args.batch_size

        self.ignore_index = -1
        self.current_round = self.args.init_round
        self.round_num = self.args.round_num

        ### LOSSES ###
        self.feat_reg_ST_loss = feat_reg_ST_loss(ignore_index=-1,
                                                 num_class=self.args.num_classes,
                                                 device = self.device)
        self.feat_reg_ST_loss.to(self.device)

        self.use_em_loss = self.args.lambda_entropy != 0.
        if self.use_em_loss:
            self.entropy_loss = IW_MaxSquareloss(ignore_index=-1,
                                                 num_class=self.args.num_classes,
                                                 ratio=self.args.IW_ratio)
            self.entropy_loss.to(self.device)

        self.consis_loss = torch.nn.KLDivLoss(reduction='mean').to(self.device)

        self.use_clustering_loss = args.lambda_cluster != 0. or args.lambdas_cluster is not None
        self.use_orthogonality_loss = args.lambda_ortho != 0.
        self.use_sparsity_loss = args.lambda_sparse != 0.

        self.loss_kwargs = {}

        if self.use_clustering_loss:
            self.clustering_params = {
                'norm_order': args.cluster_norm_order
            }
            self.loss_kwargs['clustering_params'] = self.clustering_params

        if self.use_orthogonality_loss:
            self.orthogonality_params = {
                'temp': args.ortho_temp
            }
            self.loss_kwargs['orthogonality_params'] = self.orthogonality_params

        if self.use_sparsity_loss:
            self.sparsity_params = {
                'norm_order': args.sparse_norm_order,
                'rho': args.sparse_rho
            }
            self.loss_kwargs['sparsity_params'] = self.sparsity_params


        self.best_MIou, self.best_iter, self.current_iter, self.current_epoch = None, None, None, None

        self.epoch_num = None

        self.teacher_model, _ = get_model(self.args)
        self.teacher_model = nn.DataParallel(self.teacher_model)
        self.teacher_model.to(self.device)

    def load_checkpoint_model(self, model, filename):
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename) 

            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.module.load_state_dict(checkpoint)

            self.logger.info("Checkpoint loaded successfully from " + filename)
        except OSError as e:
            self.logger.info("No checkpoint exists")


    def main(self):
        # display args details
        self.logger.info("Global configuration as follows:")
        for key, val in vars(self.args).items():
            self.logger.info("{:25} {}".format(key, val))

        # choose cuda
        current_device = torch.cuda.current_device()
        self.logger.info("This model will run on {}".format(torch.cuda.get_device_name(current_device)))

        # load pretrained checkpoint
        if self.args.pretrained_ckpt_file is not None:
            if os.path.isdir(self.args.pretrained_ckpt_file):
                self.args.pretrained_ckpt_file = os.path.join(self.args.checkpoint_dir, self.train_id + 'final.pth')
            self.load_checkpoint(self.args.pretrained_ckpt_file)
            self.load_checkpoint_model(self.teacher_model, self.args.pretrained_ckpt_file)

        if not self.args.continue_training:
            self.best_MIou, self.best_iter, self.current_iter, self.current_epoch = 0, 0, 0, 0

        if self.args.continue_training:
            self.load_checkpoint(os.path.join(self.args.checkpoint_dir, self.train_id + 'final.pth'))

        self.args.iter_max = self.dataloader.num_iterations * self.args.epoch_each_round * self.round_num
        self.logger.info('Iter max: {} \nNumber of iterations: {}'.format(self.args.iter_max, self.dataloader.num_iterations))

        # train
        self.train_round()

        self.writer.close()


    def train_round(self):
        for r in range(self.current_round, self.round_num):
            self.logger.info("\n############## Begin {}/{} Round! #################\n".format(self.current_round + 1, self.round_num))
            self.logger.info("epoch_each_round: {}".format(self.args.epoch_each_round))

            self.epoch_num = (self.current_round + 1) * self.args.epoch_each_round

            if self.current_round > 0:
                saved_state_dict =  torch.load(os.path.join(self.args.checkpoint_dir, self.train_id + 'best.pth'))
                if 'state_dict' in saved_state_dict:
                    self.teacher_model.load_state_dict(saved_state_dict['state_dict'])
                    self.model.load_state_dict(saved_state_dict['state_dict'])
                else:
                    self.teacher_model.load_state_dict(saved_state_dict)
                    self.model.load_state_dict(saved_state_dict)


            for k, v in self.model.named_parameters():
                if "layer6" in k:
                    v.requires_grad = True
                else:
                    v.requires_grad = True

            #self.validate()
            self.train()

            self.current_round += 1


    def train_one_epoch(self):

        self.logger.info("Training one epoch...")
        self.Eval.reset()


        # Set the model to be in training mode (for batchnorm and dropout)
        if self.args.freeze_bn:  # default False
            self.model.eval()
            self.logger.info("freeze bacth normalization successfully!")
        else:
            self.model.train()
            self.teacher_model.eval()

        sourceloader_iter = enumerate(self.source_dataloader)
        targetloader_iter = enumerate(self.target_dataloader)

        ### Logging setup ###
        log_list, log_strings = [None], ['Source_loss_ce']
        if self.use_em_loss:
            log_strings.append('EM_loss')
            log_list.append(None)
        if self.use_clustering_loss:
            log_strings += ['Cluster_loss', 'c_dist', 'f_dist_source']
            log_list += [None] * 4
        if self.use_orthogonality_loss:
            log_strings += ['Ortho_loss']
            log_list += [None]
        if self.use_sparsity_loss:
            log_strings += ['Sparse_loss']
            log_list += [None]
        log_string = 'epoch{}-batch-{}:' + '={:3f}-'.join(log_strings) + '={:3f}'

        batch_idx = 0

        self.poly_lr_scheduler(optimizer=self.optimizer, init_lr=self.args.lr)
        self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]["lr"], self.current_iter)

        if self.current_iter < 1: memory_check('Start (step)')

        _, batch_t = targetloader_iter.__next__()
        x_target, y_target, _, _, _ = batch_t
        x_target = Variable(x_target).to(self.device) 
        y_target = Variable(y_target).to(self.device)

        with torch.no_grad():
            pred, t_res, t_style = self.model(x_target)
            plabel = torch.argmax(F.softmax(pred, dim=1), dim=1)

        
        SourceDataSet = [] 
        SourceLabelSet = []
        CrossEntropy = []
        for i_iter in range(100):
            _, batch_s = sourceloader_iter.__next__()
            x_source, y_source, x_source_aug, y_source_aug, idx = batch_s
            x_source, y_source = Variable(x_source).to(self.device), Variable(y_source).to(device=self.device, dtype=torch.long) 
            x_source_aug, y_source_aug = Variable(x_source_aug).to(self.device), Variable(y_source_aug).to(device=self.device, dtype=torch.long) 

            u = torch.unique(y_source)

            with torch.no_grad():
                pred_s, res_s, _ = self.teacher_model(x_source, style=t_style)

                output_proto = feat_prototype_distance(self.prototypes, res_s, self.args.num_classes)
                weight_proto = get_prototype_weight(output_proto, 1)
                weight_proto = F.interpolate(weight_proto, size=pred_s.size()[2:], mode='bilinear', align_corners=True)
                pred_s  = weight_proto * F.softmax(pred_s, dim=1)

                y = torch.squeeze(y_source, 1)
                ce = self.loss(pred_s, y)
                CrossEntropy.append(ce)
                SourceDataSet.append(x_source)
                SourceLabelSet.append(y_source)

        CrossEntropy = torch.tensor(CrossEntropy)
        value, indices = torch.sort(CrossEntropy, descending=True)
        
        for i, idx in enumerate(indices[:4]):

            x_source, y_source = SourceDataSet[idx], SourceLabelSet[idx]
          
            pred_source, res_s, s_style = self.model(x_source, style=t_style)        
            #pred_source_aug, res_s_aug, s_style_aug = self.model(x_source_aug, style=t_style)        

            ### prototypes prediction
            if self.args.use_proto:
                output_proto = feat_prototype_distance(self.prototypes, res_s, self.args.num_classes)
                weight_proto = get_prototype_weight(output_proto, 1)
                weight_proto = F.interpolate(weight_proto, size=pred_source.size()[2:], mode='bilinear', align_corners=True)
                pred_source  = weight_proto * F.softmax(pred_source, dim=1)

                # update prototypes
                update_prototypes(res_s.detach(), y_source.detach(), self.prototypes, self.prototypes_num, self.args.num_classes, 0.8)

            y_source = torch.squeeze(y_source, 1)
            loss = self.loss(pred_source, y_source)

            loss_ = loss

            if self.current_iter < 1: memory_check('Model Source')

            retain_graph = self.use_clustering_loss or self.use_orthogonality_loss or self.use_sparsity_loss
            loss_.backward(retain_graph=retain_graph)

            # log
            log_ind = 0
            log_list[log_ind] = loss.item()
            log_ind += 1

            if self.current_iter < 1: memory_check('End Source')

        #######################
        # Target forward step #
        #######################

        if self.current_iter < 1: memory_check('Dataloader Target')

        ########################
        # model output ->  list of:  1) pred; 2) feat from encoder's output
        pred_target, res_t, _ = self.model(x_target)  # creates the graph

        ########################

        if self.current_iter < 1: memory_check('Model Target')

        #####################
        # Adaptation Losses #
        #####################

        if self.use_em_loss:
            em_loss = self.args.lambda_entropy * self.entropy_loss(pred_target, F.softmax(pred_target, dim=1))
            retain_graph = self.use_clustering_loss or self.use_orthogonality_loss or self.use_sparsity_loss
            em_loss.backward(retain_graph=retain_graph)
            if self.current_iter < 1: memory_check('Entropy Loss')
            # log
            log_list[log_ind] = em_loss.item()
            log_ind += 1

        # Set some inputs to the adaptation modules
        self.loss_kwargs['source_prob'] = F.softmax(pred_source, dim=1)
        self.loss_kwargs['target_prob'] = F.softmax(pred_target, dim=1)
        self.loss_kwargs['source_feat'] = res_s
        self.loss_kwargs['target_feat'] = res_t
        self.loss_kwargs['smo_coeff'] = args.centroid_smoothing

        if self.args.use_source_gt: self.loss_kwargs['source_gt'] = y_source

        # Pass the input dict to the adaptation full loss
        loss_dict = self.feat_reg_ST_loss(**self.loss_kwargs)

        if self.use_clustering_loss:
            c_dist, f_dist_source = loss_dict['c_dist'], loss_dict['f_dist_source']
            if self.args.lambdas_cluster is not None:
                cluster_loss = self.args.lambda_feat_ds * f_dist_source - self.args.lambda_feat_dc * c_dist
            else:
                cluster_loss = self.args.lambda_cluster * (f_dist_source - c_dist)
            retain_graph = self.use_orthogonality_loss or self.use_sparsity_loss
            cluster_loss.backward(retain_graph=retain_graph)
            if self.current_iter < 1: memory_check('Clustering Loss')
            # log
            log_list[log_ind:log_ind + 3] = [cluster_loss.item(), c_dist.item(), f_dist_source.item()]
            log_ind += 3

        if self.use_orthogonality_loss:
            ortho_loss = self.args.lambda_ortho * loss_dict['ortho_loss']
            retain_graph = self.use_sparsity_loss
            ortho_loss.backward(retain_graph=retain_graph)
            if self.current_iter < 1: memory_check('Orthogonality Loss')
            # log
            log_list[log_ind] = ortho_loss.item()
            log_ind += 1

        if self.use_sparsity_loss:
            sparse_loss = self.args.lambda_sparse * loss_dict['sparse_loss']
            sparse_loss.backward()
            if self.current_iter < 1: memory_check('Sparsity Loss')
            # log
            log_list[log_ind] = sparse_loss.item()
            log_ind += 1

        self.optimizer.step()
        self.optimizer.zero_grad()


        # logging
        if batch_idx % self.args.logging_interval == 0:
            self.logger.info(log_string.format(self.current_epoch, batch_idx, *log_list))
            for name, elem in zip(log_strings, log_list):
                self.writer.add_scalar(name, elem, self.current_iter)

        batch_idx += 1

        self.current_iter += 1

        if self.current_iter < 1: memory_check('End (step)')


        #tqdm_epoch.close()

        # eval on source domain
        #self.validate_source()

        if self.args.save_inter_model:
            self.logger.info("Saving model of epoch {} ...".format(self.current_epoch))
            self.save_checkpoint(self.train_id + '_epoch{}.pth'.format(self.current_epoch))



def add_UDA_train_args(arg_parser):

    # shared
    arg_parser.add_argument('--use_source_gt', default=False, type=str2bool, help='use source label or segmented image for pixel/feature classification')
    arg_parser.add_argument('--centroid_smoothing', default=-1, type=float, help="centroid smoothing coefficient, negative to disable")
    arg_parser.add_argument('--source_dataset', default='gta5', type=str, choices=['gta5', 'synthia'], help='source dataset choice')
    arg_parser.add_argument('--source_split', default='train', type=str, help='source datasets split')
    arg_parser.add_argument('--init_round', type=int, default=0, help='init_round')
    arg_parser.add_argument('--round_num', type=int, default=1, help="num round")
    arg_parser.add_argument('--epoch_each_round', type=int, default=2, help="epoch each round")

    arg_parser.add_argument('--logging_interval', type=int, default=1, help="interval in steps for logging")
    arg_parser.add_argument('--save_inter_model', type=str2bool, default=False, help="save model at the end of each epoch or not")


    # clustering
    arg_parser.add_argument('--lambda_cluster', default=0., type=float, help="lambda of clustering loss")
    arg_parser.add_argument('--lambdas_cluster', default=None, type=str, help="lambda intra-domain source, lambda intra-domain target, lambda inter-domain")
    arg_parser.add_argument('--cluster_norm_order', default=1, type=int, help="norm order of feature clustering loss")

    # orthogonality
    arg_parser.add_argument('--lambda_ortho', default=0., type=float, help="lambda of orthogonality loss")
    arg_parser.add_argument('--ortho_temp', default=1., type=float, help="temperature for similarity based-distribution")

    # sparsity
    arg_parser.add_argument('--lambda_sparse', default=0., type=float, help="lambda of sparsity loss")
    arg_parser.add_argument('--sparse_norm_order', default=2., type=float, help="sparsity loss exponent")
    arg_parser.add_argument('--sparse_rho', default=0.5, type=float, help="sparsity loss constant threshold")

    # off-the-shelf entropy loss
    arg_parser.add_argument('--lambda_entropy', type=float, default=0., help="lambda of target loss")
    arg_parser.add_argument('--IW_ratio', type=float, default=0.2, help='the ratio of image-wise weighting factor')

    return arg_parser



def init_UDA_args(args):

    def str2none(l):
        l = [l] if not isinstance(l,list) else l
        for i,el in enumerate(l):
            if el == 'None':
                l[i]=None
        return l if len(l)>1 else l[0]

    def str2float(l):
        for i,el in enumerate(l):
            try: l[i] = float(el)
            except (ValueError,TypeError): l[i] = el
        return l

    # clustering
    if isinstance(args.lambdas_cluster, str):
        args.lambda_feat_ds, args.lambda_feat_dt, args.lambda_feat_dc = str2float(args.lambdas_cluster.split(','))

    return args



if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('1.0.0'), 'PyTorch>=1.0.0 is required'

    file_os_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(file_os_dir)
    os.chdir('..')

    arg_parser = argparse.ArgumentParser()
    arg_parser = add_train_args(arg_parser)
    arg_parser = add_UDA_train_args(arg_parser)

    args = arg_parser.parse_args()
    args, train_id, logger = init_args(args)
    args = init_UDA_args(args)
    args.source_data_path = datasets_path[args.source_dataset]['data_root_path']
    args.source_list_path = datasets_path[args.source_dataset]['list_path']

    args.target_dataset = args.dataset

    train_id = str(args.source_dataset)+"2"+str(args.target_dataset)

    assert (args.source_dataset == 'synthia' and args.num_classes == 16) or (args.source_dataset == 'gta5' and args.num_classes == 19), 'dataset:{0:} - classes:{1:}'.format(args.source_dataset, args.num_classes)

    agent = UDATrainer(args=args, cuda=True, train_id=train_id, logger=logger)
    agent.main()