# Third party
import importlib
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from matplotlib import pyplot as plt

# This project
from runners import utils
from datasets import utils as ds_utils



class RunnerWrapper(nn.Module):
    @staticmethod
    def get_args(parser):
        # Networks used in train and test
        parser.add('--networks_train',       default = 'identity_embedder, texture_generator, keypoints_embedder, inference_generator, discriminator', 
                                             help    = 'order of forward passes during the training of gen (or gen and dis for sim sgd)')

        parser.add('--networks_test',        default = 'identity_embedder, texture_generator, keypoints_embedder, inference_generator', 
                                             help    = 'order of forward passes during testing')

        parser.add('--networks_calc_stats',  default = 'identity_embedder, texture_generator, keypoints_embedder, inference_generator', 
                                             help    = 'order of forward passes during stats calculation')

        parser.add('--networks_to_train',    default = 'identity_embedder, texture_generator, keypoints_embedder, inference_generator, discriminator', 
                                             help    = 'names of networks that are being trained')

        # Losses used in train and test
        parser.add('--losses_train',         default = 'adversarial, feature_matching, perceptual, pixelwise, segmentation, warping_regularizer', 
                                             help    = 'losses evaluated during training')

        parser.add('--losses_test',          default = 'lpips, csim', 
                                             help    = 'losses evaluated during testing')

        # Spectral norm options
        parser.add('--spn_networks',         default = 'identity_embedder, texture_generator, keypoints_embedder, inference_generator, discriminator',
                                             help    = 'networks to apply spectral norm to')

        parser.add('--spn_exceptions',       default = '',
                                             help    = 'a list of exceptional submodules that have spectral norm removed')

        parser.add('--spn_layers',           default = 'conv2d, linear',
                                             help    = 'layers to apply spectral norm to')

        # Weight averaging options
        parser.add('--wgv_mode',             default = 'none', 
                                             help    = 'none|running_average|average -- exponential moving averaging or weight averaging')

        parser.add('--wgv_momentum',         default = 0.999,  type=float, 
                                             help    = 'momentum value in EMA weight averaging')

        # Training options
        parser.add('--eps',                  default = 1e-7,   type=float)

        parser.add('--optims',               default = 'identity_embedder: adam, texture_generator: adam, keypoints_embedder: adam, inference_generator: adam, discriminator: adam',
                                             help    = 'network_name: optimizer')

        parser.add('--lrs',                  default = 'identity_embedder: 2e-4, texture_generator: 2e-4, keypoints_embedder: 2e-4, inference_generator: 2e-4, discriminator: 2e-4',
                                             help    = 'learning rates for each network')

        parser.add('--stats_calc_iters',     default = 100,    type=int, 
                                             help    = 'number of iterations used to calculate standing statistics')

        parser.add('--num_visuals',          default = 32,     type=int, 
                                             help    = 'the total number of output visuals')

        parser.add('--bn_momentum',          default = 1.0,    type=float, 
                                             help    = 'momentum of the batchnorm layers')

        parser.add('--adam_beta1',           default = 0.5,    type=float, 
                                             help    = 'beta1 (momentum of the gradient) parameter for Adam')

        args, _ = parser.parse_known_args()

        # Add args from the required networks
        networks_names = list(set(
            utils.parse_str_to_list(args.networks_train, sep=',')
            + utils.parse_str_to_list(args.networks_test, sep=',')))
        for network_name in networks_names:
            importlib.import_module(f'networks.{network_name}').NetworkWrapper.get_args(parser)
        
        # Add args from the losses
        losses_names = list(set(
            utils.parse_str_to_list(args.losses_train, sep=',')
            + utils.parse_str_to_list(args.losses_test, sep=',')))
        for loss_name in losses_names:
            importlib.import_module(f'losses.{loss_name}').LossWrapper.get_args(parser)

        return parser

    def __init__(self, args, training=True):
        super(RunnerWrapper, self).__init__()
        # Store general options
        self.args = args
        self.training = training

        # Read names lists from the args
        self.load_names(args)

        # Initialize classes for the networks
        nets_names = self.nets_names_test
        if self.training:
            nets_names += self.nets_names_train
        nets_names = list(set(nets_names))

        self.nets = nn.ModuleDict()

        for net_name in sorted(nets_names):
            self.nets[net_name] = importlib.import_module(f'networks.{net_name}').NetworkWrapper(args)

            if args.num_gpus > 1:
                # Apex is only needed for multi-gpu training
                from apex import parallel

                self.nets[net_name] = parallel.convert_syncbn_model(self.nets[net_name])

        # Set nets that are not training into eval mode
        for net_name in self.nets.keys():
            if net_name not in self.nets_names_to_train:
                self.nets[net_name].eval()

        # Initialize classes for the losses
        if self.training:
            losses_names = list(set(self.losses_names_train + self.losses_names_test))
            self.losses = nn.ModuleDict()

            for loss_name in sorted(losses_names):
                self.losses[loss_name] = importlib.import_module(f'losses.{loss_name}').LossWrapper(args)

        # Spectral norm
        if args.spn_layers:
            spn_layers = utils.parse_str_to_list(args.spn_layers, sep=',')
            spn_nets_names = utils.parse_str_to_list(args.spn_networks, sep=',')

            for net_name in spn_nets_names:
                self.nets[net_name].apply(lambda module: utils.spectral_norm(module, apply_to=spn_layers, eps=args.eps))

            # Remove spectral norm in modules in exceptions
            spn_exceptions = utils.parse_str_to_list(args.spn_exceptions, sep=',')

            for full_module_name in spn_exceptions:
                if not full_module_name:
                    continue

                parts = full_module_name.split('.')

                # Get the module that needs to be changed
                module = self.nets[parts[0]]
                for part in parts[1:]:
                    module = getattr(module, part)

                module.apply(utils.remove_spectral_norm)

        # Weight averaging
        if args.wgv_mode != 'none':
            # Apply weight averaging only for networks that are being trained
            for net_name, _ in self.nets_names_to_train:
                self.nets[net_name].apply(lambda module: utils.weight_averaging(module, mode=args.wgv_mode, momentum=args.wgv_momentum))

        # Check which networks are being trained and put the rest into the eval mode
        for net_name in self.nets.keys():
            if net_name not in self.nets_names_to_train:
                self.nets[net_name].eval()

        # Set the same batchnorm momentum accross all modules
        if self.training:
            self.apply(lambda module: utils.set_batchnorm_momentum(module, args.bn_momentum))

        # Store a history of losses and images for visualization
        self.losses_history = {
            True: {}, # self.training = True
            False: {}}

    def forward(self, data_dict):
        ### Set lists of networks' and losses' names ###
        if self.training:
            nets_names = self.nets_names_train
            networks_to_train = self.nets_names_to_train

            losses_names = self.losses_names_train

        else:
            nets_names = self.nets_names_test
            networks_to_train = []

            losses_names = self.losses_names_test

        # Forward pass through all the required networks
        self.data_dict = data_dict
        for net_name in nets_names:
            self.data_dict = self.nets[net_name](self.data_dict, networks_to_train, self.nets)

        # Forward pass through all the losses
        losses_dict = {}
        for loss_name in losses_names:
            if hasattr(self, 'losses') and loss_name in self.losses.keys():
                losses_dict = self.losses[loss_name](self.data_dict, losses_dict)

        # Calculate the total loss and store history
        loss = self.process_losses_dict(losses_dict)

        return loss

    ########################################################
    #                     Utility functions                #
    ########################################################

    def load_names(self, args):
        # Initialize utility lists and dicts for the networks
        self.nets_names_to_train = utils.parse_str_to_list(args.networks_to_train)
        self.nets_names_train = utils.parse_str_to_list(args.networks_train)
        self.nets_names_test = utils.parse_str_to_list(args.networks_test)
        self.nets_names_calc_stats = utils.parse_str_to_list(args.networks_calc_stats)

        # Initialize utility lists and dicts for the networks
        self.losses_names_train = utils.parse_str_to_list(args.losses_train)
        self.losses_names_test = utils.parse_str_to_list(args.losses_test)

    def get_optimizers(self, args):
        # Initialize utility lists and dicts for the optimizers
        nets_optims_names = utils.parse_str_to_dict(args.optims)
        nets_lrs = utils.parse_str_to_dict(args.lrs, value_type=float)

        # Initialize optimizers
        optims = {}

        for net_name, optim_name in nets_optims_names.items():
            # Prepare the options
            lr = nets_lrs[net_name]
            optim_name = optim_name.lower()
            params = self.nets[net_name].parameters()

            # Choose the required optimizer
            if optim_name == 'adam':
                opt = optim.Adam(params, lr=lr, eps=args.eps, betas=(args.adam_beta1, 0.999))

            elif optim_name == 'sgd':
                opt = optim.SGD(params, lr=lr)

            elif optim_name == 'fusedadam':
                from apex import optimizers

                opt = optimizers.FusedAdam(params, lr=lr, eps=args.eps, betas=(args.adam_beta1, 0.999))

            elif optim_name == 'fusedsgd':
                from apex import optimizers

                opt = optimizers.FusedSGD(params, lr=lr)

            elif optim_name == 'lbfgs':
                opt = optim.LBFGS(params, lr=lr)

            else:
                raise 'Unsupported optimizer name'

            optims[net_name] = opt

        return optims

    def process_losses_dict(self, losses_dict):
        # This function appends loss value into losses_dict
        loss = torch.zeros(1)
        if self.args.num_gpus > 0:
            loss = loss.cuda()

        for key, value in losses_dict.items():
            if key not in self.losses_history[self.training]: 
                self.losses_history[self.training][key] = []
            
            self.losses_history[self.training][key] += [value.item()]
            loss += value
            
        return loss

    def output_losses(self):
        losses = {}

        for key, values in self.losses_history[self.training].items():
            value = torch.FloatTensor(values)

            # Average the losses
            losses[key] = value.cpu().mean()

        # Clear losses hist
        self.losses_history[self.training] = {}

        if self.args.rank != 0:
            return None
        else:
            return losses

    def output_visuals(self):
        # This function creates an output grid of visuals
        visuals_data_dict = {}

        # Only first source and target frame is visualized
        for k, v in self.data_dict.items():
            if isinstance(v, torch.Tensor):
                visuals_data_dict[k] = v[:self.args.num_visuals, 0]

        # Collect the visuals from all submodules
        visuals = []
        for net_name in self.nets_names_train:
            visuals += self.nets[net_name].visualize_outputs(visuals_data_dict)

        visuals = torch.cat(visuals, 3) # cat w.r.t. width
        visuals = torch.cat(visuals.split(1, 0), 2)[0] # cat batch dim in lines w.r.t. height
        visuals = (visuals + 1.) * 0.5 # convert back to [0, 1] range
        visuals = visuals.clamp(0, 1)

        return visuals.cpu()

    def train(self, mode=True):
        self.training = mode
        # Only change the mode of modules thst are being trained
        for net_name in self.nets_names_to_train:
            if net_name in self.nets.keys():
                self.nets[net_name].train(mode)

        return self

    def calculate_batchnorm_stats(self, train_dataloader, debug=False):
        for net_name in self.nets_names_calc_stats:
            self.nets[net_name].apply(utils.stats_calculation)

            # Set spectral norm and weight averaging to eval
            def set_modules_to_eval(module):
                if 'BatchNorm' in module.__class__.__name__:
                    return module

                else:
                    module.eval()

                    return module

            self.nets[net_name].apply(set_modules_to_eval)

        for i, self.data_dict in enumerate(train_dataloader, 1):            
            # Prepare input data
            if self.args.num_gpus > 0:
                for key, value in self.data_dict.items():
                    self.data_dict[key] = value.cuda()

            # Forward pass
            with torch.no_grad():
                for net_name in self.nets_names_calc_stats:
                    self.data_dict = self.nets[net_name](self.data_dict, [], self.nets)

            # Break if the required number of iterations is done
            if i == self.args.stats_calc_iters:
                break

            # Do only one iteration in case of debugging
            if debug and i == 10:
                break

        # Merge the buffers into running stats
        for net_name in self.nets_names_calc_stats:
            self.nets[net_name].apply(utils.remove_stats_calculation)
