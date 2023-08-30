import argparse

def get_opts():
    parser = argparse.ArgumentParser()
    #########################
    ### train related    
    parser.add_argument('--val_size', type=int, default=5000,
                        help='size of validation set')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='number of batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers for data loader')
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint path to load')
    
    #########################
    ### optimizer related
    parser.add_argument("--adam_beta1", type=float, default=0.9, 
                        help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, 
                    help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, 
                        help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, 
                        help="Epsilon value for the Adam optimizer")
    
    #########################
    ### model related
    parser.add_argument('--channels', type=int, default=320,
                        help='base channel count for the model')
    parser.add_argument('--in_channels', type=int, default=4,
                        help='number of channels in the input feature map')
    parser.add_argument('--pretrained_model_name_or_path', type=str, default='CompVis/stable-diffusion-v1-4',
                        help='base SD model chosen')
    parser.add_argument('--out_channels', type=int, default=4,
                        help='number of channels in the output feature map')
    parser.add_argument('--attention_levels')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='number of attention heads in the transformers')
    parser.add_argument('--tf_layers', type=int, default=1,
                        help='number of transformer layers in the transformers')
    parser.add_argument('--d_cond', type=int, default=768,
                        help='size of the conditional embedding in the transformers')
    
    #########################
    ### data related
    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')
    parser.add_argument('--size', type=int, default=256,
                        help='input image size(height==width)')
    

    return parser.parse_args()