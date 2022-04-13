def config_opts(parser):
    parser.add('--config', '-config', required=False,
               is_config_file_arg=True, help='config file path')
    parser.add('--save_config', '-save_config', required=False,
               is_write_out_config_file_arg=True,
               help='config file save path')
    parser.add('--tensorboard_dir', '-tensorboard_dir', required=True, type=str)


def model_opts(parser):
    parser.add('--model_name', '-name', required=True,
               type=str, choices=['mbert', 'xlm', 'xlmr'])
    parser.add('--num_languages', '-num_languages', required=True,
               type=int)


def train_opts(parser):
    parser.add('--data_path', '-data_path', required=True, type=str)
    parser.add('--epoch_number', '-epoch_number', required=True,
               type=int)
    parser.add('--batch_size', '-batch_size', required=True,
               type=int)
    parser.add('--learning_rate', '-learning_rate', type=float, default=1.0,
               help="Starting learning rate. "
                    "Recommended settings: sgd = 1, adagrad = 0.1, "
                    "adadelta = 1, adam = 0.001")
    parser.add('--learning_rate_decay', '-learning_rate_decay',
               type=float, default=0.5,
               help="If update_learning_rate, decay learning rate by "
                    "this much if steps have gone past"
                    "start_decay_steps")
    parser.add('--start_decay_steps', '-start_decay_steps',
               type=int, default=50000,
               help="Start decaying every decay_steps after"
                    "start_decay_steps")
    parser.add('--decay_steps', '-decay_steps', type=int, default=10000,
               help="Decay every decay_steps")

    parser.add('--warmup_steps', '-warmup_steps', type=int, default=4000,
               help="Number of warmup steps for custom decay.")
    parser.add('--gpu', '-gpu', action="store_true", default=None)
    parser.add('--early_stopping', '-early_stopping', type=int, default=0,
              help='Number of validation steps without improving.')
    parser.add('--threshold_step', '-threshold_step', type=int, default=100)
    parser.add('--save_every_epoch', '-save_every_epoch', type=int, default=1)
    parser.add('--ckpt_dir', '-ckpt_dir', type=str, required=True)
    parser.add('--update_every_step', '-update_every_step', type=int, default=1)
    parser.add('--seed', '-seed', type=int, required=True)
    parser.add('--max_length', '-max_length', type=int)