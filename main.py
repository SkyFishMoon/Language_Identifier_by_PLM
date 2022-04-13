import random
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split
from dataset_creator import LDDataset
from transformers import get_linear_schedule_with_warmup, BertForSequenceClassification
import torch.optim as optim
from typing import Optional, Sequence
import torch.nn.functional as F
from torchsummary import summary
from tqdm import tqdm
from transformers import AdamW, BertConfig
import time
from datetime import datetime
from torch.nn import CrossEntropyLoss
from model import encoder
from utils import flat_accuracy, format_time
import configargparse
import opts
from torch.utils.tensorboard import SummaryWriter
import os


def set_seeds(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def fine_tune(opt):
    set_seeds(opt.seed)
    device = 'cuda' if torch.cuda.is_available() and opt.gpu else 'cpu'

    model = encoder[opt.model_name].from_opt(opt).to(device)
    tokenizer = model.tokenizer
    dataset = LDDataset(tokenizer, opt.data_path, max_length=opt.max_length)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=opt.batch_size  # Trains with this batch size.
    )
    validation_dataloader = DataLoader(
        val_dataset,  # The validation samples.
        sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
        batch_size=opt.batch_size  # Evaluate with this batch size.
    )
    optimizer = AdamW(model.parameters(),
                      lr=opt.learning_rate,
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )
    loss_fn = CrossEntropyLoss()
    # The BERT authors recommend between 2 and 4.
    # Total number of training steps is [number of batches] x [number of epochs].
    total_steps = len(train_dataloader) * opt.epoch_number
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=opt.warmup_steps,  # Default value in run_glue.py
                                                num_training_steps=total_steps)
    tensorboard = SummaryWriter(opt.tensorboard_dir + datetime.now().strftime("/%b-%d_%H-%M-%S"))
    ckpt_dir =opt.ckpt_dir + datetime.now().strftime("%b-%d_%H-%M-%S") + '/'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    for param in model.plm.parameters():
        param.requires_grad = False

    total_t0 = time.time()
    step = 0

    # For each epoch...
    for epoch_i in range(0, opt.epoch_number):

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, opt.epoch_number))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        model.train()
        loop = tqdm(enumerate(train_dataloader), leave=False, total=len(train_dataloader))
        for idx, batch in enumerate(loop):
            #
            # if step % opt.update_every_step == 0 and not step == 0:
            #     # Calculate elapsed time in minutes.
            #     elapsed = format_time(time.time() - t0)
            #
            #     # Report progress.
            #     print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            step = step + 1
            b_input_ids = batch[1]['ids'].to(device)
            b_input_mask = batch[1]['mask'].to(device)
            b_labels = batch[1]['target'].to(device)
            # b_token_type_ids = batch[1]['token_type_ids'].to(device)
            model.zero_grad()

            logits = model(b_input_ids,
                           # token_type_ids=b_token_type_ids,
                           mask=b_input_mask)
            loss = loss_fn(logits, b_labels)

            tensorboard.add_scalar("train_loss_batch/", loss.item(), step)
            tensorboard.add_scalar("learning_rate/", optimizer.param_groups[0]['lr'], step)
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        tensorboard.add_scalar("train_loss_epoch/", avg_train_loss, epoch_i)
        training_time, _ = format_time(time.time() - t0)
        tensorboard.add_scalar("compute_time/", training_time, epoch_i)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

        print("")
        print("Running Validation...")

        t0 = time.time()

        model.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0

        for batch in validation_dataloader:

            b_input_ids = batch['ids'].to(device)
            b_input_mask = batch['mask'].to(device)
            b_labels = batch['target'].to(device)
            # b_token_type_ids = batch['token_type_ids'].to(device)

            with torch.no_grad():
                logits = model(b_input_ids,
                               # token_type_ids=b_token_type_ids,
                               mask=b_input_mask)
                loss = loss_fn(logits, b_labels)

            total_eval_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            total_eval_accuracy += flat_accuracy(logits, label_ids)


        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        tensorboard.add_scalar("valid_accurcy/", avg_val_accuracy, epoch_i)

        avg_val_loss = total_eval_loss / len(validation_dataloader)
        tensorboard.add_scalar("valid_loss/", avg_val_loss, epoch_i)

        validation_time, _ = format_time(time.time() - t0)
        tensorboard.add_scalar("valid_time/", validation_time, epoch_i)

        if epoch_i % opt.save_every_epoch == 0:
            torch.save({'epoch': epoch_i, 'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                        }, ckpt_dir + '/epoch-' + str(epoch_i) + '.pth')

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))


def _get_parser():
    parser = configargparse.ArgumentParser(
        description='main.py',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

    opts.config_opts(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)
    return parser


def main():
    parser = _get_parser()

    opt = parser.parse_args()

    fine_tune(opt)


if __name__ == "__main__":
    main()


# def finetune(epochs, dataloader, model, loss_fn, optimizer):
#     model.train()
#     for epoch in range(epochs):
#         print(epoch)
#
#         loop = tqdm(enumerate(dataloader), leave=False, total=len(dataloader))
#         for batch, dl in loop:
#             ids = dl['ids']
#             token_type_ids = dl['token_type_ids']
#             mask = dl['mask']
#             label = dl['target']
#             label = label.unsqueeze(1)
#
#             optimizer.zero_grad()
#
#             output = model(
#                 ids=ids,
#                 mask=mask,
#                 token_type_ids=token_type_ids)
#             label = label.type_as(output)
#
#             loss = loss_fn(output, label)
#             loss.backward()
#
#             optimizer.step()
#
#             pred = np.where(output >= 0, 1, 0)
#
#             num_correct = sum(1 for a, b in zip(pred, label) if a[0] == b[0])
#             num_samples = pred.shape[0]
#             accuracy = num_correct / num_samples
#
#             print(
#                 f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')
#
#             # Show progress while training
#             loop.set_description(f'Epoch={epoch}/{epochs}')
#             loop.set_postfix(loss=loss.item(), acc=accuracy)
#
#     return model
#
# model=finetune(5, dataloader, model, loss_fn, optimizer)

# def load_checkpoint(model, checkpoint_PATH, optimizer):
#     if checkpoint != None:
#         model_CKPT = torch.load(checkpoint_PATH)
#         model.load_state_dict(model_CKPT['state_dict'])
#         print('loading checkpoint!')
#         optimizer.load_state_dict(model_CKPT['optimizer'])
#     return model, optimizer