import argparse
import json
import random
from bert_dataset import get_instances, BertDataset
import os
import torch
from torch.utils.data import DataLoader
from models.bert_model import MuCS
from models.trainers import FinetuningTrainer


def load_instances(instances_path, is_shuffled=False, random_seed=None):
    with open(instances_path, 'r')as f:
        instances = json.load(f)
    if is_shuffled:
        rng = random.Random(random_seed)
        rng.shuffle(instances)
    return instances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=12345,
                        help="Random seed for data generation.")
    parser.add_argument("--is_shuffled", type=bool, default=False,
                        help="Whether to shuffle the instances.")
    # parser.add_argument("-c", "--train_dataset", required=True, type=str, default="", help="train dataset for train bert")
    # parser.add_argument("-t", "--test_dataset", type=str, default=None, help="test set for evaluate train set")
    # parser.add_argument("-v", "--vocab_path", required=True, type=str, help="built vocab model      path with bert-vocab")
    parser.add_argument("-o", "--output_path", default="C:/Users/Shawnchan/Desktop/iSE/Multi-task code summerization/Transplant/outputdir/finetuning_model",
                        type=str, help="ex)output/bert.model")

    parser.add_argument("-hs", "--hidden", type=int,
                        default=256, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int,
                        default=8, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int,
                        default=8, help="number of attention heads")
    parser.add_argument("-s", "--seq_len", type=int,
                        default=256, help="maximum input sequence len")
    parser.add_argument("--output_seq_len", type=int,
                        default=128, help="maximum output sequence len")

    parser.add_argument("-b", "--batch_size", type=int,
                        default=64, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int,
                        default=200, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int,
                        default=5, help="dataloader worker size")

    parser.add_argument("--with_cuda", type=bool, default=False,
                        help="training with CUDA: true, or false")
    parser.add_argument("--with_test", type=bool, default=True,
                        help="whether to test")
    parser.add_argument("--with_predict", type=bool, default=False,
                        help="whether to predict")
    parser.add_argument("--with_ulm", type=bool, default=False,
                        help="whether to use unidirectional language model")             
    parser.add_argument("--log_freq", type=int, default=50,
                        help="printing loss every n iter: setting n")
    parser.add_argument("--cuda_devices", type=int, nargs='+',
                        default=None, help="CUDA device ids")
    parser.add_argument("--on_memory", type=bool, default=True,
                        help="Loading on memory: true or false")

    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float,
                        default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float,
                        default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float,
                        default=0.999, help="adam first beta value")

    args = parser.parse_args()
    dir_demo = "data/demo"
    dir_data = "C:/Users/Shawnchan/Desktop/iSE/Multi-task code summerization/Transplant/data/data"
    my_dir = dir_demo
    code_path = os.path.join(my_dir, "tokens.txt")
    NL_path = os.path.join(my_dir, "comment_tokens.txt")
    SCP_path = os.path.join(my_dir, "SCP.txt")
    AWP_path = os.path.join(my_dir, "AWP.txt")
    code_vocab_path = os.path.join(my_dir, "vocabs.txt")
    NL_vocab_path = os.path.join(my_dir, "comment_vocabs.txt")
    # output_path = os.path.join(my_dir,"instances.txt")
    instances = get_instances(code_path, NL_path, SCP_path, AWP_path,
                                                         code_vocab_path, NL_vocab_path, args.seq_len, args.output_seq_len, args.with_ulm)
    if args.is_shuffled:
        rng = random.Random(args.random_seed)
        rng.shuffle(instances)

    num_for_training = int(len(instances)*0.8)
    print("Creating Train Dataset")
    train_dataset = BertDataset(instances[:num_for_training])
    print("Creating Dataloader")
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size)
    
    if args.with_test:
        print("Initial test_dataset")
        test_dataset = BertDataset(instances[num_for_training:])
        print("Creating Dataloader")
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size)
    else:
        test_dataloader = None

    print("Building BERT model")
    model_dir = "C:/Users/Shawnchan/Desktop/iSE/Multi-task code summerization/Transplant/outputdir/pretraining_model/pretraining_model_ep150.pth"
    encoder = torch.load(model_dir)
    mucs = MuCS(encoder)

    print("Creating BERT Trainer")
    trainer = FinetuningTrainer(mucs, train_dataloader=train_dataloader,test_dataloader=test_dataloader,
                          lr=args.lr, betas=(
                              args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                          with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq)

    print("Training Start")
    for epoch in range(args.epochs):
        trainer.train(epoch)
        if epoch % args.log_freq == 0:
            trainer.save(epoch, args.output_path)
        if test_dataloader is not None:
            trainer.test(epoch)

if __name__ == '__main__':
    main()
