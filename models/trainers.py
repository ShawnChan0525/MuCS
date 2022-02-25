import time
from pyparsing import Word
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import bleu
from models.bert_model import BERT, MuCS
from optim_schedule import ScheduledOptim
from utils import id2word

import tqdm


class BERTTrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(self, bert: BERT,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = False, cuda_devices=None, log_freq: int = 5):
        """
        :param bert: BERT model which you want to train
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda" if cuda_condition else "cpu")

        # Initialize the BERT Language Model. This BERT model will be saved every epoch
        self.model = bert.to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1 and len(cuda_devices) > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model)
            self.model.d_model = self.model.module.d_model

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr,
                          betas=betas, weight_decay=weight_decay)
        # self.optim_schedule = ScheduledOptim(
        #     self.optim, self.model.d_model, n_warmup_steps=warmup_steps)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = nn.NLLLoss(ignore_index=0).to(self.device)

        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement()
              for p in self.model.parameters()]))

    def train(self, epoch):
        self.model.train()
        data_iter = tqdm.tqdm(enumerate(self.train_data),
                              desc="EP%d_train" % (epoch),
                              total=len(self.train_data),
                              bar_format="{l_bar}{r_bar}")

        total_awp_correct = 0
        total_scp_correct = 0
        total_ulm_correct = 0
        total_element = 0
        total_ulm_element = 0

        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the next_sentence_prediction and masked_lm model
            _, ulm_output, awp_output, scp_output = self.model.forward(
                data["input_ids"], data["src_mask"])

            ulm_loss = self.criterion(
                ulm_output.transpose(1, 2), data["input_ids"])
            # 2-1. NLL(negative log likelihood) loss of is_next classification result
            awp_loss = self.criterion(awp_output, data["awp_label"])
            scp_loss = self.criterion(scp_output, data["scp_label"])
            # 2-2. NLLLoss of predicting masked token word
            # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
            loss = ulm_loss + awp_loss + scp_loss

            # awp accuracy
            awp_correct = awp_output.argmax(
                dim=-1).eq(data["awp_label"]).sum().item()
            # scp accuracy
            scp_correct = scp_output.argmax(
                dim=-1).eq(data["scp_label"]).sum().item()
            # ulm accuracy
            ulm_correct = ulm_output.argmax(
                dim=-1).eq(data["input_ids"]).sum().item()
            # 3. backward and optimization only in train
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            total_element += data["awp_label"].nelement()
            # total_ulm_element += data["code_size"].sum().item() 也不对
            total_ulm_element += data["code_size"].nelement()
            total_awp_correct += awp_correct
            total_scp_correct += scp_correct
            total_ulm_correct += ulm_correct

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_awp_acc": total_awp_correct / total_element * 100,
                "avg_scp_acc": total_scp_correct / total_element * 100,
                "avg_ulm_acc": total_ulm_correct / total_ulm_element * 100,
                "loss": loss.item()
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))

        print("EP%d_train" % (epoch), " total_awp_acc=",
              total_awp_correct * 100.0 / total_element, " total_scp_acc=",
              total_scp_correct * 100.0 / total_element, " total_ulm_acc=",
              total_ulm_correct * 100.0 / total_ulm_element)

    def test(self, epoch, output_txt):
        self.model.eval()
        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(self.test_data),
                              desc="EP%d_test" % (epoch),
                              total=len(self.test_data),
                              bar_format="{l_bar}{r_bar}")

        total_awp_correct = 0
        total_scp_correct = 0
        total_ulm_correct = 0
        total_element = 0
        total_ulm_element = 0

        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the next_sentence_prediction and masked_lm model
            _, ulm_output, awp_output, scp_output = self.model.forward(
                data["input_ids"], data["src_mask"])

            ulm_loss = self.criterion(
                ulm_output.transpose(1, 2), data["input_ids"])
            # 2-1. NLL(negative log likelihood) loss of is_next classification result
            awp_loss = self.criterion(awp_output, data["awp_label"])
            scp_loss = self.criterion(scp_output, data["scp_label"])
            # 2-2. NLLLoss of predicting masked token word
            # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
            loss = ulm_loss + awp_loss + scp_loss

            # awp accuracy
            awp_correct = awp_output.argmax(
                dim=-1).eq(data["awp_label"]).sum().item()
            # scp accuracy
            scp_correct = scp_output.argmax(
                dim=-1).eq(data["scp_label"]).sum().item()
            # ulm accuracy
            ulm_correct = ulm_output.argmax(
                dim=-1).eq(data["input_ids"]).sum().item()
            # 3. backward and optimization only in train

            total_element += data["awp_label"].nelement()
            # total_ulm_element += data["code_size"].sum().item() 也不对
            total_ulm_element += data["code_size"].nelement()
            total_awp_correct += awp_correct
            total_scp_correct += scp_correct
            total_ulm_correct += ulm_correct

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_awp_acc": total_awp_correct / total_element * 100,
                "avg_scp_acc": total_scp_correct / total_element * 100,
                "avg_ulm_acc": total_ulm_correct / total_ulm_element * 100,
                "loss": loss.item()
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
                output_txt.write(str(post_fix))
                output_txt.write('\n')

        result = "EP%d_test" % (epoch)+" total_awp_acc=" + str(total_awp_correct * 100.0 / total_element) + " total_scp_acc="+str(
            total_scp_correct * 100.0 / total_element)+" total_ulm_acc="+str(total_ulm_correct * 100.0 / total_ulm_element)
        print(result)

    def save(self, epoch, file_path="outputdir/pretraining_model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + "/ep%d.pth" % epoch
        torch.save(self.model.to(self.device), output_path)
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path


class FinetuningTrainer:
    def __init__(self, bert: MuCS,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None, predict_dataloader:DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = False, cuda_devices=None, log_freq: int = 50):
        """
        :param bert: MuCS model which you want to train
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda" if cuda_condition else "cpu")

        # Initialize the BERT Language Model. This BERT model will be saved every epoch
        self.model = bert.to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1 and len(cuda_devices) > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model)
            self.model.d_model = self.model.module.d_model

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.predict_data = predict_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr,
                          betas=betas, weight_decay=weight_decay)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = nn.NLLLoss(ignore_index=0).to(self.device)

        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement()
              for p in self.model.parameters()]))

    def train(self, epoch):
        self.model.train()
        data_iter = tqdm.tqdm(enumerate(self.train_data),
                              desc="EP%d_train" % (epoch),
                              total=len(self.train_data),
                              bar_format="{l_bar}{r_bar}")

        total_NL_correct = 0
        total_element = 0

        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the next_sentence_prediction and masked_lm model
            seq_output = self.model.forward(
                data["input_ids"], data["output_ids"], data["src_mask"], data["tgt_mask"])

            # 2-2. NLLLoss of predicting masked token word
            shift_output = data["output_ids"][..., 1:]
            loss = self.criterion(
                seq_output.transpose(1, 2), shift_output)

            NL_correct = seq_output.argmax(
                dim=-1).eq(shift_output).sum().item()
            # 3. backward and optimization only in train
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            # 减去batch_szie的大小，因为output shift过
            # total_element += (data["NL_size"].sum().item() -
            #                   data["NL_size"].shape[0])
            total_element += (data["NL_size"].nelement() -
                              data["NL_size"].shape[0])
            total_NL_correct += NL_correct

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_NL_acc": total_NL_correct / total_element * 100,
                "loss": loss.item()
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
        print("EP%d_train" % (epoch), " total_NL_acc=",
              total_NL_correct * 100.0 / total_element)

    def test(self, epoch, output_txt):
        self.model.eval()
        data_iter = tqdm.tqdm(enumerate(self.test_data),
                              desc="EP%d_test" % (epoch),
                              total=len(self.test_data),
                              bar_format="{l_bar}{r_bar}")

        total_NL_correct = 0
        total_element = 0

        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the next_sentence_prediction and masked_lm model
            seq_output = self.model.forward(
                data["input_ids"], data["output_ids"], data["src_mask"], data["tgt_mask"])

            # 2-2. NLLLoss of predicting masked token word
            shift_output = data["output_ids"][..., 1:]
            loss = self.criterion(
                seq_output.transpose(1, 2), shift_output)

            NL_correct = seq_output.argmax(
                dim=-1).eq(shift_output).sum().item()
            # 3. backward and optimization only in train
            # 减去batch_szie的大小，因为output shift过
            total_element += (data["NL_size"].sum().item() -
                              data["NL_size"].shape[0])
            total_NL_correct += NL_correct

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_NL_acc": total_NL_correct / total_element * 100,
                "loss": loss.item()
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
                if epoch % self.log_freq == 0:
                    output_txt.write(str(post_fix))
                    output_txt.write('\n')
                
        print("EP%d_test" % (epoch), " total_NL_acc=",
              total_NL_correct * 100.0 / total_element)

    def predict(self, NL_dict):
        data_iter = tqdm.tqdm(enumerate(self.predict_data),
                              total=len(self.predict_data),
                              bar_format="{l_bar}{r_bar}")
        candidates = []
        references = []

        '''计算bleu的步骤'''
        # 1. 取每个batch中的pred[0]
        # 2. output = id2word(list(pred))
        # 3. 将output转换成string
        # 4. 生成字典predictionMap: idx, pred
        # 5. 调用bleuFromMaps()计算bleu

        for _, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}
            # batch_size*beam_size*seq_len
            preds = self.model.forward(
                data["input_ids"], source_mask=data["src_mask"])
            for ref in data['output_ids']:
                ref = list(ref.cpu().numpy())
                if 0 in ref:
                    ref = ref[:ref.index(0)]
                text = id2word(NL_dict, ref)
                ref = " ".join(text)
                references.append(ref)

            for pred in preds:
                t = pred[0].cpu().numpy()
                t = list(t)
                if 0 in t:  # 或者应该是EOSid(2)？
                    t = t[:t.index(0)]
                text = id2word(NL_dict, t)
                candidate = " ".join(text)
                candidates.append(candidate)

        # 构造字典
        dict_size = len(candidates)
        predictionMap = dict(zip(range(dict_size), candidates))
        refMap = dict(zip(range(dict_size), references))

        # 计算bleu
        bleu_score = bleu.bleuFromMaps(refMap, predictionMap)
        return bleu_score[0]

    def save(self, epoch, file_path="outputdir/finetuning_model"):
        output_path = file_path + "/ep%d.pth" % epoch
        torch.save(self.model.to(self.device), output_path)
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
