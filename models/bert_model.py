from torch.nn import LayerNorm
import torch.nn.functional as F
import torch.nn as nn
import torch
import math
import json
import copy
import numpy as np
from tqdm import trange

def gelu(x):
    '''gelu激活函数'''
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def sigmoid(x):
    '''sigmoid激活函数'''
    return x * torch.sigmoid(x)


def activations(hidden_act):
    '''选择激活函数'''
    hidden_acts = {"gelu": gelu, "relu": F.relu, "sigmoid": sigmoid}
    return hidden_acts[hidden_act]


def get_mask(input_ids, nhead, is_ulm=False):
    ''' 
        输出的mask的尺寸[batch_size, from_seq_length, to_seq_length]的尺寸，maskT*mask再扩至batch_size维即可
        改进：用torch.extend()或矩阵叉乘
    '''
    batch_size = input_ids.shape[0]
    mask = (input_ids != 0).long()
    mask_0 = mask[0].unsqueeze(dim=0)
    output_mask = torch.mm(torch.transpose(mask_0, 0, 1), mask_0)
    if is_ulm:
        # 取下三角矩阵
        output_mask = torch.tril(output_mask, 0)
    output_mask = output_mask.unsqueeze(dim=0)
    for i in range(batch_size-1):
        mask_i = mask[i+1].unsqueeze(dim=0)
        mask_i = torch.mm(torch.transpose(mask_i, 0, 1), mask_i)
        if is_ulm:
            # 取下三角矩阵
            mask_i = torch.tril(mask_i, 0)
        mask_i = mask_i.unsqueeze(dim=0)
        output_mask = torch.cat((output_mask, mask_i), 0)
    output_mask = output_mask.repeat(nhead, 1, 1)
    return output_mask


class MultiheadAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout_rate, attention_scale=True,
                 return_attention_scores=False):
        super(MultiheadAttention, self).__init__()

        assert hidden_size % num_attention_heads == 0

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.attention_scale = attention_scale
        self.return_attention_scores = return_attention_scores

        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)

        self.o = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout_rate)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, attention_mask: torch.Tensor = None):

        # query shape: [batch_size, query_len, hidden_size]
        # key shape: [batch_size, key_len, hidden_size]
        # value shape: [batch_size, value_len, hidden_size]
        # 一般情况下，query_len、key_len、value_len三者相等

        mixed_query_layer = self.q(query)
        mixed_key_layer = self.k(key)
        mixed_value_layer = self.v(value)

        # mixed_query_layer shape: [batch_size, query_len, hidden_size]
        # mixed_query_layer shape: [batch_size, key_len, hidden_size]
        # mixed_query_layer shape: [batch_size, value_len, hidden_size]

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # query_layer shape: [batch_size, num_attention_heads, query_len, attention_head_size]
        # key_layer shape: [batch_size, num_attention_heads, key_len, attention_head_size]
        # value_layer shape: [batch_size, num_attention_heads, value_len, attention_head_size]

        # 交换k的最后两个维度，然后q和k执行点积, 获得attention score
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))

        # attention_scores shape: [batch_size, num_attention_heads, query_len, key_len]

        # 是否进行attention scale
        if self.attention_scale:
            attention_scores = attention_scores / \
                math.sqrt(self.attention_head_size)
        # 执行attention mask，对于mask为0部分的attention mask，
        # 值为-1e10，经过softmax后，attention_probs几乎为0，所以不会attention到mask为0的部分

        if attention_mask is not None:
            # add. 将(N, S, S)的attention_mask扩充至(N, Nhead, S, S)
            attention_mask = attention_mask.unsqueeze(dim=1)
            attention_mask = attention_mask.repeat(
                1, self.num_attention_heads, 1, 1)
            # attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e10)
            attention_mask = (1.0 - attention_mask) * -10000.0
            attention_scores = attention_scores + attention_mask

        # 将attention score 归一化到0-1
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        # context_layer shape: [batch_size, num_attention_heads, query_len, attention_head_size]

        # transpose、permute等维度变换操作后，tensor在内存中不再是连续存储的，而view操作要求tensor的内存连续存储，
        # 所以在调用view之前，需要contiguous来返回一个contiguous copy；
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # context_layer shape: [batch_size, query_len, num_attention_heads, attention_head_size]

        new_context_layer_shape = context_layer.size()[
            :-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # 是否返回attention scores
        if self.return_attention_scores:
            # 这里返回的attention_scores没有经过softmax, 可在外部进行归一化操作
            return self.o(context_layer), attention_scores
        else:
            return self.o(context_layer)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate=0.1, hidden_act='gelu', is_dropout=True):
        super(FeedForward, self).__init__()
        self.is_dropout = is_dropout
        self.dropout_rate = dropout_rate
        self.intermedia_act = activations(hidden_act)
        self.intermedia_linear = nn.Linear(d_model, d_ff)
        self.output = nn.Linear(d_ff, d_model)
        if self.is_dropout:
            self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        x = self.intermedia_act(self.intermedia_linear(x))
        if self.is_dropout:
            x = self.dropout(x)
        x = self.output(x)
        return x


class BertEmbeddings(nn.Module):
    """
        embeddings层
        构造word and position embeddings.
    """

    def __init__(self, vocab_size, d_model, max_encoder_seq, drop_rate, layer_norm_eps):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, d_model, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_encoder_seq, d_model)

        self.layerNorm = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, token_ids):
        seq_length = token_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=token_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(token_ids)

        words_embeddings = self.word_embeddings(token_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.layerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertLayer(nn.Module):
    """
        Transformer层:
        顺序为: Attention --> Add --> LayerNorm --> Feed Forward --> Add --> LayerNorm
        注意: 1、以上都不计dropout层，并不代表没有dropout，每一层的dropout使用略有不同，注意区分
              2、原始的Transformer的encoder中的Feed Forward层一共有两层linear，
              config.intermediate_size的大小不仅是第一层linear的输出尺寸，也是第二层linear的输入尺寸
    """

    def __init__(self, d_model, num_attention_heads, dropout_rate, dim_feedforward, hidden_act, is_dropout=False, eps=1e-12):
        super(BertLayer, self).__init__()
        self.multiHeadAttention = MultiheadAttention(
            d_model, num_attention_heads, dropout_rate)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layerNorm1 = LayerNorm(d_model, eps=eps)
        self.feedForward = FeedForward(
            d_model, dim_feedforward, hidden_act, is_dropout=is_dropout)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layerNorm2 = LayerNorm(d_model, eps=eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        # 为了与multiHeadAttention中query的shape对上
        # hidden_states = hidden_states.transpose(0, 1)
        self_attn_output = self.multiHeadAttention(
            hidden_states, hidden_states, hidden_states, attention_mask=attention_mask)  # 有可能需要的是attn_weights，当decoder的时候，改变后两个hidden的shape即可
        hidden_states = hidden_states + self.dropout1(self_attn_output)
        hidden_states = self.layerNorm1(hidden_states)
        self_attn_output2 = self.feedForward(hidden_states)
        hidden_states = hidden_states + self.dropout2(self_attn_output2)
        hidden_states = self.layerNorm2(hidden_states)
        # hidden_states = hidden_states.transpose(0, 1)  # 再换回来
        return hidden_states


class BERT(nn.Module):

    """
        构建BERT模型，预训练只用bert即可，微调时作为MuCS的encoder部分
    """

    def __init__(
            self,
            d_model=512,  # 隐藏层维度
            nhead=8,  # multiheadattention的头数
            vocab_size=30000,  # 输入（CODE）的词典大小
            num_layers=6,
            dim_feedforward=1024,
            dropout=0.1,
            activation="relu",
            layer_norm_eps=1e-05,
            max_seq=256,  # 输入序列的最大长度
            initializer_range=0.02,  # 权重初始化方差
            with_pool=True,  # 是否包含Pool部分
            with_mlm=True,  # 是否包含MLM部分
            with_ulm=True,  # 是否包含ULM部分
            scp_cls=6,  # scp分类个数
            awp_cls=40  # awp分类个数
    ):
        self.d_model = d_model
        self.nhead = nhead
        self.vocab_size = vocab_size
        self.max_seq = max_seq
        self.initializer_range = initializer_range
        self.with_pool = with_pool
        self.with_mlm = with_mlm
        self.with_ulm = with_ulm
        self.scp_cls = scp_cls
        self.awp_cls = awp_cls
        self.dropout = dropout
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        super(BERT, self).__init__()
        self.embeddings = BertEmbeddings(
            self.vocab_size, self.d_model, self.max_seq, self.dropout, self.layer_norm_eps)
        layer = BertLayer(self.d_model, self.nhead, self.dropout,
                          self.dim_feedforward, self.activation, is_dropout=False, eps=self.layer_norm_eps)
        self.encoderLayer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(self.num_layers)])
        if self.with_pool:
            # Pooler部分（提取CLS向量）
            self.pooler = nn.Linear(
                self.d_model, self.d_model)  # Linear(a,a)的意义是什么？
            self.pooler_activation = nn.Tanh()
            self.awp_pooler = nn.Linear(self.d_model, awp_cls)
            self.scp_pooler = nn.Linear(self.d_model, scp_cls)
        else:
            self.pooler = None
            self.pooler_activation = None
        if self.with_mlm:
            self.mlmDecoder = nn.Linear(
                self.d_model, self.vocab_size, bias=False)
            # self.mlmDecoder.weight = self.embeddings.word_embeddings.weight
            self.mlmBias = nn.Parameter(torch.zeros(self.vocab_size))
            self.mlmDecoder.bias = self.mlmBias
            self.mlmDense = nn.Linear(self.d_model, self.d_model)
            self.transform_act_fn = activations(self.activation)
            self.mlmLayerNorm = LayerNorm(
                self.d_model, eps=self.layer_norm_eps)
        self.apply(self.init_model_weights)

    def init_model_weights(self, module):
        """ 初始化权重
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # bert参数初始化, tf版本在linear和Embedding层使用的是截断正态分布, pytorch没有实现该函数,
            # 此种初始化对于加载预训练模型后进行finetune没有任何影响，
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, token_ids, attention_mask=None, output_all_encoded_layers=False):
        """
            token_ids： 一连串token在vocab中对应的id
            attention_mask：各元素的值为0或1,避免在padding的token上计算attention, 1进行attetion, 0不进行attention
            以上两个参数的shape为： (batch_size, sequence_length); type为tensor
        """
        # ULM时 需要修改mask变成下三角矩阵
        if attention_mask is None:
            attention_mask = get_mask(token_ids, self.nhead, self.with_ulm)

        # 兼容fp16
        attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
        # 对mask矩阵中，数值为0的转换成很大的负数，使得不需要attention的位置经过softmax后,分数趋近于0
        # attention_mask = (1.0 - attention_mask) * -10000.0
        # 执行embedding
        hidden_states = self.embeddings(token_ids)
        # 执行encoder
        encoded_layers = [hidden_states]  # 添加embedding的输出
        for layer_module in self.encoderLayer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                encoded_layers.append(hidden_states)
        if not output_all_encoded_layers:
            encoded_layers.append(hidden_states)

        sequence_output = encoded_layers[-1]

        # 是否取最后一层输出
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        # 是否添加pool层
        if self.with_pool:
            pooled_output = self.pooler_activation(
                self.pooler(sequence_output[:, 0]))
            awp = self.awp_pooler(pooled_output)
            scp = self.scp_pooler(pooled_output)  # 做消融实验的时候 需要分开来写
        else:
            pooled_output = None
            awp = None
            scp = None
        # 是否添加mlm
        if self.with_mlm:
            mlm_hidden_state = self.mlmDense(sequence_output)
            mlm_hidden_state = self.transform_act_fn(mlm_hidden_state)
            mlm_hidden_state = self.mlmLayerNorm(mlm_hidden_state)
            mlm_scores = self.mlmDecoder(mlm_hidden_state)
        else:
            mlm_scores = None
        # 根据情况返回值，应当使其全部返回，然后择需取用
        return encoded_layers, mlm_scores, awp, scp


class MuCS(nn.Module):
    def __init__(
            self,
            encoder: BERT,  # 以BERT作为encoder
            # 以下是decoder的参数
            d_model=512,  # 隐藏层维度
            nhead=8,  # multiheadattention的头数
            num_layers=6,
            max_seq=128,  # 输出（NL）的最大长度
            NL_vocab_size=30000,  # 输出（NL）的词典大小
            activation=torch.tanh,
            eps=1e-05,
            beam_size=5,  # beam的大小
            sos_id=2,  # 开始符的id
            eos_id=3  # 结束符的id
    ):
        super(MuCS, self).__init__()
        self.encoder = encoder
        self.d_model = d_model
        self.nhead = nhead
        self.max_seq = max_seq
        self.NL_vocab_size = NL_vocab_size
        self.register_buffer("bias", torch.tril(
            torch.ones(2048, 2048)))  # 将bias放入模型缓冲区
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=num_layers)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.activation = activation
        self.dense = nn.Linear(d_model, d_model)
        self.seq2seq_dense = nn.Linear(d_model, self.NL_vocab_size)
        self.layer_norm = LayerNorm(
            self.d_model, eps=eps)
        self.beam_size = beam_size
        self.sos_id = sos_id
        self.eos_id = eos_id

    def forward(self, source_ids, target_ids=None, source_mask=None, target_mask: torch.Tensor = None):
        encoder_output, _, _, _ = self.encoder(
            source_ids, source_mask)
        encoder_output = encoder_output.contiguous()
        # encoder_output shape: [batch_size, seq_len, d_model]
        if target_ids is not None:
            # 有目标id，属于训练，输出的是seq_output
            # attn_mask = -1e4 * \
            #     (1-self.bias[:target_ids.shape[1], :target_ids.shape[1]])
            target_embeddings = self.encoder.embeddings(
                target_ids).contiguous()
            target_mask = target_mask.float()
            # add. 将(N, S, S)的target_mask扩充至(N* Nhead, S, S)
            target_mask = target_mask.repeat(self.nhead, 1, 1)
            # 少一个memory_mask
            decoder_output = self.decoder(
                target_embeddings, encoder_output, tgt_mask=target_mask)
            hidden_states = self.dense(decoder_output)
            hidden_states = self.activation(hidden_states)
            hidden_states = self.layer_norm(hidden_states)  # layer_norm自己加的
            seq_output = self.seq2seq_dense(hidden_states)
            # target_ids 为 [sos, ...]，而output为[..., eos]，因此要shift一下，并且直接在forward函数中计算loss（本实验中SOS即CLS）
            shift_output = seq_output[..., :-1, :].contiguous()
            shift_output = self.log_softmax(shift_output)
            return shift_output

        else:
            # 没有目标id，属于预测，输出的是预测的向量
            preds = []
            zero = torch.LongTensor(1).fill_(0)
            for i in range(source_ids.shape[0]):
                context = encoder_output[i:i+1, :]
                context_mask = source_mask[i:i+1, :]
                beam = Beam(self.beam_size, self.sos_id, self.eos_id)
                input_ids = beam.getCurrentState()
                context = context.repeat(self.beam_size, 1, 1) # 上下文，也即前面的输入
                context_mask = context_mask.repeat(self.beam_size, 1,1)
                for _ in range(self.max_seq):
                    if beam.done():
                        break
                    attn_mask = -1e4 * \
                        (1-self.bias[:input_ids.shape[1], :input_ids.shape[1]])
                    tgt_embeddings = self.encoder.embeddings(
                        input_ids).contiguous()
                    out = self.decoder(tgt_embeddings, context, tgt_mask=attn_mask)
                    out = torch.tanh(self.dense(out))
                    hidden_states = out.contiguous()[:, -1, :]
                    out = self.log_softmax(
                        self.seq2seq_dense(hidden_states)).data
                    beam.advance(out)
                    input_ids.data.copy_(input_ids.data.index_select(
                        0, beam.getCurrentOrigin()))
                    input_ids = torch.cat(
                        (input_ids, beam.getCurrentState()), -1)
                hyp = beam.getHyp(beam.getFinal())
                pred = beam.buildTargetTokens(hyp)[:self.beam_size]
                pred = [torch.cat([x.view(-1) for x in p]+[zero]
                                  * (self.max_seq-len(p))).view(1, -1) for p in pred]
                preds.append(torch.cat(pred, 0).unsqueeze(0))

            preds = torch.cat(preds, 0)
            return pred


class Beam(object):
    def __init__(self, size, sos, eos):
        self.size = size
        self.tt = torch
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                       .fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished = []
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i))
            unfinished.sort(key=lambda a: -a[0])
            self.finished += unfinished[:self.size-len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps = []
        for _, timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j+1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps

    def buildTargetTokens(self, preds):
        sentence = []
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok == self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence


if __name__ == "__main__":
    model_dir = "outputdir/pretraining_model/pretraining_model_ep150.pth"
    model = torch.load(model_dir)
    input = torch.ones(128)
    mask = get_mask(input,8)
    pred = model(input,mask)

