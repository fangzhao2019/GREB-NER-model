import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from model.crf import CRF
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import model.attention as attention
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Joint_Bert_BiLSTM_CRF(nn.Module):

    def __init__(self, labelIndex, GPU):
        super(Joint_Bert_BiLSTM_CRF, self).__init__()
        print("build batched Joint_Bert_BiLSTM_CRF...")
        dropout=0.3
        self.alpha=0.2
        self.embedding_dim = 768
        self.hidden_dim = 300
        self.attention=attention.BertAttention()
        self.drop = nn.Dropout(dropout)
        self.droplstm = nn.Dropout(dropout)

        # 声明LSTM
        self.bilstm_flag = True
        self.lstm_layer = 1
        if self.bilstm_flag:
            lstm_hidden = self.hidden_dim // 2#整除
        else:
            lstm_hidden = self.hidden_dim
        self.lstm = nn.LSTM(self.embedding_dim, lstm_hidden,
                            num_layers=self.lstm_layer, batch_first=True,
                            bidirectional=self.bilstm_flag)

        # 声明CRF
        self.index2label = {}#将instance2index的键值调换
        for ele in labelIndex:
            self.index2label[labelIndex[ele]] = ele
        self.hidden2tag = nn.Linear(self.hidden_dim, len(self.index2label)+2)
        self.crf = CRF(len(self.index2label), GPU)

        # 将模型载入到GPU中
        self.gpu = GPU
        if self.gpu:
            self.attention=self.attention.cuda()
            self.drop = self.drop.cuda()
            self.droplstm = self.droplstm.cuda()
            self.hidden2tag = self.hidden2tag.cuda()
            self.lstm = self.lstm.cuda()

    def _get_lstm_features(self, batch_word, batch_knowledge, knowledge_mask, batch_wordlen, dynanmic_meta_embedding):
        batch_size,max_seq_length,max_entity_num= knowledge_mask.size()

        batch_wordlen=batch_wordlen.cpu()
        if dynanmic_meta_embedding:
            batch_knowledge=batch_knowledge.view(batch_size*max_seq_length, max_entity_num, 768)
            knowledge_mask=knowledge_mask.view(batch_size*max_seq_length, max_entity_num)
            batch_knowledge=self.attention(batch_knowledge,knowledge_mask)
            batch_knowledge=batch_knowledge.view(batch_size,max_seq_length,max_entity_num,768)[:,:,-1,:]
        else:
            batch_knowledge=batch_knowledge.sum(axis=2)
        merged_batch_word=self.alpha*batch_word+(1-self.alpha)*batch_knowledge
        embeds_pack = pack_padded_sequence(merged_batch_word, batch_wordlen, batch_first=True)
        #LSTM的输出
        out_packed, (h, c) = self.lstm(embeds_pack)
        lstm_feature, _ = pad_packed_sequence(out_packed, batch_first=True)
        # lstm_feature: ([batch_size, max_word_length, HP_hidden_dim])
        lstm_feature = self.droplstm(lstm_feature)
        lstm_feature = self.hidden2tag(lstm_feature)
        # lstm_feature: ([batch_size, max_word_length, len(self.index2label)+2])
        return lstm_feature

    def neg_log_likelihood(self, batch_word, batch_knowledge, mask, knowledge_mask, batch_label, batch_wordlen, dynanmic_meta_embedding):
        lstm_feature = self._get_lstm_features(batch_word, batch_knowledge, knowledge_mask, batch_wordlen, dynanmic_meta_embedding)
        total_loss = self.crf.neg_log_likelihood_loss(lstm_feature, mask, batch_label)
        scores, tag_seq = self.crf._viterbi_decode(lstm_feature, mask)
        return total_loss, tag_seq

    def forward(self, batch_word, batch_knowledge, mask, knowledge_mask, batch_label, batch_wordlen, dynanmic_meta_embedding):
        lstm_feature = self._get_lstm_features(batch_word, batch_knowledge, knowledge_mask, batch_wordlen, dynanmic_meta_embedding)
        scores, best_path = self.crf._viterbi_decode(lstm_feature, mask)
        return best_path


