import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def batchify_with_label(instanceVectors, input_batch_list, input_batch_label, knowledgeExamples, mask, gpu, padding_label, max_entity_num):
    batch_size = len(input_batch_list)
    words=[]
    labels=[]
    for i in range(batch_size):
        words.append([input_batch_list[i][j] for j in range(len(mask[i])) if mask[i][j]==1])
        labels.append([input_batch_label[i][j] for j in range(len(mask[i])) if mask[i][j]==1])
    word_seq_lengths = list(map(len, words))  # 得到batch中每个句子的长度
    max_seq_len = max(word_seq_lengths)  # batch中最长句子的长度

    word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=False).long()
    label_seq_tensor = torch.ones((batch_size, max_seq_len), requires_grad=False).long()
    input_vectors = torch.ones((batch_size, max_seq_len,768), requires_grad=False).float()
    input_knowledge_vectors = torch.zeros((batch_size, max_seq_len, max_entity_num, 768), requires_grad=False).float()
    input_knowledge_mask=torch.zeros((batch_size, max_seq_len, max_entity_num), requires_grad=False).long()

    label_seq_tensor = padding_label * label_seq_tensor
    mask = torch.zeros((batch_size, max_seq_len), requires_grad=False).bool()  # 标记是否有数据，有为1，无为0
    for idx, (seq, label, seqlen, vector) in enumerate(zip(words, labels, word_seq_lengths, instanceVectors)):
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        mask[idx, :seqlen] = torch.Tensor([1] * seqlen)
        input_vectors[idx, :seqlen] = torch.FloatTensor(vector[:seqlen])
    
    for i in range(len(knowledgeExamples)):
        for j in range(len(knowledgeExamples[i])):
            entity_num=len(knowledgeExamples[i][j])
            input_knowledge_mask[i,j,:entity_num]=torch.Tensor([1]*entity_num)
            for k in range(len(knowledgeExamples[i][j])):
                input_knowledge_vectors[i][j][k]=torch.FloatTensor(knowledgeExamples[i][j][k])

    word_seq_lengths = torch.LongTensor(word_seq_lengths)
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    input_vectors=input_vectors[word_perm_idx]
    input_knowledge_vectors=input_knowledge_vectors[word_perm_idx]
    # 在原始序列中每个句子的长度排名
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    word_seq_recover=word_seq_recover.long()
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        mask = mask.cuda()
        input_vectors=input_vectors.cuda()
        input_knowledge_vectors=input_knowledge_vectors.cuda()
        input_knowledge_mask=input_knowledge_mask.cuda()
    return input_vectors, input_knowledge_vectors, word_seq_tensor, word_seq_lengths, word_seq_recover, label_seq_tensor, mask, input_knowledge_mask