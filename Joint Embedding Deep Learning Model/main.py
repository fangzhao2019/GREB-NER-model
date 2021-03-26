import argparse
from utils.batchify_with_label import batchify_with_label
from utils.metric import get_ner_fmeasure
import time
import sys
import torch.optim as optim
from model.Joint_Bert_BiLSTM_CRF import Joint_Bert_BiLSTM_CRF
from utils import dataLoading
from utils import knowledgeLoading
import json
import random
import logging
import numpy as np
import torch
import os
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
import gc
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def showExample(example):
    logger.info("\n*** Example ***")
    logger.info("unique_id: %s" % (example.unique_id))
    logger.info("tokens: %s" % " ".join([str(x) for x in example.tokens]))
    logger.info("input_ids: %s" % " ".join([str(x) for x in example.input_ids]))
    logger.info("input_mask: %s" % " ".join([str(x) for x in example.input_mask]))
    logger.info(
        "input_type_ids: %s" % " ".join([str(x) for x in example.input_type_ids]))
    logger.info(
        "input_labels: %s" % " ".join([str(x) for x in example.input_labels]))

def getKnowledgeEmbedding(features,entity2vecDict):
    knowledgeVectors={}
    for unique_id in features.keys():
        feature=features[unique_id]
        sentenceVector=[]
        for j in range(len(feature)):
            characterVector=[]
            candidateEntity=[]
            weights=[]
            for k in range(len(feature[j])):
                character=feature[j][k]
                candidateEntity.append(entity2vecDict[character[0]])
                weights.append(float(character[1]))
            candidateEntity=np.array(candidateEntity)
            weights=np.array(weights)
            weights=weights/sum(weights)
            characterVector=[candidateEntity[m]*weights[m] for m in range(len(weights))]
            sentenceVector.append(characterVector)
        knowledgeVectors[unique_id]=sentenceVector
    return knowledgeVectors

def getBertEmbedding(features,bertModel):
    all_input_unique_id=torch.tensor([f.unique_id for f in features],dtype=torch.long)
    all_input_ids=torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_input_labels = torch.Tensor([f.input_labels for f in features])
    eval_data = TensorDataset(all_input_ids, all_input_mask)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=bert_batch_size)
    bertModel.eval()
    i=0
    for input_ids, input_mask in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        all_encoder_layers, _ = bertModel(input_ids, token_type_ids=None, attention_mask=input_mask,
                                      output_all_encoded_layers=False)
        all_encoder_layers = all_encoder_layers.cpu().detach().numpy()
        if i==0:
            vectorSet=all_encoder_layers
        else:
            vectorSet=np.concatenate([vectorSet, all_encoder_layers],axis=0)
        #all_encoder_layers=torch.rand(bert_batch_size,512,768)
        i+=1
        if i%100==0:
            print('已经处理bert词向量%d条'%min(i*bert_batch_size,len(features)))
    return all_input_unique_id, all_input_ids,all_input_mask, torch.tensor(vectorSet), all_input_labels

#更新学习率
def lr_decay(optimizer, epoch, decay_rate, init_lr):
    # 用于衰减学习率
    lr = init_lr * ((1-decay_rate)**epoch)
    print(" Learning rate is setted as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def predict_check(pred_variable, gold_variable, mask_variable):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)
    right_token = np.sum(overlaped * mask)
    total_token = mask.sum()
    # print("right: %s, total: %s"%(right_token, total_token))
    return right_token, total_token

def recover_label(pred_variable, gold_variable, mask_variable, word_recover, labelRindex):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """

    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = gold_variable.size(0)
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [labelRindex[pred_tag[idx][idy]] for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [labelRindex[gold_tag[idx][idy]] for idy in range(seq_len) if mask[idx][idy] != 0]
        assert (len(pred) == len(gold))
        pred_label.append(pred)
        gold_label.append(gold)
    return pred_label, gold_label

def evaluate(dataUniqueId,dataIds,dataMask,dataVectors,dataLabels,dataKnowledgeVector, model, padding_label,labelRindex):
    ## 评价函数
    pred_results = []
    gold_results = []
    ## set model in eval model
    model.eval()
    batch_size = 1
    start_time = time.time()

    eval_data=TensorDataset(dataUniqueId,dataIds,dataMask,dataVectors,dataLabels)
    eval_sampler=SequentialSampler(eval_data)
    eval_dataloader=DataLoader(eval_data,sampler=eval_sampler,batch_size=batch_size)
    for uniqueIds,input_batch_list,mask,instanceVectors,input_batch_label in eval_dataloader:
        knowledgeExamples=[dataKnowledgeVector[int(idx)] for idx in uniqueIds]
        max_entity_num=max([len(xxx) for yyy in knowledgeExamples for xxx in yyy])
        batch_word, batch_knowledge, word_seq_tensor, batch_wordlen, batch_wordrecover, batch_label, mask, knowledge_mask=batchify_with_label(instanceVectors, input_batch_list, input_batch_label, knowledgeExamples, mask, GPU, padding_label, max_entity_num)
        tag_seq = model.forward(batch_word, batch_knowledge, mask, knowledge_mask, batch_label, batch_wordlen, dynanmic_meta_embedding)
        pred_label, gold_label = recover_label(tag_seq, batch_label, mask, batch_wordrecover,labelRindex)
        pred_results += pred_label
        gold_results += gold_label

    decode_time = time.time() - start_time
    speed=len(dataIds)/decode_time
    fmeasure, acc= get_ner_fmeasure(gold_results, pred_results)
    return speed, fmeasure, acc

def drawFigure(x,y1,y2,y3,filename):
    plt.figure()
    plt.plot(x,y1)
    plt.plot(x,y2)
    plt.plot(x,y3)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('the accuracy of iteration')
    plt.savefig(filename)
    plt.show()

def saveToTxt(x,train,dev,test,filename):
    f=open(filename,'w',encoding='utf-8')
    for i in range(len(x)):
        f.write('%.4f %.4f %.4f %.4f'%(x[i],train[i],dev[i],test[i]))
        f.write('\n')
    f.close()

if __name__ == '__main__':
    time1=time.time()
    seed_num = 100
    random.seed(seed_num)
    torch.manual_seed(seed_num)
    np.random.seed(seed_num)
    torch.cuda.manual_seed(seed_num)

    dynanmic_meta_embedding=False
    bert_model='bert-base-chinese'
    do_lower_case=True
    max_seq_length = 512
    bert_embedding=768
    HP_iteration=50
    bert_batch_size=5
    batch_size=20
    HP_lr=0.01
    HP_lr_decay = 0.05
    weight_decay = 0.00000005
    if torch.cuda.is_available():
        GPU = True
        device=torch.device('cuda')
    else:
        GPU = False
        device=torch.device('cpu')
    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)

    trainFeatures=dataLoading.generate_instance("data/trainData", max_seq_length, tokenizer)
    devFeatures=dataLoading.generate_instance("data/devData", max_seq_length, tokenizer)
    testFeatures=dataLoading.generate_instance("data/testData", max_seq_length, tokenizer)
    labelIndex=dataLoading.labelSummary(trainFeatures)
    trainFeatures = dataLoading.dataLabelIndexed(trainFeatures,labelIndex)
    devFeatures = dataLoading.dataLabelIndexed(devFeatures, labelIndex)
    testFeatures = dataLoading.dataLabelIndexed(testFeatures, labelIndex)
    print('************成功载入数据************')
    print('标签类型为：', labelIndex)
    #showExample(trainFeatures[0])
    labelRindex={}
    for k,v in labelIndex.items():
        labelRindex[v]=k

    trainKnowledge=knowledgeLoading.read_rawKnowledge('data/trainDataKnowledge')
    devKnowledge=knowledgeLoading.read_rawKnowledge('data/devDataKnowledge')
    testKnowledge=knowledgeLoading.read_rawKnowledge('data/testDataKnowledge')

    entity2vecDict=json.load(open('data/entity2vecDict.json'))
    for k in entity2vecDict.keys():
        entity2vecDict[k]=np.array([float(row) for row in entity2vecDict[k].split(',')])

    trainKnowledgeVector=getKnowledgeEmbedding(trainKnowledge,entity2vecDict)
    devKnowledgeVector=getKnowledgeEmbedding(devKnowledge,entity2vecDict)
    testKnowledgeVector=getKnowledgeEmbedding(testKnowledge,entity2vecDict)
    print('\n************成功载入知识************')

    bertModel = BertModel.from_pretrained(bert_model)
    bertModel.to(device)
    trainUniqueId, trainIds, trainMask, trainVectors,trainLabels=getBertEmbedding(trainFeatures,bertModel)
    print("trainData done")
    devUniqueId, devIds, devMask, devVectors,devLabels = getBertEmbedding(devFeatures, bertModel)
    print("devData done")
    testUniqueId, testIds, testMask, testVectors,testLabels = getBertEmbedding(testFeatures, bertModel)
    print("testData done")
    time2=time.time()
    print('载入bert向量耗时%d秒'%(time2-time1))

    #训练模型
    model = Joint_Bert_BiLSTM_CRF(labelIndex,GPU)
    print("************成功载入模型************")
    print("打印模型可优化的参数名称")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    print("opimizer is Adam")
    optimizer = optim.Adam(parameters, lr=HP_lr, weight_decay=weight_decay)

    # print("opimizer is SGD")
    # optimizer = optim.SGD(parameters, lr=data.HP_lr, weight_decay=data.weight_decay)
    best_dev = -1
    best_test = -1
    padding_label=labelIndex['O']
    epoch_id=[]
    loss_list=[]
    train_acc=[]
    dev_acc=[]
    test_acc=[]


    for idx in range(HP_iteration):
        epoch_start=time.time()
        print("Epoch: %s/%s" % (idx, HP_iteration))
        optimizer = lr_decay(optimizer, idx, HP_lr_decay, HP_lr)
        total_loss=0
        right_token=0
        whole_token=0
        
        model.train()
        model.zero_grad()
        
        train_data=TensorDataset(trainUniqueId,trainIds,trainMask,trainVectors,trainLabels)
        train_sampler=SequentialSampler(train_data)
        train_dataloader=DataLoader(train_data,sampler=train_sampler,batch_size=batch_size)
        for uniqueIds,input_batch_list,mask,instanceVectors,input_batch_label in train_dataloader:
            knowledgeExamples=[trainKnowledgeVector[int(idx)] for idx in uniqueIds]
            max_entity_num=max([len(xxx) for yyy in knowledgeExamples for xxx in yyy])
            batch_word, batch_knowledge, word_seq_tensor, batch_wordlen, word_seq_recover, batch_label, mask, knowledge_mask=batchify_with_label(instanceVectors, input_batch_list, input_batch_label, knowledgeExamples, mask, GPU, padding_label, max_entity_num)
            
            loss, tag_seq = model.neg_log_likelihood(batch_word, batch_knowledge, mask, knowledge_mask, batch_label, batch_wordlen, dynanmic_meta_embedding)

            right, whole = predict_check(tag_seq, batch_label, mask)
            right_token += right
            whole_token += whole
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            model.zero_grad()
        acc=(right_token + 0.) / whole_token
        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print("Epoch: %s training finished. Time: %.2fs,  total loss: %s, acc: %s/%s=%.4f"%(
            idx, epoch_cost, total_loss, right_token, whole_token, acc))
        epoch_id.append(idx)
        loss_list.append(total_loss+0./len(trainIds))
        train_acc.append(acc)

        #在dev集上评价
        speed1, fmeasure1, acc1=evaluate(devUniqueId,devIds,devMask,devVectors,devLabels,devKnowledgeVector, model, padding_label,labelRindex)
        dev_finish=time.time()
        dev_cost=dev_finish-epoch_finish
        current_score_dev = acc1
        print("Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f" % (dev_cost, speed1, current_score_dev))
        for k in fmeasure1.keys():
            print('label %s    p: %.4f, r: %.4f, f: %.4f'%(k, fmeasure1[k]['p'], fmeasure1[k]['r'], fmeasure1[k]['f']))
        dev_acc.append(acc1)
        

        # 在test集上评价
        speed2, fmeasure2, acc2 = evaluate(testUniqueId,testIds,testMask,testVectors,testLabels,testKnowledgeVector, model, padding_label,labelRindex)
        test_finish = time.time()
        test_cost = test_finish - dev_finish
        print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f" % (test_cost, speed2, acc2))
        for k in fmeasure2.keys():
            print('label %s    p: %.4f, r: %.4f, f: %.4f'%(k, fmeasure2[k]['p'], fmeasure2[k]['r'], fmeasure2[k]['f']))
        test_acc.append(acc2)

        #输出最优模型
        if current_score_dev>best_dev:
            print("Exceed previous best acc score:", best_dev)
            #model_name = save_model_dir + '_' + str(idx) + ".model"
            #torch.save(model.state_dict(), model_name)
            #print(model_name)
            best_dev = current_score_dev
        print('\n\n')
        gc.collect()  # 删除输出，清理内存
    time3=time.time()
    print('共耗时%d秒'%(time3-time1))

    drawFigure(epoch_id,train_acc, dev_acc, test_acc,'/home/amax/old_lab/amax/Documents/robot/Lee/zhao/Bert-BiLSTM-CRF/result/accuracy.jpg')
    saveToTxt(epoch_id,train_acc, dev_acc, test_acc,'/home/amax/old_lab/amax/Documents/robot/Lee/zhao/Bert-BiLSTM-CRF/result/accuracy.txt')

    drawFigure(epoch_id,loss_list,loss_list,loss_list,'/home/amax/old_lab/amax/Documents/robot/Lee/zhao/Bert-BiLSTM-CRF/result/loss.jpg')
    saveToTxt(epoch_id,loss_list,loss_list,loss_list,'/home/amax/old_lab/amax/Documents/robot/Lee/zhao/Bert-BiLSTM-CRF/result/loss.txt')

        