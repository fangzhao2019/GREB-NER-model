import numpy as np

def get_ner_fmeasure(fact_results, predict_results):
    #O B-BRAND I-BRAND B-CAR I-CAR
    
    results_count=np.zeros((2,3))
    for i in range(len(fact_results)):
        assert (len(fact_results[i]) == len(predict_results[i]))
        fact_brand_starts=[j for j in range(len(fact_results[i])) if fact_results[i][j]=='B-BRAND']
        fact_car_starts=[j for j in range(len(fact_results[i])) if fact_results[i][j]=='B-CAR']
        predict_brand_starts=[j for j in range(len(predict_results[i])) if predict_results[i][j]=='B-BRAND']
        predict_car_starts=[j for j in range(len(predict_results[i])) if predict_results[i][j]=='B-CAR']
        for start in fact_brand_starts:
            end=start+1
            for k in range(start+1, len(fact_results[i])):
                if fact_results[i][k]=='I-BRAND':
                    end+=1
                else:
                    break
            if fact_results[i][start:end]==predict_results[i][start:end]:
                results_count[0][0]+=1

        for start in fact_car_starts:
            end=start+1
            for k in range(start+1, len(fact_results[i])):
                if fact_results[i][k]=='I-CAR':
                    end+=1
                else:
                    break
            if fact_results[i][start:end]==predict_results[i][start:end]:
                results_count[1][0]+=1

        results_count[0][1]+=len(fact_brand_starts)
        results_count[0][2]+=len(predict_brand_starts)
        results_count[1][1]+=len(fact_car_starts)
        results_count[1][2]+=len(predict_car_starts)

    fmeasure={}
    labelSet=['brand','car']
    for idx in range(len(results_count)):
        metric={}
        precision=results_count[idx][0]/results_count[idx][2]
        recall=results_count[idx][0]/results_count[idx][1]
        f_score=2*precision*recall/float(recall+precision)
        metric['p']=precision
        metric['r']=recall
        metric['f']=f_score
        fmeasure[labelSet[idx]]=metric
    
    metric={}
    precision=(results_count[0][0]+results_count[1][0])/(results_count[0][2]+results_count[1][2])
    recall=(results_count[0][0]+results_count[1][0])/(results_count[0][1]+results_count[1][1])
    f_score=2*precision*recall/float(recall+precision)
    metric['p']=precision
    metric['r']=recall
    metric['f']=f_score
    fmeasure['overall']=metric
    return fmeasure,f_score
                
    
                
                 
