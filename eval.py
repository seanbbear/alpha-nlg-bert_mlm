import jsonlines
import torch
from transformers import BertConfig, BertForMaskedLM, AutoTokenizer
import torch.nn.functional as F
from nlgeval import NLGEval
from tqdm import tqdm

def token_to_ids(sentence_input):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence_input))

def get_model_output(obs1, obs2):
    obs1 = "[CLS]" + obs1 + "[SEP]"
    obs2 = obs2 + "[SEP]"

    obs1_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obs1))
    obs2_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obs2))

    input_ids = []
    token_type_ids = []
    attention_mask = []

    # 先append第一句 token_type為0
    for i in range(len(obs1_ids)):
        input_ids.append(obs1_ids[i])
        token_type_ids.append(0)
        attention_mask.append(1)

    # append[MASK] token_type為1
    output_length = 0
    hyp = ""                 # 預測之ans
    maskpos = len(input_ids) # 欲預測位置

    input_ids.append(103)
    token_type_ids.append(1)
    attention_mask.append(1)
    
    # append第三句 token_type為0
    for i in range(len(obs2_ids)):
        input_ids.append(obs2_ids[i])
        token_type_ids.append(0)
        attention_mask.append(1)

    

    #補齊長度為512
    while len(input_ids)<512:
        input_ids.append(0)
        token_type_ids.append(0)
        attention_mask.append(0)

    #答案長度
    while output_length < 512:
        input_ids_tensor = torch.LongTensor([input_ids])
        token_type_ids_tensor = torch.LongTensor([token_type_ids])
        attention_mask_tensor = torch.LongTensor([attention_mask])

        output = model(
            input_ids = input_ids_tensor,
            token_type_ids = token_type_ids_tensor, 
            attention_mask = attention_mask_tensor
        )

        predicts = output[0]
        predicts_index = torch.argmax(predicts[0, maskpos]).item()
        predicts_token = tokenizer.convert_ids_to_tokens(predicts_index)

        if predicts_token == "[SEP]":
            break
        elif predicts_token.startswith("##"):
            hyp = hyp + predicts_token[2:]
        else:
            hyp = hyp + " " + predicts_token
        
        input_ids[maskpos] = predicts_index
        token_type_ids[maskpos] = 1
        attention_mask[maskpos] = 1
        maskpos += 1

        if maskpos < 512:
            input_ids[maskpos] = 103
        else:
            break

        output_length += 1
    
    return hyp

def score_eval(predict_list, target_list):
    nlgeval = NLGEval(no_skipthoughts=True, no_glove=True, metrics_to_omit=["METEOR"])
    result = nlgeval.compute_metrics(ref_list=[target_list],hyp_list=predict_list)
    return result

if __name__=="__main__":
     

    epoch = "0"
    path = "bert_trained_model_regular/" + epoch +"/"

    # model setting
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    config = BertConfig.from_pretrained( path + "config.json" )
    model = BertForMaskedLM.from_pretrained( path + "pytorch_model.bin", config=config)

    # setting device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if  torch.cuda.device_count()>1:         
        model = torch.nn.DataParallel(model,device_ids=[0,1])
    model.to(device)

    model.eval()

    obs1 = []
    obs2 = []
    predict = []
    target = []
    with jsonlines.open('../../data/regular/dev.jsonl') as f:        
        for obj in tqdm(f):
            hyp = get_model_output(obj['obs1'], obj['obs2'])
            predict.append(hyp)
            target.append(obj['hyp_plus'])
            obs1.append(obj['obs1'])
            obs2.append(obj['obs2'])

        score = score_eval(predict, target)

    with jsonlines.open(path + "result_log_len", mode='w') as writer:
        for i in range(len(obs1)):
            obj = {}
            obj['obs1'] = obs1[i]
            obj['obs2'] = obs2[i]
            obj['predict'] = predict[i]
            obj['target'] = target[i]
            writer.write(obj)
        writer.write({'score':score})
                

   
