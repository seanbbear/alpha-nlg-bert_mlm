import jsonlines
from transformers import AutoTokenizer
import numpy as np
import torch
from torch.utils.data import TensorDataset

def get_dataset(path):
    # obs1 + obs2 + hyp1 + hyp2 在 train 的最大長度為448 在 dev 的為329

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") 

    with jsonlines.open(path) as f:
        data_len = 0
        for obj in f:
            hyp = obj['hyp_plus'] + '[SEP]'
            data_len += len(tokenizer.tokenize(hyp))
        print(data_len)

    with jsonlines.open(path) as f:

        input_ids = np.zeros(shape=(data_len,512))     
        token_type_ids = np.zeros(shape=(data_len,512))     
        attention_mask = np.zeros(shape=(data_len,512))  
        answer = np.zeros(shape=(data_len,512))  
        
        index = 0
        for obj in f:

            obs1 = '[CLS]' + obj['obs1'] + '[SEP]'
            obs2 = obj['obs2'] + '[SEP]'
            hyp = obj['hyp_plus'] + '[SEP]'

            obs1_word_piece = tokenizer.tokenize(obs1)
            obs2_word_piece = tokenizer.tokenize(obs2)
            hyp_word_piece = tokenizer.tokenize(hyp)

            tensor_list = input_feature(tokenizer, obs1_word_piece, obs2_word_piece, hyp_word_piece)
            for tensor in tensor_list:
                # print(tensor['input_ids'])
                input_ids[index] = tensor['input_ids']
                token_type_ids[index] = tensor['token_type_ids']
                attention_mask[index] = tensor['attention_mask']     
                answer[index] = tensor['masked_lm_labels']
                # print(index)
                index += 1
            # break

        input_ids = torch.LongTensor(input_ids)
        token_type_ids = torch.LongTensor(token_type_ids)
        attention_mask = torch.LongTensor(attention_mask)     
        answer = torch.LongTensor(answer)
        print(input_ids.shape, token_type_ids.shape, attention_mask.shape, answer.shape)
    return TensorDataset(input_ids, token_type_ids, attention_mask, answer)

def input_feature(tokenizer, obs1, obs2, hyp):
    tensor_list = []

    max_length = 512
    obs1_ids = tokenizer.convert_tokens_to_ids(obs1)
    obs2_ids = tokenizer.convert_tokens_to_ids(obs2)
    hyp_ids = tokenizer.convert_tokens_to_ids(hyp)

    input_ids  = []
    token_type_ids  = []
    attention_mask = []
    masked_lm_labels = []

    # 先append第一句 token_type為0
    for i in range(len(obs1_ids)):
        input_ids.append(obs1_ids[i])
        token_type_ids.append(0)
        attention_mask.append(1)
        masked_lm_labels.append(-100)

    # append第二句 token_type為1
    for i in range(len(obs2_ids)):
        input_ids.append(obs2_ids[i])
        token_type_ids.append(1)
        attention_mask.append(1)
        masked_lm_labels.append(-100)

    # append answer
    # [MASK]的id為103
    for i in range(len(hyp_ids)):
        input_ids.append(hyp_ids[i])
        token_type_ids.append(0)
        attention_mask.append(1)
        masked_lm_labels.append(-100)

        tensor_list.append(full_tensor(input_ids, token_type_ids, attention_mask, masked_lm_labels, max_length))

        
    return tensor_list

def full_tensor(input_ids, token_type_ids, attention_mask, masked_lm_labels, max_length):
    new_input_ids = input_ids[:]
    new_token_type_ids = token_type_ids[:]
    new_attention_mask = attention_mask[:]
    new_masked_lm_labels = masked_lm_labels[:]

    temp = new_input_ids[-1]
    new_input_ids[-1] = 103 # 把最後一個input_ids改成[MASK]
    new_masked_lm_labels[-1] = temp # 把最後一個[MASK]改成要預測的id
    
    while len(new_input_ids) < max_length:
        new_input_ids.append(0)
        new_token_type_ids.append(0)
        new_attention_mask.append(0)
        new_masked_lm_labels.append(-100)

    
    assert len(new_input_ids) == len(new_token_type_ids) == len(new_attention_mask) == len(new_masked_lm_labels) == 512
    
    new_features = {'input_ids':new_input_ids,
                    'token_type_ids':new_token_type_ids,
                    'attention_mask':new_attention_mask,
                    'masked_lm_labels':new_masked_lm_labels}
    return new_features
     


if __name__=="__main__":
    # get_dataset('../../data/unique/train-unique_obs1_obs2.jsonl')
    get_dataset('../../data/unique/short.jsonl')