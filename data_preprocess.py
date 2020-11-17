import jsonlines
from transformers import AutoTokenizer
import numpy as np
import torch
from torch.utils.data import TensorDataset

def get_dataset(path):
    # obs1 + obs2 + hyp1 + hyp2 在 train 的最大長度為448 在 dev 的為329
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") 
    tokenizer = AutoTokenizer.from_pretrained("roberta-base") 
    
    cls = tokenizer.cls_token
    sep = tokenizer.sep_token
    
    with jsonlines.open(path) as f:
        data_len = 0
        for obj in f:
            hyp = obj['hyp_plus'] + sep
            data_len += len(tokenizer.tokenize(hyp))
        print(data_len)

    with jsonlines.open(path) as f:

        input_ids = np.zeros(shape=(data_len,512))     
        token_type_ids = np.zeros(shape=(data_len,512))     
        attention_mask = np.zeros(shape=(data_len,512))  
        answer = np.zeros(shape=(data_len,512))  
        
        index = 0
        for obj in f:

            obs1 = cls + obj['obs1'] + sep
            obs2 = obj['obs2'] + sep
            hyp = obj['hyp_plus'] + sep


            obs1_word_piece = tokenizer.tokenize(obs1)
            obs2_word_piece = tokenizer.tokenize(obs2)
            hyp_word_piece = tokenizer.tokenize(hyp)

            tensor_list = input_feature(tokenizer, obs1_word_piece, obs2_word_piece, hyp_word_piece)
            # print(tensor_list[0])
            for tensor in tensor_list:
                input_ids[index] = tensor['input_ids']
                token_type_ids[index] = tensor['token_type_ids']
                attention_mask[index] = tensor['attention_mask']     
                answer[index] = tensor['masked_lm_labels']
                
                index += 1
        input_ids = torch.LongTensor(input_ids)
        token_type_ids = torch.LongTensor(token_type_ids)
        attention_mask = torch.LongTensor(attention_mask)     
        answer = torch.LongTensor(answer)
        print(input_ids.shape, token_type_ids.shape, attention_mask.shape, answer.shape)
    return TensorDataset(input_ids, token_type_ids, attention_mask, answer)

def input_feature(tokenizer, obs1, obs2, hyp):
    tensor_list = []

    obs1_tensor = sentence_to_ids(tokenizer, obs1, token_type=0)
    hyp_tensor = sentence_to_ids(tokenizer, hyp, token_type=1)
    obs2_tensor = sentence_to_ids(tokenizer, obs2, token_type=0)


    for i in range(1,len(hyp_tensor['input_ids'])+1):
        new_hyp_tensor = change_tensor(tokenizer, hyp_tensor, i)
        tensor_list.append(padding_tensor(tokenizer, obs1_tensor, new_hyp_tensor, obs2_tensor))     
    return tensor_list

def sentence_to_ids(tokenizer, sentence, token_type=0):
    ids = tokenizer.convert_tokens_to_ids(sentence)
    input_ids = ids[:] 
    token_type_ids = [token_type] * len(sentence)
    attention_mask = [1] * len(sentence)
    masked_lm_labels = [-100] * len(sentence)

    features = {'input_ids':input_ids,
                'token_type_ids':token_type_ids,
                'attention_mask':attention_mask,
                'masked_lm_labels':masked_lm_labels}
    return features

def change_tensor(tokenizer, ori_tensor, length):
    new_input_ids = ori_tensor['input_ids'][:length]
    new_token_type_ids = ori_tensor['token_type_ids'][:length]
    new_attention_mask = ori_tensor['attention_mask'][:length]
    new_masked_lm_labels = ori_tensor['masked_lm_labels'][:length]

    mask = tokenizer.mask_token_id

    temp = new_input_ids[-1]
    new_input_ids[-1] = mask # 把最後一個input_ids改成[MASK]
    new_masked_lm_labels[-1] = temp # 把最後一個[MASK]改成要預測的id
    
    
    
    new_features = {'input_ids':new_input_ids,
                    'token_type_ids':new_token_type_ids,
                    'attention_mask':new_attention_mask,
                    'masked_lm_labels':new_masked_lm_labels}
    return new_features
     
def padding_tensor(tokenizer, sentence_a, sentence_b, sentence_c, max_length=512):
    pad = tokenizer.pad_token_id
    input_ids = sentence_a['input_ids'] + sentence_b['input_ids'] + sentence_c['input_ids']
    token_type_ids = sentence_a['token_type_ids'] + sentence_b['token_type_ids'] + sentence_c['token_type_ids']
    attention_mask = sentence_a['attention_mask'] + sentence_b['attention_mask'] + sentence_c['attention_mask']
    masked_lm_labels = sentence_a['masked_lm_labels'] + sentence_b['masked_lm_labels'] + sentence_c['masked_lm_labels']

    while len(input_ids) < max_length:
        input_ids.append(pad)
        token_type_ids.append(pad)
        attention_mask.append(pad)
        masked_lm_labels.append(-100)

    
    assert len(input_ids) == len(token_type_ids) == len(attention_mask) == len(masked_lm_labels) == max_length

    padded_features = {'input_ids':input_ids,
                    'token_type_ids':token_type_ids,
                    'attention_mask':attention_mask,
                    'masked_lm_labels':masked_lm_labels}

    return padded_features


if __name__=="__main__":
    # get_dataset('../../data/unique/train-unique_obs1_obs2.jsonl')
    # get_dataset('../../data/regular/dev.jsonl')
    get_dataset('../../data/regular/short.jsonl')