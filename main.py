from data_preprocess import get_dataset
from torch.utils.data import DataLoader
# from transformers import BertConfig, BertForMaskedLM, AdamW
from transformers import RobertaConfig, RobertaForMaskedLM, AdamW
import os
import torch

import wandb

if __name__=="__main__":
    # wandb專案名稱
    wandb.init(project="alpha-nlg")

    # setting device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    batch_size = 30
    lr = 5e-6
    training_epoch = 5
    
    # 
    # config = BertConfig.from_pretrained('bert-base-uncased')
    # model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    config = RobertaConfig.from_pretrained('roberta-base')
    # config.type_vocab_size = 2
    model = RobertaForMaskedLM.from_pretrained('roberta-base')

    # 多GPU
    if  torch.cuda.device_count()>1:         
        model = torch.nn.DataParallel(model,device_ids=[0,1])

    model.to(device)

    wandb.watch(model)

    # train_dataset = get_dataset('../../data/regular/short.jsonl')
    # dev_dataset = get_dataset('../../data/regular/short.jsonl')
    train_dataset = get_dataset('../../data/unique_augment/train.jsonl')
    dev_dataset = get_dataset('../../data/regular/dev.jsonl')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=True)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.1},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.1}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)

    model.zero_grad()
    for epoch in range(training_epoch):
        train_loss = 0.0
        train_acc = 0.0
        model.train()
        for batch_index, batch_dict in enumerate(train_dataloader):
            batch_dict = tuple(t.to(device) for t in batch_dict)
            # roberta的 type_vocab_size=1 所以token_type_ids只能給0
            outputs = model(
                input_ids = batch_dict[0],
                # token_type_ids = batch_dict[1],
                attention_mask = batch_dict[2],
                labels=batch_dict[3]
            )
            loss, logits = outputs[:2]
            
            if (device=='cuda' and device,torch.cuda.device_count()>1):
                loss = loss.mean()

            loss.backward()
            optimizer.step()
            model.zero_grad()

            # 計算loss
            loss_t = loss.item()
            train_loss += (loss_t - train_loss)/(batch_index + 1)

        # log
        print("epoch:%2d batch:%4d train_loss:%2.4f "%(epoch+1, batch_index+1, train_loss))
        wandb.log({"Train Loss":train_loss})

        test_loss = 0.0
        test_acc = 0.0
        model.eval()
        for batch_index, batch_dict in enumerate(dev_dataloader):
            batch_dict = tuple(t.to(device) for t in batch_dict)
            
            outputs = model(
                input_ids = batch_dict[0],
                # token_type_ids = batch_dict[1],
                attention_mask = batch_dict[2],
                labels=batch_dict[3]
            )
            loss,logits = outputs[:2]

            if (device=='cuda' and device,torch.cuda.device_count()>1):                 
                loss = loss.mean()
            
            
            # 計算loss
            loss_t = loss.item()
            test_loss += (loss_t - test_loss) / (batch_index + 1)

            # log
        print("epoch:%2d batch:%4d test_loss:%2.4f "%(epoch+1, batch_index+1, test_loss))
        wandb.log({"Test Loss":test_loss})
        
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))

        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained('bert_trained_model_regular/' + str(epoch))