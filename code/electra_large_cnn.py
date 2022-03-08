import os
import math
import random
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from transformers import AdamW
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import AutoConfig
from transformers import get_cosine_schedule_with_warmup

from sklearn.model_selection import KFold
from tqdm.notebook import tqdm

import gc
gc.enable()

NUM_FOLDS = 5
NUM_EPOCHS = 3
BATCH_SIZE = 24
MAX_LEN = 256
EVAL_SCHEDULE = [(0.50, 2048), (0.4, 2048), (0.3, 2048), (0.2, 2048), (-1., 2048)]
TOKENIZER_PATH = ROBERTA_PATH = "google/electra-large-discriminator"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    torch.backends.cudnn.deterministic = True

train_df = pd.read_csv("train.csv")

train_df

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

class LitDataset(Dataset):
    def __init__(self, df, inference_only=False):
        super().__init__()

        self.df = df        
        self.inference_only = inference_only
        self.text = df.text.tolist()
        #self.text = [text.replace("\n", " ") for text in self.text]
        
        if not self.inference_only:
            self.target = torch.tensor(df.y.values, dtype=torch.float32)        
    
        self.encoded = tokenizer.batch_encode_plus(
            self.text,
            padding = 'max_length',            
            max_length = MAX_LEN,
            truncation = True,
            return_attention_mask=True
        )        
 

    def __len__(self):
        return len(self.df)

    
    def __getitem__(self, index):        
        input_ids = torch.tensor(self.encoded['input_ids'][index])
        attention_mask = torch.tensor(self.encoded['attention_mask'][index])
        
        if self.inference_only:
            return (input_ids, attention_mask)            
        else:
            target = self.target[index]
            return (input_ids, attention_mask, target)

class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()

        config = AutoConfig.from_pretrained(ROBERTA_PATH)
        config.update({"output_hidden_states":True, 
                       "hidden_dropout_prob": 0.0,
                       "layer_norm_eps": 1e-7})                       
        
        self.roberta = AutoModel.from_pretrained(ROBERTA_PATH, config=config)  
        #self.roberta.base_model.embeddings.requires_grad_(False)    
        self.attention = nn.Sequential(            
            nn.Linear(config.hidden_size, 512),            
            nn.Tanh(),                       
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )        
        self.fc = nn.Linear(config.hidden_size, 1)                        
        
    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(input_ids=input_ids,
                                      attention_mask=attention_mask)        
        last_layer_hidden_states = roberta_output.hidden_states[-1]
        weights = self.attention(last_layer_hidden_states)

        context_vector = torch.sum(weights * last_layer_hidden_states, dim=1)       
        return self.fc(context_vector)

class DenseModel(nn.Module):
    def __init__(self):
        super().__init__()

        config = AutoConfig.from_pretrained(ROBERTA_PATH)
        config.update({"output_hidden_states":True, 
                       "hidden_dropout_prob": 0.0,
                       "layer_norm_eps": 1e-6,
                      'return_dict':False})                       
        
        self.roberta = AutoModel.from_pretrained(ROBERTA_PATH, config=config)  
            
        #self.attention = nn.Sequential(            
        #    nn.Linear(768, 512),            
        #    nn.CELU(),                       
        #    nn.Linear(512, 1),
        #    nn.Softmax(dim=1)
        #)        

        self.fc = nn.Linear(768, 1)     
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        os, ms, hs = self.roberta(input_ids=input_ids,
                                      attention_mask=attention_mask)        

        x = torch.stack([hs[-1], hs[-2], os])
        x = torch.mean(x, (0, 2))
        x = self.fc(x)
        return x

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        config = AutoConfig.from_pretrained(ROBERTA_PATH)
        config.update({"output_hidden_states":True, 
                "hidden_dropout_prob": 0.0,
                'return_dict':True})                      
        
        self.roberta = AutoModel.from_pretrained(ROBERTA_PATH, config=config)
            
        self.conv1 = nn.Conv1d(config.hidden_size, 512, 3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv2 = nn.Conv1d(512, 1, 3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros')


    def forward(self, input_ids, attention_mask):
        output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)        
        hs = output.hidden_states
        #x = hs[-2]
        x = torch.stack(hs)
        x = torch.mean(x, 0)
        conv1_logits = self.conv1(x.transpose(1, 2))
        conv2_logits = self.conv2(conv1_logits)
        logits = conv2_logits.transpose(1, 2)
        x = torch.mean(logits, 1)
        return x

def eval_mse(model, data_loader):
    """Evaluates the mean squared error of the |model| on |data_loader|"""
    model.eval()            
    mse_sum = 0

    with torch.no_grad():
        for batch_num, (input_ids, attention_mask, target) in enumerate(data_loader):
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)                        
            target = target.to(DEVICE)           
            
            pred = model(input_ids, attention_mask)                       

            mse_sum += nn.MSELoss(reduction="sum")(pred.flatten(), target).item()
                

    return mse_sum / len(data_loader.dataset)

def train(model, model_path, train_loader, val_loader,
          optimizer, scheduler=None, num_epochs=NUM_EPOCHS):    
    best_val_rmse = None
    best_epoch = 0
    step = 0
    last_eval_step = 0
    eval_period = EVAL_SCHEDULE[0][1]    
    scaler = GradScaler()

    start = time.time()

    for epoch in range(num_epochs):                           
        val_rmse = None         

        for batch_num, (input_ids, attention_mask, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)            
            target = target.to(DEVICE)                        
            with autocast():
                optimizer.zero_grad()
                
                model.train()

                pred = model(input_ids, attention_mask)
                                                            
                mse = nn.MSELoss(reduction="mean")(pred.flatten(), target)
                            
                #mse.backward()
                scaler.scale(mse).backward()
                scaler.step(optimizer)
                scaler.update()
                #optimizer.step()
                if scheduler:
                    scheduler.step()
            
            if step >= last_eval_step + eval_period:
                # Evaluate the model on val_loader.
                elapsed_seconds = time.time() - start
                num_steps = step - last_eval_step
                print(f"\n{num_steps} steps took {elapsed_seconds:0.3} seconds")
                last_eval_step = step
                
                val_rmse = math.sqrt(eval_mse(model, val_loader))                            

                print(f"Epoch: {epoch} batch_num: {batch_num}", 
                      f"val_rmse: {val_rmse:0.4}")

                for rmse, period in EVAL_SCHEDULE:
                    if val_rmse >= rmse:
                        eval_period = period
                        break                               
                
                if not best_val_rmse or val_rmse < best_val_rmse:                    
                    best_val_rmse = val_rmse
                    best_epoch = epoch
                    torch.save(model.state_dict(), model_path)
                    print(f"New best_val_rmse: {best_val_rmse:0.4}")
                else:       
                    print(f"Still best_val_rmse: {best_val_rmse:0.4}",
                          f"(from epoch {best_epoch})")                                    
                    
                start = time.time()
                                            
            step += 1
                        
    
    return best_val_rmse

def create_optimizer(model):
    named_parameters = list(model.named_parameters())    
    
    roberta_parameters = named_parameters[:389]    
    attention_parameters = named_parameters[391:395]
    regressor_parameters = named_parameters[395:]
        
    attention_group = [params for (name, params) in attention_parameters]
    regressor_group = [params for (name, params) in regressor_parameters]

    parameters = []
    parameters.append({"params": attention_group})
    parameters.append({"params": regressor_group})

    for layer_num, (name, params) in enumerate(roberta_parameters):
        weight_decay = 0.0 if "bias" in name else 0.01

        lr = 1e-5

        if layer_num >= 129:        
            lr = 2e-5

        if layer_num >= 250:
            lr = 3e-5

        parameters.append({"params": params,
                           "weight_decay": weight_decay,
                           "lr": lr})

    return AdamW(parameters)

def train_folds(name='', model_name=None):
    torch.cuda.empty_cache()
    gc.collect()
    SEED = 1024
    list_val_rmse = []

    kfold = KFold(n_splits=NUM_FOLDS, random_state=SEED, shuffle=True)

    for fold, (train_indices, val_indices) in enumerate(kfold.split(train_df)):    
        print(f"\nFold {fold + 1}/{NUM_FOLDS}")
        model_path = f"{name}_{fold + 1}.pth"
            
        set_random_seed(SEED + fold)
        
        train_dataset = LitDataset(train_df.loc[train_indices])    
        val_dataset = LitDataset(train_df.loc[val_indices])    
            
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True, num_workers=16)    
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=False, shuffle=False, num_workers=16)    
               
        
        model = model_name().to(DEVICE)
        
        optimizer = create_optimizer(model)                        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_training_steps=NUM_EPOCHS * len(train_loader),
            num_warmup_steps=80)    
        
        list_val_rmse.append(train(model, model_path, train_loader, val_loader, optimizer, scheduler=scheduler))

        del model
        gc.collect()
        
        print("\nPerformance estimates:")
        print(list_val_rmse)
        print("Mean:", np.array(list_val_rmse).mean())
    return list_val_rmse

train_folds(name='electra_large_cnn', model_name=CNNModel)
