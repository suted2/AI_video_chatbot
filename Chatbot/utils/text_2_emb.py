import sys 
sys.path.append("./")

from transformers import BertModel, BertConfig,  BertTokenizerFast
from transformers import XLMRobertaTokenizerFast, RobertaModel, RobertaConfig

from src.encoder import PolyEncoder 

import torch
from torch.utils.data import DataLoader, Dataset
import argparse
import pandas as pd 
import os 
import pickle
from tqdm import tqdm

class OneSentenceDataset(Dataset):
    def __init__(self, file_path,  response_transform, mode='poly'):
        self.response_transform = response_transform
        self.data_source = []
        self.mode = mode
        with open(file_path, encoding='utf-8') as f:
            for line in f.readlines():
                self.data_source.append(line.strip())

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, index):
        response = self.data_source[index]
        responses_token_ids_list, responses_input_masks_list = self.response_transform(response)  # [token_ids],[seg_ids],[masks]
        long_tensors = [responses_token_ids_list, responses_input_masks_list]
        responses_token_ids_list, responses_input_masks_list = (torch.tensor(t, dtype=torch.long, device=device) for t in long_tensors)
        return responses_token_ids_list, responses_input_masks_list

class SelectionSequentialTransform(object):
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, text):
        tokenized_dict = self.tokenizer.encode_plus(text, max_length=self.max_len, padding='max_length')
        input_ids, input_masks = tokenized_dict['input_ids'], tokenized_dict['attention_mask']
        assert len(input_ids) == self.max_len
        assert len(input_masks) == self.max_len

        return input_ids, input_masks


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default='ckpt/pretrained/bert-small-uncased', type=str)
    parser.add_argument("--text_path", default='path/to/origianl_text.txt', type=str)  # 한 줄에 '몇 번 버스를 타시면 됩니다.\n' 이렇게 한문장 저장
    parser.add_argument("--max_response_length", default=128, type=int)
    parser.add_argument("--output_dir", default='path/to/preprocessed_emb.df', type=str)
    parser.add_argument("--model_type", default='bert', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    print(args)

    MODEL_CLASSES = {
        'bert': (BertConfig, BertTokenizerFast, BertModel),
        'roberta' : (RobertaConfig, XLMRobertaTokenizerFast, RobertaModel)
    }
    ConfigClass, TokenizerClass, BertModelClass = MODEL_CLASSES[args.model_type]

    bert_config = ConfigClass.from_json_file(os.path.join(args.bert_model, 'config.json'))

    previous_model_file = os.path.join(args.bert_model, 'pytorch_model.bin')
    print('Loading parameters from', previous_model_file)

    model_state_dict = torch.load(previous_model_file, map_location="cpu")

    tokenizer = TokenizerClass.from_pretrained(args.bert_model, do_lower_case=True, clean_text=False)
    tokenizer.add_tokens(['\n'], special_tokens=True)
    response_transform = SelectionSequentialTransform(tokenizer=tokenizer, max_len=args.max_response_length)
    
    bert = BertModelClass(bert_config)
    bert.resize_token_embeddings(len(tokenizer))

    model = PolyEncoder(bert_config, bert=bert, poly_m=16)
    model.load_state_dict(model_state_dict)

    dataset = OneSentenceDataset(args.text_path, response_transform, mode='poly')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.resize_token_embeddings(len(tokenizer)) 
    model.to(device)
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        
        for ids, masks in tqdm(dataloader): 
            ids = ids.unsqueeze(0)
            masks = masks.unsqueeze(0)
            batch_size, res_cnt, seq_length = ids.shape
            ids = ids.view(-1, seq_length)
            masks = masks.view(-1, seq_length)
            cand_emb = model.bert(ids, masks)[0][:,0,:] # [bs, dim]
            embeddings.append(cand_emb.to('cpu'))

    emb_df = pd.DataFrame({'text':dataset.data_source, 'embedding' : embeddings})
    
    with open(args.output_dir, 'wb') as f: 
        pickle.dump(emb_df, f)