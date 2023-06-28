import sys 
sys.path.append("./")

import torch
from src.transform import SelectionSequentialTransform, SelectionJoinTransform, SelectionConcatTransform
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 

class Callcenter(): 
    def __init__(self, poly_encoder, cross_encoder, tokenizer, emb_df, device, topk=5): 
        self.context_transform = SelectionJoinTransform(tokenizer=tokenizer, max_len=128)
        self.response_transform = SelectionSequentialTransform(tokenizer=tokenizer, max_len=128)
        self.concat_transform = SelectionConcatTransform(tokenizer=tokenizer, max_len=128+128)
        self.poly_encoder = poly_encoder
        self.cross_encoder = cross_encoder
        self.emb_df = emb_df
        self.device = device
        self.topk=topk

    def cosine_score(self, query, idx): 
        with torch.no_grad():
            response =[query]
            responses_token_ids_list, responses_input_masks_list = self.response_transform(response)  # [token_ids],[seg_ids],[masks]
            long_tensors = [responses_token_ids_list, responses_input_masks_list]
            responses_token_ids_list, responses_input_masks_list = (torch.tensor(t, dtype=torch.long, device=self.device) for t in long_tensors)

            ids = responses_token_ids_list.unsqueeze(1)
            masks = responses_input_masks_list.unsqueeze(1)
            seq_length = ids.shape[-1]

            ids = ids.view(-1, seq_length)
            masks = masks.view(-1, seq_length)

            question_emb = self.poly_encoder.bert(ids, masks)[0][:, 0, :].to('cpu').detach().numpy()
            compare_question_embs = torch.stack(list(self.emb_df.iloc[idx][['q1', 'q2', 'q3', 'q4']].values), dim=1)
            compare_question_embs = compare_question_embs.to('cpu').detach().numpy()[0]
            # 코사인 유사도 계산
            similarity = cosine_similarity(question_emb, compare_question_embs)[0]
            return np.max(similarity)
    def cross_score(self, query, indices): 
        with torch.no_grad():
            query = [query]
            responses = [self.emb_df['text'].iloc[idx] for idx in indices]
            ret_input_ids, ret_input_masks, ret_segment_ids = self.concat_transform(query, responses)
            ret_input_ids = torch.tensor(ret_input_ids).unsqueeze(0).to(self.device)
            ret_input_masks = torch.tensor(ret_input_masks).unsqueeze(0).to(self.device)
            ret_segment_ids = torch.tensor(ret_segment_ids).unsqueeze(0).to(self.device)
            c_score = self.cross_encoder(ret_input_ids, ret_input_masks, ret_segment_ids)
            return c_score.to('cpu').detach().numpy()

    def inference(self, query): 
        def context_input(context):
            context_input_ids, context_input_masks = self.context_transform(context)
            contexts_token_ids_list_batch, contexts_input_masks_list_batch = [context_input_ids], [context_input_masks]
            long_tensors = [contexts_token_ids_list_batch, contexts_input_masks_list_batch]
            contexts_token_ids_list_batch, contexts_input_masks_list_batch = (torch.tensor(t, dtype=torch.long, device=self.device) for t in long_tensors)
            return contexts_token_ids_list_batch, contexts_input_masks_list_batch
        
        def embs_gen(contexts_token_ids_list_batch, contexts_input_masks_list_batch):
            ctx_out = self.poly_encoder.bert(contexts_token_ids_list_batch, contexts_input_masks_list_batch)[0]  # [bs, length, dim]
            poly_code_ids = torch.arange(self.poly_encoder.poly_m, dtype=torch.long).to(self.device)
            poly_code_ids = poly_code_ids.unsqueeze(0).expand(1, self.poly_encoder.poly_m)
            poly_codes = self.poly_encoder.poly_code_embeddings(poly_code_ids) # [bs, poly_m, dim]
            embs = self.poly_encoder.dot_attention(poly_codes, ctx_out, ctx_out) # [bs, poly_m, dim]
            return embs
        
        def score(embs, cand_emb):
            ctx_emb = self.poly_encoder.dot_attention(cand_emb, embs, embs) # [bs, res_cnt, dim]
            dot_product = (ctx_emb*cand_emb).sum(-1)
            return dot_product

        with torch.no_grad(): 
            cand_embs = torch.stack(self.emb_df['embedding'].tolist(), dim=1).to(self.device)
            embs = embs_gen(*context_input([query]))
            embs = embs.to(self.device)
            s = score(embs, cand_embs)
            top_indices = torch.topk(s, k=self.topk)[1].detach()[0].cpu().numpy()
            top_cross_score = self.cross_score(query, top_indices)
            return top_cross_score[0], top_indices