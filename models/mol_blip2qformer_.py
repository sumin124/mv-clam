"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F


from lavis.models.blip_models.blip_outputs import BlipOutput
# from lavis.common.dist_utils import is_dist_avail_and_initialized
from model.mol_blip2_ import MolBlip2Base
#from model.dist_funs import pl_concat_all_gather
# from pytorch_lightning.utilities import distributed


class MolBlip2Qformer(MolBlip2Base):

    def __init__(
        self,
        gtm,
        lm,
        bert_name,
        temperature,
        gin_num_layers,
        gin_hidden_dim,
        gin_drop_ratio,
        tune_gnn=False,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        args=None,
    ):
        super().__init__()
        self.gtm = gtm
        self.lm = lm
        self.args = args
        self.tokenizer = self.init_tokenizer()


        print('Use both 2d AND 3d information')
        self.unimol_encoder, self.ln_unimol, self.dictionary = self.init_3d_graph_encoder(args)
        self.d2_graph_encoder, self.ln_d2_graph = self.init_2d_graph_encoder() ###need to add 2d gnn specific arguments


        self.tune_gnn = tune_gnn
        if not tune_gnn:
            for name, param in self.d2_graph_encoder.named_parameters():
                param.requires_grad = False
            for name, param in self.unimol_encoder.named_parameters():
                param.requires_grad = False
            self.d2_graph_encoder = self.d2_graph_encoder.eval()
            self.unimol_encoder = self.unimol_encoder.eval()
            # self.d2_graph_encoder.train = disabled_train
            # self.unimol_encoder.train = disabled_train
            logging.info("freeze 2D AND 3D graph encoder")

        self.Qformer, self.query_tokens = self.init_Qformer(bert_name, num_query_token, 1024, 512, cross_attention_freq)
        #self.Qformer, self.query_tokens = self.init_Qformer(bert_name, num_query_token, 1024, 512, cross_attention_freq, self.args) #### d2_graph_encoder.num_features 대신 값 고정 : 512

#        self.lin_proj_2d = nn.Linear(1024, 512)

        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.graph_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.alpha = nn.Parameter(torch.tensor(0.5))

        print(f'Queries Aggregation Method: {self.args.agg_method}') ###################################

        self.gtm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temperature = temperature

    def contrast(self, features_graph, features_text, return_sim=False):
        '''
        features_graph: shape = [B, num_qs, D]
        features_text: shape = [B, D]
        '''
        batch_size = features_graph.size(0)

        # normalized features: already normalized in the forward function
        # features_graph = F.normalize(features_graph, dim=-1)
        # features_text = F.normalize(features_text, dim=-1)

        # cosine similarity as logits
        sim_q2t = (features_graph.unsqueeze(1) @ features_text.unsqueeze(-1)).squeeze() # shape = [B, 1, num_qs, D]; shape = [B, D, 1]; output shape = [B, B, num_qs]
        sim_g2t, _ = sim_q2t.max(-1) # shape = [B, B]

        logits_per_graph = sim_g2t / self.temperature
        logits_per_text = logits_per_graph.t()

        labels = torch.arange(batch_size, dtype=torch.long, device=self.device)  # 大小为B
        loss_graph = F.cross_entropy(logits_per_graph, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_graph + loss_text) / 2

        if return_sim:
            return logits_per_graph, logits_per_text, loss
        else:
            return loss

    def contrast_global(self, features_graph, features_text, features_graph_all, features_text_all, return_sim=False):
        '''
        features_graph: shape = [B, num_qs, D]
        features_text: shape = [B, D]
        features_text_all: shape = [B * num_gpus, D]
        features_graph_all: shape = [B * num_gpus, num_qs, D]
        '''
        bs = features_graph.size(0)

        # cosine similarity as logits
        sim_q2t = (features_graph.unsqueeze(1) @ features_text_all.unsqueeze(-1)).squeeze(dim=-1) # shape = [B, 1, num_qs, D]; shape = [B * num_gpus, D, 1]; output shape = [B, B * num_gpus, num_qs]
        sim_g2t, _ = sim_q2t.max(-1) # shape = [B, B * num_gpus]

        logits_per_graph = sim_g2t / self.temperature
    
        sim_t2q = (features_text.unsqueeze(1).unsqueeze(1) @ features_graph_all.permute(0, 2, 1)).squeeze(dim=-2) # shape = [B, 1, 1, D]; [B*num_gpus, D, num_qs]; output shape = [B, B*num_gpus, 1, num_qs]
        sim_t2g, _ = sim_t2q.max(-1)
        logits_per_text = sim_t2g / self.temperature

        # labels = torch.arange(bs, dtype=torch.long, device=self.device)
        rank = dist.get_rank()
        labels = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(self.device)

        loss_graph = F.cross_entropy(logits_per_graph, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_graph + loss_text) / 2

        if return_sim:
            return logits_per_graph[:, rank*bs:rank*bs+bs], logits_per_text[:, rank*bs:rank*bs+bs], loss
        else:
            return loss

    def forward(self, batch):
        device = self.device

        '''
        Use both 2d and 3d information
        '''
            
        d3_batch, text_batch, d2_batch = batch

        batch_node_3d, batch_mask_3d = self.unimol_encoder(d3_batch[0], d3_batch[1], d3_batch[2])
        # batch_node_2d, batch_mask_2d = self.d2_graph_encoder(d2_batch)
        batch_mask_2d = torch.sum(torch.abs(d2_batch[1]), dim=-1) != 0
        # batch_node_2d = self.d2_graph_encoder(*d2_batch) #######
        batch_node_2d, _ = self.d2_graph_encoder(d2_batch[0], batch_mask_2d, d2_batch[1], d2_batch[2], None) # output : embedding / src_mask

        if not self.tune_gnn:
            batch_node_3d = batch_node_3d.detach()
            batch_node_2d = batch_node_2d.detach()  

        
        batch_size = batch_node_2d.shape[0]
        batch_node_3d = self.ln_unimol(batch_node_3d)

#        batch_node_2d = self.lin_proj_2d(batch_node_2d)
        batch_node_2d = self.ln_d2_graph(batch_node_2d)

        query_tokens = self.query_tokens.expand(batch_size, -1, -1)

        query_output_2d = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=batch_node_2d,
            encoder_attention_mask=batch_mask_2d,
            use_cache=True,
            return_dict=True,
            is_2d=True,
        )

        query_output_3d = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=batch_node_3d,
            encoder_attention_mask=batch_mask_3d, # fixme: check whether this mask is correct
            use_cache=True,
            return_dict=True,
            is_2d=False,
        )

        query_output_2d_3d = {}

        ### Combine Queries ###
        ###### 240617 kjh ########
        if self.args.agg_method == 'linear_combination':
            query_output_2d_3d['last_hidden_state'] = self.alpha * query_output_2d.last_hidden_state + (1 - self.alpha) * query_output_3d.last_hidden_state
        elif self.args.agg_method == 'concat':
            query_output_2d_3d['last_hidden_state'] = torch.cat((query_output_2d.last_hidden_state, query_output_3d.last_hidden_state), dim=1)
            # print(query_output_2d_3d)
        else:
            raise AggregationMethodError(f"Invalid aggregation method: {self.args.agg_method}")
        ########################
        #query_output_2d_3d['past_key_values']: 똑같이 하기엔 좀 곤란한 것 같다,, 
        

        graph_feats = self.graph_proj(query_output_2d_3d['last_hidden_state']) # shape = [B, num_q, D]
        text_output = self.Qformer.bert(text_batch['input_ids'], attention_mask=text_batch['attention_mask'], return_dict=True) # shape = [B, n_max, D]
        text_feats = self.text_proj(text_output.last_hidden_state[:, 0, :])
        
        text_feats, graph_feats = F.normalize(text_feats, p=2, dim=-1), F.normalize(graph_feats, p=2, dim=-1)
        #text_feats_all, graph_feats_all = pl_concat_all_gather(text_feats), pl_concat_all_gather(graph_feats) # shape = [B * num_gpus, D]
#        sim_g2t, sim_t2g, loss_gtc = self.contrast_global(graph_feats, text_feats, graph_feats_all, text_feats_all, return_sim=True)
        sim_g2t, sim_t2g, loss_gtc = self.contrast(graph_feats, text_feats, return_sim=True)

#        print(f'loss_gtc: {loss_gtc}')

        ###============== Molecule-text Matching ===================###
        loss_gtm = 0
        if self.gtm:

            batch_size = batch_node_2d.shape[0]
            ## not aggregate global tensor because of their different shapes
            text_ids_world = text_batch['input_ids']
            text_mask_world = text_batch['attention_mask']
            with torch.no_grad():
                weights_t2g = F.softmax(sim_t2g, dim=1) + 1e-4
                weights_t2g.fill_diagonal_(0)
                weights_g2t = F.softmax(sim_g2t, dim=1) + 1e-4
                weights_g2t.fill_diagonal_(0)

            # select a negative graph for each text
            d2_graph_embeds_neg, d3_graph_embeds_neg = [], []
            d2_graph_mask_neg, d3_graph_mask_neg = [], []

            for b in range(batch_size):
                neg_idx = torch.multinomial(weights_t2g[b], 1).item()
                d2_graph_embeds_neg.append(batch_node_2d[neg_idx])
                d2_graph_mask_neg.append(batch_mask_2d[neg_idx])
                d3_graph_embeds_neg.append(batch_node_3d[neg_idx])
                d3_graph_mask_neg.append(batch_mask_3d[neg_idx])

            d2_graph_embeds_neg = torch.stack(d2_graph_embeds_neg, dim=0)
            d2_graph_mask_neg = torch.stack(d2_graph_mask_neg, dim=0)
            d3_graph_embeds_neg = torch.stack(d3_graph_embeds_neg, dim=0)
            d3_graph_mask_neg = torch.stack(d3_graph_mask_neg, dim=0)
            
            # select a negative text for each image
            text_ids_neg = []
            text_atts_neg = []
            for b in range(batch_size):
                neg_idx = torch.multinomial(weights_g2t[b], 1).item()
                text_ids_neg.append(text_ids_world[neg_idx])
                text_atts_neg.append(text_mask_world[neg_idx])

            text_ids_neg = torch.stack(text_ids_neg, dim=0)
            text_atts_neg = torch.stack(text_atts_neg, dim=0)

            text_ids_all = torch.cat(
                [text_ids_world, text_ids_world, text_ids_neg], dim=0
            )  # pos, pos, neg
            text_atts_all = torch.cat(
                [text_mask_world , text_mask_world , text_atts_neg],
                dim=0,
            )

            query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
            query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long, device=device)
            attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

            d2_graph_embeds_all = torch.cat([batch_node_2d, d2_graph_embeds_neg, batch_node_2d], dim=0)  # pos, neg, pos
            d2_graph_atts_all = torch.cat([batch_mask_2d, d2_graph_mask_neg, batch_mask_2d], dim=0)

            d3_graph_embeds_all = torch.cat([batch_node_3d, d3_graph_embeds_neg, batch_node_3d], dim=0)
            d3_graph_atts_all = torch.cat([batch_mask_3d, d3_graph_mask_neg, batch_mask_3d], dim=0)

            d2_output_itm = self.Qformer.bert(
                text_ids_all,
                query_embeds=query_tokens_itm,
                attention_mask=attention_mask_all,
                encoder_hidden_states=d2_graph_embeds_all,
                encoder_attention_mask=d2_graph_atts_all,
                return_dict=True,
                is_2d=True,
            )

            d3_output_itm = self.Qformer.bert(
                text_ids_all,
                query_embeds=query_tokens_itm,
                attention_mask=attention_mask_all,
                encoder_hidden_states=d3_graph_embeds_all,
                encoder_attention_mask=d3_graph_atts_all,
                return_dict=True,
                is_2d=False,
            )

            ######### 240617 kjh #####
            if self.args.agg_method == 'linear_combination':
                vl_combined_output = self.alpha * d2_output_itm.last_hidden_state + (1-self.alpha) * d3_output_itm.last_hidden_state
            elif self.args.agg_method == 'concat':
                vl_combined_output = torch.cat((d2_output_itm.last_hidden_state, d3_output_itm.last_hidden_state), dim=1)
            else:
                raise AggregationMethodError(f"Invalid aggregation method: {self.args.agg_method}")
            #######################
                

            vl_embeddings = vl_combined_output[:, : query_tokens_itm.size(1), :] # keep query tokens only
            vl_output = self.gtm_head(vl_embeddings)
            logits = vl_output.mean(dim=1)

            itm_labels = torch.cat(
                [torch.ones(batch_size, dtype=torch.long), torch.zeros(2 * batch_size, dtype=torch.long)],
                dim=0,
            ).to(device)
            loss_gtm = F.cross_entropy(logits, itm_labels)

        ##================= Molecule Captioning ========================##
        loss_lm = 0
        if self.lm:
            decoder_input_ids = text_batch['input_ids'].clone()
            decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
            labels = decoder_input_ids.masked_fill(
                decoder_input_ids == self.tokenizer.pad_token_id, -100
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=device)
            
            attention_mask = torch.cat([query_atts, text_batch['attention_mask']], dim=1)
            
            lm_output_d2 = self.Qformer(
                decoder_input_ids,
                attention_mask=attention_mask,
                past_key_values=query_output_2d.past_key_values,
                return_dict=True,
                labels=labels,
            )

            lm_output_d3 = self.Qformer(
                decoder_input_ids,
                attention_mask=attention_mask,
                past_key_values=query_output_3d.past_key_values,
                return_dict=True,
                labels=labels,
            )

            ##text KL divergence / contrastive loss 추가
            # 
            
            ######################## 240617 kjh ########################
            if self.args.agg_method == 'linear_combination': 
                loss_lm = self.alpha * lm_output_d2.loss + (1-self.alpha) * lm_output_d3.loss
            elif self.args.agg_method == 'concat':
                loss_lm = lm_output_d2.loss + lm_output_d3.loss
            else:
                raise AggregationMethodError(f"Invalid aggregation method: {self.args.agg_method}")
            ########################
        
        return BlipOutput(
            loss=loss_gtc + loss_gtm + self.args.lm_weight * loss_lm,
            loss_itc=loss_gtc,
            loss_itm=loss_gtm,
            loss_lm=loss_lm,
        )

    
    def graph_forward(self, graph, is_2d = True):


        '''
        only_2d -> graph_forward(is_2d = True) | only_3d -> graph_forward(is_2d=False)
        '''

        if is_2d: 
            # batch_node, batch_mask = self.d2_graph_encoder(graph)
            batch_mask_2d = torch.sum(torch.abs(graph[1]), dim=-1) != 0
            batch_node, batch_mask = self.d2_graph_encoder(graph[0], batch_mask_2d, graph[1], graph[2], None) # output : embedding / src_mask
            
            # batch_node = self.lin_proj_2d(batch_node)

            ln_graph = self.ln_d2_graph
        else: 
            batch_node, batch_mask = self.unimol_encoder(graph[0], graph[1], graph[2])
            ln_graph = self.ln_unimol
            
        batch_node = ln_graph(batch_node)

        query_tokens = self.query_tokens.expand(batch_node.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=batch_node,
            encoder_attention_mask=batch_mask, # fixme: check whether this mask is correct
            use_cache=False,
            return_dict=True,
            is_2d=is_2d,
        )
        # graph_feats = self.graph_proj(query_output.last_hidden_state) # shape = [B, num_q, D]
        # graph_feats = F.normalize(graph_feats, p=2, dim=-1)
        return query_output, batch_node, batch_mask

    def text_forward(self, text, mask):
        text_output = self.Qformer.bert(text, attention_mask=mask, return_dict=True) # shape = [B, n_max, D]
        text_feats = self.text_proj(text_output.last_hidden_state[:, 0, :] )
        text_feats = F.normalize(text_feats, dim=-1, p=2)
        return text_feats
    
    def compute_gtm(self, d2_node, d2_mask, d3_node, d3_mask, text_ids, text_atts, args):
        '''
        batch_node shape = [B, N, D]
        batch_mask shape = [B, N]
        text_ids shape = [B, N]
        text_atts shape = [B, N]
        '''

        if args.only_3d:

            
            query_tokens = self.query_tokens.expand(d3_node.shape[0], -1, -1) # shape = [B, Nq, D]
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                d3_node.device
            ) # shape = [B, Nq]
            attention_mask = torch.cat([query_atts, text_atts], dim=1) # shape = [B, Nq + N]
    
    
            output_gtm_3d = self.Qformer.bert(
                text_ids, 
                query_embeds = query_tokens, 
                attention_mask = attention_mask,
                encoder_hidden_states = d3_node,
                encoder_attention_mask = d3_mask,
                return_dict = True,
                is_2d = False
            )

            g1_combined_ouput = output_gtm_3d.last_hidden_state
    
    
        elif args.only_2d:

            
            query_tokens = self.query_tokens.expand(d2_node.shape[0], -1, -1) # shape = [B, Nq, D]
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                d2_node.device
            ) # shape = [B, Nq]
            attention_mask = torch.cat([query_atts, text_atts], dim=1) # shape = [B, Nq + N]
    
    
            output_gtm_2d = self.Qformer.bert(
                text_ids, 
                query_embeds = query_tokens, 
                attention_mask = attention_mask,
                encoder_hidden_states = d2_node,
                encoder_attention_mask = d2_mask,
                return_dict = True,
                is_2d = True
            )
    
    
            g1_combined_ouput = output_gtm_2d.last_hidden_state

        
        else:

                
            query_tokens = self.query_tokens.expand(d2_node.shape[0], -1, -1) # shape = [B, Nq, D]
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                d2_node.device
            ) # shape = [B, Nq]
            attention_mask = torch.cat([query_atts, text_atts], dim=1) # shape = [B, Nq + N]
    
    
            output_gtm_2d = self.Qformer.bert(
                text_ids, 
                query_embeds = query_tokens, 
                attention_mask = attention_mask,
                encoder_hidden_states = d2_node,
                encoder_attention_mask = d2_mask,
                return_dict = True,
                is_2d = True
            )
    
            output_gtm_3d = self.Qformer.bert(
                text_ids, 
                query_embeds = query_tokens, 
                attention_mask = attention_mask,
                encoder_hidden_states = d3_node,
                encoder_attention_mask = d3_mask,
                return_dict = True,
                is_2d = False
            )
    
            ####### 240617 kjh ####
            if self.args.agg_method == 'linear_combination':
                g1_combined_ouput = self.alpha * output_gtm_2d.last_hidden_state + (1 - self.alpha) * output_gtm_3d.last_hidden_state
            elif self.args.agg_method == 'concat':
                g1_combined_ouput = torch.cat((output_gtm_2d.last_hidden_state, output_gtm_3d.last_hidden_state), dim=1)
            else:
                raise AggregationMethodError(f"Invalid aggregation method: {self.args.agg_method}")
            ##############
            
  
        gl_embeddings = g1_combined_ouput[:, : query_tokens.size(1), :] # shape = [B, Nq, D]
        gtm_logit = self.gtm_head(gl_embeddings).mean(dim=1) # shape = [B, Nq, 2]
        # gtm_logit = F.softmax(gtm_logit, dim=-1)[:, 1] # select the axis of the positive class
        gtm_logit = gtm_logit[:, 1] # select the axis of the positive class
        
        return gtm_logit


    # @staticmethod
    # def add_model_specific_args(parent_parser):
    #     parser = parent_parser.add_argument_group("molqformer")
        
    #     parser.add_argument('--agg_method', type=str, default='linear_combination', choices=['linear_combination', 'concat'], help='the aggregation method to use')
        
    #     return parent_parser