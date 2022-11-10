import logging

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.distributed as dist

from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.models.roberta.modeling_roberta import (
    RobertaPreTrainedModel,
    ROBERTA_START_DOCSTRING,
    RobertaModel,
    RobertaLMHead,
    RobertaPooler,
    ROBERTA_INPUTS_DOCSTRING,
    gelu
)


logger = logging.getLogger(__name__)

class ConcordRobertaTreePredictionHead(nn.Module):
    """Roberta Head for tree prediction."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.tree_vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.tree_vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias

@add_start_docstrings("""RoBERTa Model with a `language modeling` head on top for contrastive pretraining. """, ROBERTA_START_DOCSTRING)
class ConcordRobertaForContrastivePretraining(RobertaPreTrainedModel):
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = []

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.roberta = RobertaModel(config, add_pooling_layer=True)
        self.lm_head = RobertaLMHead(config)
        self.tree_pred_head = ConcordRobertaTreePredictionHead(config)
        self.sim = nn.CosineSimilarity(dim=-1)
        self.temp = config.temp
        self.many_negative = True
        self.emb_pooler_type = "cls"
        self.mlm_weight = config.mlm_weight
        self.clr_weight = config.clr_weight
        self.tree_pred_weight = config.tree_pred_weight

        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
        self,
        input_ids=None,
        mlm_input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mlm_labels=None,
        mlm_attention_mask=None,
        tree_labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        assert attention_mask is not None and mlm_attention_mask is not None
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.size(0)
        num_sent = input_ids.size(1)  # 2 means only positive counterparts, 3 means positive and hard negative counterparts
        # batch looks like [[A, A+, (A-)], [B, B+, (B-)]]
        input_ids = input_ids.view((-1, input_ids.size(-1)))  # (bs, num_sent, hidden) -> # (bs * num_sent, hidden)
        # input_ids inputs look like [A, A+, A-, B, B+, B-]
        attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs, num_sent, hidden) -> # (bs * num_sent, hidden)

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]
        pooled_output = pooled_output.view((batch_size, num_sent, pooled_output.size(-1)))  # (bs * num_sent, hidden) -> # (bs, num_sent, hidden)

        # sequence_output = outputs[0]
        # sequence_output = sequence_output.view((batch_size, num_sent, sequence_output.size(1), sequence_output.size(2)))  # (bs * num_sent, num_token, hidden) -> # (bs, num_sent, num_token, hidden)
        # orig_sequence_output = sequence_output[:, 0, :, :]  # (bs, num_token, hidden)
        # z1 is the original samples [A, B], z2 is positive counterparts [A+, B+]
        z1, z2 = pooled_output[:, 0], pooled_output[:, 1]  # (bs, hidden)
        if num_sent == 3:
            # z3 is the hard negative counterparts [A-, B-]
            z3 = pooled_output[:, 2]

        # Gather all embeddings if using distributed training
        if dist.is_initialized():
            # Gather hard negative
            if num_sent == 3:
                z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
                dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
                z3_list[dist.get_rank()] = z3
                z3 = torch.cat(z3_list, 0)

            # Dummy vectors for allgather
            z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
            z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
            # Allgather
            dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
            dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

            # Since allgather results do not have gradients, we replace the
            # current process's corresponding embeddings with original tensors
            z1_list[dist.get_rank()] = z1
            z2_list[dist.get_rank()] = z2
            # Get full batch embeddings: (bs x N, hidden)
            z1 = torch.cat(z1_list, 0)
            z2 = torch.cat(z2_list, 0)

        # cos_sim: [[sim(AA+), sim(AB+)], [sim(BA+), sim(BB+)]]
        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0)) / self.temp

        # Hard negative
        if num_sent == 3:
            if self.many_negative:
                z1_z3_cos = self.sim(z1.unsqueeze(1), z3.unsqueeze(0)) / self.temp  # (bs, bs)
            else:
                z1_z3_cos = self.sim(z1, z3).unsqueeze(1) / self.temp
            # cos_sim (num_sent==3): [[sim(AA+), sim(AB+), sim(AA-), sim(AB-)], [sim(BA+), sim(BB+), sim(BA-), sim(BB-)]]
            # cos_sim (num_sent==2): [[sim(AA+), sim(AB+), sim(AA-)], [sim(BA+), sim(BB+), sim(BB-)]]
            cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

        # Labels are to indicate the location of sim(AA+), sim(BB+)
        labels = torch.arange(cos_sim.size(0)).long().to(self.device)
        loss_fct = nn.CrossEntropyLoss()
        # Calculate loss with hard negatives
        if num_sent == 3:
            # Note that actual weights are actually ln(weights)
            z3_weight = self.config.hard_negative_weight
            weights = torch.tensor(
                [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (
                            z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
            ).to(self.device)
            cos_sim = cos_sim + weights
        clr_loss = loss_fct(cos_sim, labels)
        total_loss = clr_loss

        mlm_outputs = self.roberta(
            mlm_input_ids,
            attention_mask=mlm_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        mlm_sequence_output = mlm_outputs[0]
        # mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = self.lm_head(mlm_sequence_output)
        mlm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1))
        
        if tree_labels is not None:
            tree_prediction_scores = self.tree_pred_head(mlm_sequence_output)
            tree_pred_loss = loss_fct(tree_prediction_scores.view(-1, self.config.tree_vocab_size), tree_labels.view(-1))
            total_loss = self.clr_weight * clr_loss + self.mlm_weight * mlm_loss + self.tree_pred_weight * tree_pred_loss
            output = total_loss, mlm_loss, clr_loss, tree_pred_loss, None, None
            return output
        else:
            total_loss = self.clr_weight * clr_loss + self.mlm_weight * mlm_loss

            output = total_loss, mlm_loss, clr_loss, None
            return output
        
class ConcordRobertaForEncoder(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=True)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs
    ):
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
        assert attention_mask is not None
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output, pooled_output = outputs[:2]  # token rep (bs * num_sent, seq_len, hidden), [CLS] rep (bs * num_sent, hidden)

        return pooled_output


class ConcordRobertaForCls(RobertaPreTrainedModel):
    def __init__(self, config, new_pooler):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=True)
        self.pooler = RobertaPooler(config) if new_pooler else None
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_proj = nn.Linear(config.hidden_size, 2)
        self.init_weights()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        assert attention_mask is not None
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if self.pooler is not None:
            pooled_output = self.pooler(outputs[0])
        else:
            pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.output_proj(pooled_output)
        outputs = (logits,)

        assert labels is not None

        loss_fct = CrossEntropyLoss()
        cls_loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        outputs = (cls_loss,) + outputs

        return outputs

class ConcordRobertaForContrastiveInference(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config, add_pooling_layer=True)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs
    ):
        assert attention_mask is not None
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        return outputs[:2]  # (sequence_output), (pooled_output)
