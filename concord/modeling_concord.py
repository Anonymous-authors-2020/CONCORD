import logging

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
import torch.distributed as dist

from transformers.activations import gelu, gelu_new
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.models.bert.modeling_bert import (
    BertOnlyMLMHead,
    BertPreTrainedModel,
    BertModel,
    BertEncoder,
    BertPooler,
    BertPredictionHeadTransform,
    BERT_START_DOCSTRING,
    BERT_INPUTS_DOCSTRING,
)



logger = logging.getLogger(__name__)

BERT_PRETRAINED_MODEL_ARCHIVE_LIST = []


def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "gelu_new": gelu_new, "mish": mish}


BertLayerNorm = torch.nn.LayerNorm

class ConcordTreePredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.tree_vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.tree_vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class ConcordForContrastivePreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)  # lm_head
        self.tree_pred_head = ConcordTreePredictionHead(config)
        self.sim = nn.CosineSimilarity(dim=-1)
        # temperature for Contrastive Loss
        self.temp = config.temp
        # many negative is to control how we build negative pairs with hard negative samples
        # e.g. batch = [[A, A+, A-], [B, B+, B-]]
        # sim matrix with many negative: [[AA+, AB+, AA-, AB-], [BA+, BB+, BA-, BB-]]
        # sim matrix w/o many negative: [[AA+, AB+, AA-], [BA+, BB+, BB-]]
        self.many_negative = config.many_negative
        self.emb_pooler_type = config.emb_pooler_type
        self.mlm_weight = config.mlm_weight
        self.clr_weight = config.clr_weight
        self.tree_pred_weight = config.tree_pred_weight
        assert self.emb_pooler_type in ["cls", "avg"]

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
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
        tree_labels=None,
        mlm_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs
    ):
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
        assert attention_mask is not None and mlm_attention_mask is not None
        batch_size = input_ids.size(0)
        num_sent = input_ids.size(1)  # 2 means only positive counterparts, 3 means positive and hard negative counterparts
        # batch looks like [[A, A+, (A-)], [B, B+, (B-)]]
        input_ids = input_ids.view((-1, input_ids.size(-1)))  # (bs, num_sent, hidden) -> # (bs * num_sent, hidden)
        # input_ids inputs look like [A, A+, A-, B, B+, B-]
        attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs, num_sent, hidden) -> # (bs * num_sent, hidden)
        outputs = self.bert(
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

        if self.emb_pooler_type == 'avg':
            pooled_output = (sequence_output * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
        pooled_output = pooled_output.view((batch_size, num_sent, pooled_output.size(-1)))  # (bs * num_sent, hidden) -> # (bs, num_sent, hidden)

        # z1 is the original samples [A, B], z2 is positive counterparts [A+, B+]
        z1, z2 = pooled_output[:, 0], pooled_output[:, 1]  # (bs, hidden)
        if num_sent == 3:
            # z3 is the hard negative counterparts [A-, B-]
            z3 = pooled_output[:, 2]
        # TODO: check whether this is working for distributed training
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
        output = outputs[:2]

        # mlm_inputs batch look like [A, B]
        if mlm_input_ids is not None and mlm_labels is not None and tree_labels is not None:
            mlm_outputs = self.bert(
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
            prediction_scores = self.cls(mlm_sequence_output)
            mlm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1))

            tree_prediction_scores = self.tree_pred_head(mlm_sequence_output)
            tree_pred_loss = loss_fct(tree_prediction_scores.view(-1, self.config.tree_vocab_size), tree_labels.view(-1))
            total_loss = self.clr_weight * clr_loss + self.mlm_weight * mlm_loss + self.tree_pred_weight * tree_pred_loss

        output = (total_loss, mlm_loss, clr_loss, tree_pred_loss) + output  # add hidden states and attention if they are here

        return output  # (loss, mlm_loss, clr_loss), (sequence_output), (pooled_output)


@add_start_docstrings(
    """Class for pre-trained model evaluation . """,
    BERT_START_DOCSTRING,
)
class ConcordForContrastiveInference(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)

        self.init_weights()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
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
        outputs = self.bert(
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


@add_start_docstrings(
    """Class for binary classification. """,
    BERT_START_DOCSTRING,
)
class ConcordForCls(BertPreTrainedModel):
    def __init__(self, config, new_pooler):
        super().__init__(config)

        self.bert = BertModel(config)
        self.pooler = BertPooler(config) if new_pooler else None
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.emb_pooler_type = config.emb_pooler_type

        self.init_weights()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
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
        **kwargs
    ):
        assert attention_mask is not None
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output, pooled_output = outputs[:2]
        if self.emb_pooler_type == "avg":
            pooled_output = (sequence_output * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)

        if self.pooler is not None:
            pooled_output = self.pooler(sequence_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.output_proj(pooled_output)
        outputs = (logits,) + outputs[2:]

        assert labels is not None

        loss_fct = CrossEntropyLoss()
        cls_loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        outputs = (cls_loss,) + outputs

        return outputs

@add_start_docstrings(
    """Concord Model for POJ 104 MAP@R experiments""",
    BERT_START_DOCSTRING,
)
class ConcordForPOJ104Map(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.emb_pooler_type = config.emb_pooler_type
        self.init_weights()
        self.sim = nn.CosineSimilarity(dim=-1)
        # temperature for Contrastive Loss
        # self.temp = config.temp

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
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
        batch_size = input_ids.size(0)
        num_sent = input_ids.size(1)
        input_ids = input_ids.view((-1, input_ids.size(-1)))  # (bs, num_sent, hidden) -> # (bs * num_sent, hidden)
        attention_mask = attention_mask.view((-1, attention_mask.size(-1)))
        # attention_mask = input_ids.ne(0)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output, pooled_output = outputs[:2]
        if self.emb_pooler_type == 'avg':
            pooled_output = (sequence_output * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)

        pooled_output = pooled_output.view(
            (batch_size, num_sent, pooled_output.size(-1)))  # (bs * num_sent, hidden) -> # (bs, num_sent, hidden)

        z1, z2, z3 = pooled_output[:, 0], pooled_output[:, 1], pooled_output[:, 2]  # (bs, hidden)


        # -----implementation 1-----
        prob_1 = (z1 * z2).sum(-1)
        prob_2 = (z1 * z3).sum(-1)
        temp = torch.cat((z1, z2), 0)
        temp_labels = torch.cat((labels, labels), 0)
        prob_3 = torch.mm(z1, temp.t())
        # a = labels[:, None]
        # b = temp_labels[None, :]
        mask = labels[:, None] == temp_labels[None, :]
        prob_3 = prob_3 * (1 - mask.float()) - 1e9 * mask.float()

        prob = torch.softmax(torch.cat((prob_1[:, None], prob_2[:, None], prob_3), -1), -1)

        loss = torch.log(prob[:, 0] + 1e-10)
        loss = -loss.mean()

        return loss, z1


class ConcordForEncoder(BertPreTrainedModel):
    def __init__(self, config, new_pooler=None):
        super().__init__(config)
        self.bert = BertModel(config)
        self.emb_pooler_type = config.emb_pooler_type
        self.pooler = BertPooler(config) if new_pooler else None
        assert self.emb_pooler_type in ["cls", "avg"]

        self.init_weights()


    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
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
        outputs = self.bert(
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
        if self.pooler is not None:
            pooled_output = self.pooler(sequence_output)

        if self.emb_pooler_type == 'avg':
            pooled_output = (sequence_output * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)

        return pooled_output  # (loss, mlm_loss, clr_loss), (sequence_output), (pooled_output)
