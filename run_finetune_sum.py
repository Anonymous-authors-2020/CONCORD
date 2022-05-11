from __future__ import absolute_import
import os
from generation.bleu import computeMaps, bleuFromMaps
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
import torch.nn as nn
from generation.model import Seq2Seq
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AdamW, get_linear_schedule_with_warmup, AutoConfig, AutoTokenizer,
)

from concord.modeling_concord import ConcordForContrastiveInference as ConcordModel
from generation.bleu_for_code import _bleu as bleu_for_code

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO
)
logger = logging.getLogger(__name__)


class Example(object):
    """A single training/test example."""

    def __init__(
            self, idx, source, target,
    ):
        self.idx = idx
        self.source = source
        self.target = target


def read_examples(filename, src_tag, tgt_tag):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            code = js[src_tag].replace('\n', '')
            nl = js[tgt_tag].replace('\n', '')
            nl = js[tgt_tag].replace('\n', '')
            examples.append(
                Example(
                    idx=idx,
                    source=code,
                    target=nl,
                )
            )
    return examples


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(
            self, example_id, source_ids, target_ids, source_mask, target_mask,
    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask


def convert_examples_to_features(examples, tokenizer, args, stage=None):
    features = []
    input_len, output_len = [], []
    for example_index, example in enumerate(tqdm(examples)):
        # source
        source_tokens = tokenizer.tokenize(example.source)
        input_len.append(len(source_tokens))
        source_tokens = [tokenizer.cls_token] + source_tokens[:args.max_source_length - 2] + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_tokens))
        padding_length = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length
        # target
        if stage == "test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)
        output_len.append(len(target_tokens))
        target_tokens = [tokenizer.cls_token] + target_tokens[:args.max_target_length - 2] + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length
        """if example_index < 2:
            if stage == 'train':
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.idx))

                logger.info("source_tokens as list : {}".format([x.replace('\u0120', '_') for x in source_tokens]))
                logger.info(f"Detoken source tokens : " + detokenize(" ".join(source_tokens)))
                logger.info("source_ids         : {}".format(' '.join(map(str, source_ids))))
                logger.info("source_mask        : {}".format(' '.join(map(str, source_mask))))

                logger.info("target_tokens as list : {}".format([x.replace('\u0120', '_') for x in target_tokens]))
                logger.info("Detoken target tokens : " + detokenize(" ".join(target_tokens)))
                logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))
                logger.info("target_mask: {}".format(' '.join(map(str, target_mask))))"""
        features.append(
            InputFeatures(
                example_index,
                source_ids,
                target_ids,
                source_mask,
                target_mask,
            )
        )
    print(min(input_len), max(input_len), np.mean(input_len), np.histogram(input_len))
    print(min(output_len), max(output_len), np.mean(output_len), np.histogram(output_len))
    return features


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def detokenize(text):
    assert isinstance(text, str)
    return text.replace(' ', "").replace('\u0120', " ").replace("▁", " ").strip()


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--model_name_or_path", default=None, type=str, required=True,
        help="Path to pre-trained model: e.g. roberta-base"
    )
    parser.add_argument(
        "--output_dir", default=None, type=str, required=True,
        help="The output directory where the model predictions and checkpoints will be written."
    )
    parser.add_argument(
        "--load_model_path", default=None, type=str,
        help="Path to trained model: Should contain the .bin files"
    )
    ## Other parameters
    parser.add_argument(
        "--train_filename", default=None, type=str,
        help="The train filename. Should contain the .jsonl files for this task."
    )
    parser.add_argument(
        "--dev_filename", default=None, type=str,
        help="The dev filename. Should contain the .jsonl files for this task."
    )
    parser.add_argument(
        "--test_filename", default=None, type=str,
        help="The test filename. Should contain the .jsonl files for this task."
    )
    parser.add_argument(
        "--src_tag", type=str, default="code_token_string",
        help="Name of the source key in the data"
    )
    parser.add_argument(
        "--tgt_tag", type=str, default="docstring_token_string",
        help="Name of the target key in the data"
    )

    parser.add_argument(
        "--max_source_length", default=64, type=int,
        help="The maximum total source sequence length after tokenization. Sequences longer than this will be"
             "truncated, sequences shorter will be padded."
    )
    parser.add_argument(
        "--max_target_length", default=32, type=int,
        help="The maximum total target sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run eval on the dev set.")

    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")

    parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument(
        '--gradient_accumulation_steps', type=int, default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass."
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int, help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--patience", default=20, type=int, help="Patience counter!")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--lang", type=str, default=None, help="Language for generation task, if the generated lang is a code."
    )
    # print arguments
    args = parser.parse_args()
    logger.info(args)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    args.device = device
    # Set seed
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer_kwargs = {
        "cache_dir": "cache",
        "use_fast": True,
        "config": config,
        "do_lower_case": False,
        "do_basic_tokenize": False,
        "tokenize_chinese_chars": False,
        "bos_token": "<s>",
        "eos_token": "</s>",
        "use_auth_token": None,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **tokenizer_kwargs)

    logger.info(
        "DEBUG: Tokenizer: %s, dict size = %d, mask token = %s,  cls token = %s, pad token = %s",
        tokenizer.__class__.__name__, len(tokenizer.vocab), tokenizer.mask_token, tokenizer.cls_token,
        tokenizer.pad_token
    )

    encoder = ConcordModel.from_pretrained(
        args.model_name_or_path,
        config=config,
    ).bert

    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_hidden_layers)
    model = Seq2Seq(
        encoder=encoder, decoder=decoder, config=config,
        beam_size=args.beam_size, max_length=args.max_target_length,
        sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id
    )
    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model = DDP(model)
    elif args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    original_train_batch_size = args.train_batch_size
    args.train_batch_size = int(original_train_batch_size / args.gradient_accumulation_steps)
    original_eval_batch_size = args.eval_batch_size
    args.eval_batch_size = int(original_eval_batch_size / args.gradient_accumulation_steps)
    logger.info(f"Train Batch Size : {args.train_batch_size} Valid Batch Size : {args.eval_batch_size}")
    _examples = read_examples(args.test_filename, args.src_tag, args.tgt_tag)
    _features = convert_examples_to_features(_examples, tokenizer, args, stage='train')

    if args.do_train:
        # Prepare training data loader
        train_examples = read_examples(args.train_filename, args.src_tag, args.tgt_tag)
        train_features = convert_examples_to_features(train_examples, tokenizer, args, stage='train')
        all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
        all_target_mask = torch.tensor([f.target_mask for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler,
            batch_size=args.train_batch_size
        )
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay
            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        t_total = (len(train_dataloader) // args.gradient_accumulation_steps) * args.num_train_epochs
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(t_total * 0.1), num_training_steps=t_total
        )

        # Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples   = %d", len(train_examples))
        logger.info("  Org Batch size = %d", original_train_batch_size)
        logger.info("  Batch size     = %d", args.train_batch_size)
        logger.info("  Num epoch      = %d", args.num_train_epochs)

        model.train()
        dev_dataset = {}
        nb_tr_examples, nb_tr_steps, tr_loss, global_step, best_bleu, best_loss = 0, 0, 0, 0, 0, 1e6
        train_loss = 9e9
        patience = 0
        for epoch in range(args.num_train_epochs):
            bar = tqdm(train_dataloader, total=len(train_dataloader))
            for batch in bar:
                batch = tuple(t.to(device) for t in batch)
                source_ids, source_mask, target_ids, target_mask = batch
                loss, _, _ = model(
                    source_ids=source_ids, source_mask=source_mask, target_ids=target_ids, target_mask=target_mask
                )
                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()
                train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                bar.set_description("epoch {} loss {}".format(epoch, train_loss))
                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()
                if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

            if args.do_eval:
                # Eval model with dev dataset
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                if 'dev_loss' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_loss']
                else:
                    eval_examples = read_examples(args.dev_filename, args.src_tag, args.tgt_tag)
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='dev')
                    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                    all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
                    all_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)
                    all_target_mask = torch.tensor([f.target_mask for f in eval_features], dtype=torch.long)
                    eval_data = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask)
                    dev_dataset['dev_loss'] = eval_examples, eval_data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                logger.info("\n***** Running evaluation *****")
                logger.info("  Num examples   = %d", len(eval_examples))
                logger.info("  Org Batch size = %d", original_eval_batch_size)
                logger.info("  Batch size     = %d", args.eval_batch_size)

                # Start Evaling model
                model.eval()
                eval_loss, tokens_num = 0, 0
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids, source_mask, target_ids, target_mask = batch

                    with torch.no_grad():
                        _, loss, num = model(
                            source_ids=source_ids, source_mask=source_mask,
                            target_ids=target_ids, target_mask=target_mask
                        )
                    eval_loss += loss.sum().item()
                    tokens_num += num.sum().item()
                # Pring loss of dev dataset
                model.train()
                eval_loss = eval_loss / tokens_num
                result = {
                    'eval_ppl': round(np.exp(eval_loss), 5),
                    'global_step': global_step + 1,
                    'train_loss': round(train_loss, 5)
                }
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  " + "*" * 20)

                # save last checkpoint
                last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)
                if eval_loss < best_loss:
                    logger.info("  Best ppl:%s", round(np.exp(eval_loss), 5))
                    logger.info("  " + "*" * 20)
                    best_loss = eval_loss
                    # Save best checkpoint for best ppl
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)

                    # Calculate bleu
                if 'dev_bleu' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_bleu']
                else:
                    eval_examples = read_examples(args.test_filename, args.src_tag, args.tgt_tag)
                    eval_examples = random.sample(eval_examples, min(20000, len(eval_examples)))
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
                    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                    all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
                    eval_data = TensorDataset(all_source_ids, all_source_mask)
                    dev_dataset['dev_bleu'] = eval_examples, eval_data

                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                model.eval()
                p = []
                logger.info("Starting bleu evaluation")
                for batch in tqdm(eval_dataloader):
                    batch = tuple(t.to(device) for t in batch)
                    source_ids, source_mask = batch
                    with torch.no_grad():
                        preds = model(source_ids=source_ids, source_mask=source_mask)
                        for pred in preds:
                            t = pred[0].cpu().numpy()
                            t = list(t)
                            if 0 in t:
                                t = t[:t.index(0)]
                            text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                            p.append(text)
                model.train()
                predictions = []
                dev_accs = []
                gold_fn = os.path.join(args.output_dir, "dev.gold")
                output_fn = os.path.join(args.output_dir, "dev.output")
                with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1:
                    for ref, gold in zip(p, eval_examples):
                        ref_detokenized = detokenize(ref)
                        gold_detokenized = detokenize(gold.target)
                        dev_accs.append(ref_detokenized.strip() == gold_detokenized.strip())
                        predictions.append(str(gold.idx) + '\t' + ref_detokenized)
                        f.write(str(gold.idx) + '\t' + ref_detokenized + '\n')
                        f1.write(str(gold.idx) + '\t' + gold_detokenized + '\n')
                if args.lang is None or args.lang.lower() == "nl":
                    (goldMap, predictionMap) = computeMaps(
                        predictions, os.path.join(args.output_dir, "dev.gold")
                    )
                    dev_bleu = round(bleuFromMaps(goldMap, predictionMap)[0], 2)
                    logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
                    logger.info("  " + "*" * 20)
                else:
                    em = np.mean(dev_accs) * 100
                    bleu = bleu_for_code(ref_file=gold_fn, trans_file=output_fn)
                    logger.info("  %s = %.2f " % ("EM       ", em))
                    logger.info("  %s = %.2f " % ("BLEU     ", bleu))
                    dev_bleu = bleu + em
                if dev_bleu > best_bleu:
                    patience = 0
                    logger.info("  Best bleu : %s", dev_bleu)
                    logger.info("  " + "*" * 20)
                    best_bleu = dev_bleu
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    logger.info("Saving model to %s" % (output_model_file))
                    torch.save(model_to_save.state_dict(), output_model_file)
                else:
                    patience += 1
                    logger.info("BLEU did not increase for %d epochs %.2f %.2f" % (patience, dev_bleu, best_bleu))
                    if patience >= args.patience:
                        logger.info("Hitting maximum patience %d" % patience)
                        break

    if args.do_test:
        files = []
        output_model = os.path.join(args.output_dir, "checkpoint-last", "pytorch_model.bin")
        model.load_state_dict(torch.load(open(output_model, "rb")))
        if args.test_filename is not None:
            files.append(args.test_filename)
        for idx, file in enumerate(files):
            logger.info("Test file: {}".format(file))
            eval_examples = read_examples(file, args.src_tag, args.tgt_tag)
            eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
            all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
            all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
            eval_data = TensorDataset(all_source_ids, all_source_mask)
            # Calculate bleu
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
            model.eval()
            p = []
            for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                batch = tuple(t.to(device) for t in batch)
                source_ids, source_mask = batch
                with torch.no_grad():
                    preds = model(source_ids=source_ids, source_mask=source_mask)
                    for pred in preds:
                        t = pred[0].cpu().numpy()
                        t = list(t)
                        if 0 in t:
                            t = t[:t.index(0)]
                        text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                        p.append(text)          
            model.train()
            predictions = []
            output_fn = os.path.join(args.output_dir, "test_{}.output".format(str(idx)))
            gold_fn = os.path.join(args.output_dir, "test_{}.gold".format(str(idx)))
            dev_accs = []
            with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1:
                for ref, gold in zip(p, eval_examples):
                    ref_detokenized = detokenize(ref)
                    gold_detokenized = detokenize(gold.target)
                    dev_accs.append(ref_detokenized.strip() == gold_detokenized.strip())
                    predictions.append(str(gold.idx) + '\t' + ref_detokenized)
                    f.write(str(gold.idx) + '\t' + ref_detokenized + '\n')
                    f1.write(str(gold.idx) + '\t' + gold_detokenized + '\n')
            if args.lang is None or args.lang.lower() == "nl":
                (goldMap, predictionMap) = computeMaps(
                    predictions, os.path.join(args.output_dir, "test_{}.gold".format(idx))
                )
                dev_bleu = round(bleuFromMaps(goldMap, predictionMap)[0], 2)
                logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
                logger.info("  " + "*" * 20)
            else:
                em = np.mean(dev_accs) * 100
                bleu = bleu_for_code(ref_file=gold_fn, trans_file=output_fn) * 100
                logger.info("  %s = %.2f " % ("EM       ", em))
                logger.info("  %s = %.2f " % ("BLEU     ", bleu))
                pass


if __name__ == "__main__":
    test_str = "▁Creates ▁an ▁Unicast Subject ▁with ▁the ▁given ▁internal ▁buffer ▁capacity ▁hint ▁and ▁a " \
               "▁callback ▁for ▁the ▁case ▁when ▁the ▁single ▁Subscriber ▁c ance ls ▁its ▁subscription ▁."
    detokenize(test_str)
    main()
