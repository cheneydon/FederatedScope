#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import codecs
import os
import os.path as osp
import torch
from tqdm import tqdm
from federatedscope.nlp.metrics.generation.utils import test_rouge, tile
from federatedscope.nlp.metrics.generation.beam import GNMTGlobalScorer


def build_predictor(args, tokenizer, symbols, model, logger=None):
    scorer = GNMTGlobalScorer(args.alpha,length_penalty='wu')

    translator = Translator(args, model, tokenizer, symbols, global_scorer=scorer, logger=logger)
    return translator


class Translator(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
       logger(logging.Logger): logger.
    """

    def __init__(self,
                 args,
                 model,
                 vocab,
                 symbols,
                 global_scorer=None,
                 logger=None,
                 dump_beam=""):
        self.logger = logger
        self.cuda = args.visible_gpus != '-1'

        self.args = args
        self.model = model
        self.generator = self.model.generator
        self.vocab = vocab
        self.symbols = symbols
        self.start_token = symbols['BOS']
        self.end_token = symbols['EOS']

        self.global_scorer = global_scorer
        self.beam_size = args.beam_size
        self.min_length = args.min_length
        self.max_length = args.max_length
        self.dump_beam = dump_beam

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None

        # tensorboard_log_dir = args.model_path
        # self.tensorboard_writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")
        self.tensorboard_writer = None

        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def _build_target_tokens(self, pred):
        # vocab = self.fields["tgt"].vocab
        tokens = []
        for tok in pred:
            tok = int(tok)
            tokens.append(tok)
            if tokens[-1] == self.end_token:
                tokens = tokens[:-1]
                break
        tokens = [t for t in tokens if t < len(self.vocab)]
        tokens = self.vocab.DecodeIds(tokens).split(' ')
        return tokens

    def from_batch(self, translation_batch):
        batch = translation_batch["batch"]
        assert (len(translation_batch["gold_score"]) == len(translation_batch["predictions"]))

        src = batch['token_ids']
        tgt = batch['labels']
        batch_size = len(src)
        preds, pred_score, gold_score =  \
            translation_batch["predictions"], \
            translation_batch["scores"], \
            translation_batch["gold_score"]

        translations = []
        for b in range(batch_size):
            pred_sents = ' '.join(self.vocab.convert_ids_to_tokens(preds[b][0])).replace(' ##', '')
            gold_sents = ' '.join(self.vocab.convert_ids_to_tokens(tgt[b])).replace(' ##', '')
            raw_src = ' '.join(self.vocab.convert_ids_to_tokens(src[b])[:500]).replace(' ##', '')
            translation = (pred_sents, gold_sents, raw_src)
            translations.append(translation)

        return translations

    def translate(self,
                  data_iter,
                  step,
                  client_id):

        self.model.eval()
        pred_dir = osp.join(self.args.result_path, 'pred')
        src_dir = osp.join(self.args.result_path, 'src')
        tgt_dir = osp.join(self.args.result_path, 'tgt')
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(src_dir, exist_ok=True)
        os.makedirs(tgt_dir, exist_ok=True)

        can_path = osp.join(pred_dir, '%d.txt' % client_id)
        src_path = osp.join(src_dir, '%d.txt' % client_id)
        gold_path = osp.join(tgt_dir, '%d.txt' % client_id)
        self.can_out_file = codecs.open(can_path, 'w', 'utf-8')
        self.src_out_file = codecs.open(src_path, 'w', 'utf-8')
        self.gold_out_file = codecs.open(gold_path, 'w', 'utf-8')

        def _remove_special_tokens(sent):
            return sent.replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '').\
                        replace('[unused0]', '').replace('[unused3]', '').replace('[unused1]', '').\
                        replace(r' +', ' ').replace(' [unused2] ', '<q>').replace('[unused2]', '').strip()

        ct = 0
        with torch.no_grad():
            for batch in tqdm(data_iter):
                if self.args.recall_eval:
                    src, seg, mask_src, tgt = batch
                    gold_tgt_len = tgt.size(1)
                    self.min_length = gold_tgt_len + 20
                    self.max_length = gold_tgt_len + 60
                batch_data = self.translate_batch(batch)
                translations = self.from_batch(batch_data)

                for trans in translations:
                    pred, gold, src = trans
                    pred_str = _remove_special_tokens(pred)
                    gold_str = _remove_special_tokens(gold)
                    src_str = _remove_special_tokens(src)
                    if self.args.recall_eval:
                        _pred_str = ''
                        for sent in pred_str.split('<q>'):
                            can_pred_str = _pred_str + '<q>'+sent.strip()
                            if(len(can_pred_str.split()) >= len(gold_str.split()) + 10):
                                pred_str = _pred_str
                                break
                            else:
                                _pred_str = can_pred_str

                    self.can_out_file.write(pred_str + '\n')
                    self.gold_out_file.write(gold_str + '\n')
                    self.src_out_file.write(src_str + '\n')
                    ct += 1
                self.can_out_file.flush()
                self.gold_out_file.flush()
                self.src_out_file.flush()

        self.can_out_file.close()
        self.gold_out_file.close()
        self.src_out_file.close()

        rouges = None
        if step != -1:
            rouges = self._report_rouge(gold_path, can_path)
            # self.logger.info('Rouge results:\n{}'.format(rouge_results_to_str(rouges)))
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.add_scalar('test/rouge1-F', rouges['rouge_1_f_score'], step)
                self.tensorboard_writer.add_scalar('test/rouge2-F', rouges['rouge_2_f_score'], step)
                self.tensorboard_writer.add_scalar('test/rougeL-F', rouges['rouge_l_f_score'], step)
        # return rouges
        return {k: v for k, v in rouges.items()
                if k in {'rouge_1_f_score', 'rouge_2_f_score', 'rouge_l_f_score'}}

    def _report_rouge(self, gold_path, can_path):
        # self.logger.info("Calculating Rouge")
        results_dict = test_rouge(self.args.temp_dir, can_path, gold_path)
        return results_dict

    def translate_batch(self, batch, fast=False):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           fast (bool): enables fast beam search (may not support all features)

        Todo:
           Shouldn't need the original dataset.
        """
        with torch.no_grad():
            return self._fast_translate_batch(
                batch,
                self.max_length,
                min_length=self.min_length)

    def _fast_translate_batch(self,
                              batch,
                              max_length,
                              min_length=0):
        # TODO: faster code path for beam_size == 1.

        # TODO: support these blacklisted features.
        assert not self.dump_beam

        beam_size = self.beam_size
        # batch_size = batch.batch_size
        # src = batch.src
        # segs = batch.segs
        # mask_src = batch.mask_src
        device = self.model.encoder.device
        src = batch['token_ids'].to(device)
        segs = batch['token_type_ids'].to(device)
        mask_src = batch['attention_mask'].to(device)
        batch_size = len(src)

        src_features = self.model.encoder.bert(src, segs, mask_src).last_hidden_state
        dec_states = self.model.decoder.init_decoder_state(src, src_features, with_cache=True)

        # Tile states and memory beam_size times.
        dec_states.map_batch_fn(
            lambda state, dim: tile(state, beam_size, dim=dim))
        src_features = tile(src_features, beam_size, dim=0)
        batch_offset = torch.arange(
            batch_size, dtype=torch.long, device=device)
        beam_offset = torch.arange(
            0,
            batch_size * beam_size,
            step=beam_size,
            dtype=torch.long,
            device=device)
        alive_seq = torch.full(
            [batch_size * beam_size, 1],
            self.start_token,
            dtype=torch.long,
            device=device)

        # Give full probability to the first beam on the first step.
        topk_log_probs = (
            torch.tensor([0.0] + [float("-inf")] * (beam_size - 1),
                         device=device).repeat(batch_size))

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]  # noqa: F812

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["scores"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["gold_score"] = [0] * batch_size
        results["batch"] = batch

        for step in range(max_length):
            decoder_input = alive_seq[:, -1].view(1, -1)

            # Decoder forward.
            decoder_input = decoder_input.transpose(0,1)

            dec_out, dec_states = self.model.decoder(decoder_input, src_features, dec_states,
                                                     step=step)

            # Generator forward.
            log_probs = self.generator.forward(dec_out.transpose(0,1).squeeze(0))
            vocab_size = log_probs.size(-1)

            if step < min_length:
                log_probs[:, self.end_token] = -1e20

            # Multiply probs by the beam probability.
            log_probs += topk_log_probs.view(-1).unsqueeze(1)

            alpha = self.global_scorer.alpha
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha

            # Flatten probs into a list of possibilities.
            curr_scores = log_probs / length_penalty

            if self.args.block_trigram:
                cur_len = alive_seq.size(1)
                if cur_len > 3:
                    for i in range(alive_seq.size(0)):
                        fail = False
                        words = [int(w) for w in alive_seq[i]]
                        # words = [self.vocab.ids_to_tokens[w] for w in words]
                        words = self.vocab.convert_ids_to_tokens(words)
                        words = ' '.join(words).replace(' ##','').split()
                        if len(words) <= 3:
                            continue
                        trigrams = [(words[i-1],words[i],words[i+1]) for i in range(1,len(words)-1)]
                        trigram = tuple(trigrams[-1])
                        if trigram in trigrams[:-1]:
                            fail = True
                        if fail:
                            curr_scores[i] = -10e20

            curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
            topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)

            # Recover log probs.
            topk_log_probs = topk_scores * length_penalty

            # Resolve beam origin and true word ids.
            topk_beam_index = topk_ids.div(vocab_size, rounding_mode='trunc')
            topk_ids = topk_ids.fmod(vocab_size)

            # Map beam_index to batch_index in the flat representation.
            batch_index = (
                    topk_beam_index
                    + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
            select_indices = batch_index.view(-1)

            # Append last prediction.
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices),
                 topk_ids.view(-1, 1)], -1)

            is_finished = topk_ids.eq(self.end_token)
            if step + 1 == max_length:
                is_finished.fill_(1)
            # End condition is top beam is finished.
            end_condition = is_finished[:, 0].eq(1)
            # Save finished hypotheses.
            if is_finished.any():
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(1)
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    # Store finished hypotheses for this batch.
                    for j in finished_hyp:
                        hypotheses[b].append((
                            topk_scores[i, j],
                            predictions[i, j, 1:]))
                    # If the batch reached the end, save the n_best hypotheses.
                    if end_condition[i]:
                        best_hyp = sorted(
                            hypotheses[b], key=lambda x: x[0], reverse=True)
                        score, pred = best_hyp[0]

                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)
                non_finished = end_condition.eq(0).nonzero().view(-1)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break
                # Remove finished batches for the next step.
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished) \
                    .view(-1, alive_seq.size(-1))
            # Reorder states.
            select_indices = batch_index.view(-1)
            src_features = src_features.index_select(0, select_indices)
            dec_states.map_batch_fn(
                lambda state, dim: state.index_select(dim, select_indices))

        return results


class Translation(object):
    """
    Container for a translated sentence.

    Attributes:
        src (`LongTensor`): src word ids
        src_raw ([str]): raw src words

        pred_sents ([[str]]): words from the n-best translations
        pred_scores ([[float]]): log-probs of n-best translations
        attns ([`FloatTensor`]) : attention dist for each translation
        gold_sent ([str]): words from gold translation
        gold_score ([float]): log-prob of gold translation

    """

    def __init__(self, fname, src, src_raw, pred_sents,
                 attn, pred_scores, tgt_sent, gold_score):
        self.fname = fname
        self.src = src
        self.src_raw = src_raw
        self.pred_sents = pred_sents
        self.attns = attn
        self.pred_scores = pred_scores
        self.gold_sent = tgt_sent
        self.gold_score = gold_score

    def log(self, sent_number):
        """
        Log translation.
        """

        output = '\nSENT {}: {}\n'.format(sent_number, self.src_raw)

        best_pred = self.pred_sents[0]
        best_score = self.pred_scores[0]
        pred_sent = ' '.join(best_pred)
        output += 'PRED {}: {}\n'.format(sent_number, pred_sent)
        output += "PRED SCORE: {:.4f}\n".format(best_score)

        if self.gold_sent is not None:
            tgt_sent = ' '.join(self.gold_sent)
            output += 'GOLD {}: {}\n'.format(sent_number, tgt_sent)
            output += ("GOLD SCORE: {:.4f}\n".format(self.gold_score))
        if len(self.pred_sents) > 1:
            output += '\nBEST HYP:\n'
            for score, sent in zip(self.pred_scores, self.pred_sents):
                output += "[{:.4f}] {}\n".format(score, sent)

        return output
