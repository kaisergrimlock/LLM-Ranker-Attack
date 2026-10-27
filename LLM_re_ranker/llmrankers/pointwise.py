import json
from typing import List
from urllib.request import Request, urlopen
from .rankers import LlmRanker, SearchResult
try:
    from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoConfig
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding
    import torch
except ImportError:  # AutoJev HTTP mode does not require the local T5 stack.
    T5Tokenizer = T5ForConditionalGeneration = AutoConfig = None
    DataLoader = DataCollatorWithPadding = torch = None
try:
    from .pairwise import Text2TextGenerationDataset
except ImportError:  # Keep the HTTP adapter usable without the legacy stack.
    Text2TextGenerationDataset = None
try:
    from tqdm import tqdm
except ImportError:  # Only needed by the legacy local T5 implementation.
    tqdm = None


class PointwiseLlmRanker(LlmRanker):

    def __init__(self, model_name_or_path, tokenizer_name_or_path, device, method="qlm", batch_size=1, cache_dir=None):
        if torch is None or T5Tokenizer is None:
            raise RuntimeError("Local T5 pointwise ranking requires torch and transformers")
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name_or_path
                                                     if tokenizer_name_or_path is not None else
                                                     model_name_or_path,
                                                     cache_dir=cache_dir)
        self.config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        if self.config.model_type == 't5':
            self.llm = T5ForConditionalGeneration.from_pretrained(model_name_or_path,
                                                                  device_map='auto',
                                                                  torch_dtype=torch.float16 if device == 'cuda'
                                                                  else torch.float32,
                                                                  cache_dir=cache_dir)
        else:
            raise NotImplementedError(f"Model type {self.config.model_type} is not supported yet for pointwise :(")

        self.device = device
        self.method = method
        self.batch_size = batch_size

        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0

    def rerank(self, query: str, ranking: List[SearchResult]) -> List[SearchResult]:
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0

        if self.method == "qlm":
            prompt = "Passage: {text}\nPlease write a question based on this passage."
            data = [prompt.format(text=doc.text) for doc in ranking]
            dataset = Text2TextGenerationDataset(data, self.tokenizer)
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                collate_fn=DataCollatorWithPadding(
                    self.tokenizer,
                    max_length=512,
                    padding='longest',
                ),
                shuffle=False,
                drop_last=False,
                num_workers=4
            )

            labels = self.tokenizer.encode(f"<pad> {query}",
                                           return_tensors="pt",
                                           add_special_tokens=False).to(self.llm.device).repeat(self.batch_size, 1)
            current_id = 0
            with torch.no_grad():
                for batch_inputs in tqdm(loader):
                    self.total_compare += 1
                    self.total_prompt_tokens += batch_inputs['input_ids'].shape[0] * batch_inputs['input_ids'].shape[1]

                    batch_labels = labels if labels.shape[0] == len(batch_inputs['input_ids']) \
                        else labels[:len(batch_inputs['input_ids']), :]  # last batch might be smaller
                    self.total_prompt_tokens += batch_labels.shape[0] * batch_labels.shape[
                        1]  # we count decoder inputs as part of prompt.

                    batch_inputs = batch_inputs.to(self.llm.device)
                    logits = self.llm(input_ids=batch_inputs['input_ids'],
                                      attention_mask=batch_inputs['attention_mask'],
                                      labels=batch_labels).logits

                    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                    scores = loss_fct(logits.view(-1, logits.size(-1)), batch_labels.view(-1))
                    scores = -1 * scores.view(-1, batch_labels.size(-1)).sum(dim=1)  # neg log prob
                    for score in scores:
                        ranking[current_id].score = score.item()
                        current_id += 1

        elif self.method == "yes_no":
            prompt = "Passage: {text}\nQuery: {query}\nDoes the passage answer the query? Answer 'Yes' or 'No'"
            yes_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
            no_id = self.tokenizer.encode("No", add_special_tokens=False)[0]
            data = [prompt.format(text=doc.text, query=query) for doc in ranking]
            dataset = Text2TextGenerationDataset(data, self.tokenizer)
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                collate_fn=DataCollatorWithPadding(
                    self.tokenizer,
                    max_length=512,
                    padding='longest',
                ),
                shuffle=False,
                drop_last=False,
                num_workers=4
            )
            decoder_input_ids = torch.Tensor([self.tokenizer.pad_token_id]).to(self.llm.device, dtype=torch.long).repeat(self.batch_size, 1)
            current_id = 0
            with torch.no_grad():
                for batch_inputs in tqdm(loader):
                    self.total_compare += 1
                    self.total_prompt_tokens += batch_inputs['input_ids'].shape[0] * batch_inputs['input_ids'].shape[1]

                    batch_inputs = batch_inputs.to(self.llm.device)

                    batch_decoder_input_ids = decoder_input_ids if decoder_input_ids.shape[0] == len(batch_inputs['input_ids']) \
                        else decoder_input_ids[:len(batch_inputs['input_ids']), :]  # last batch might be smaller

                    self.total_prompt_tokens += batch_decoder_input_ids.shape[0] * batch_decoder_input_ids.shape[
                        1]

                    logits = self.llm(input_ids=batch_inputs['input_ids'],
                                      attention_mask=batch_inputs['attention_mask'],
                                      decoder_input_ids=batch_decoder_input_ids).logits
                    yes_scores = logits[:, :, yes_id]
                    no_scores = logits[:, :, no_id]
                    batch_scores = torch.cat((yes_scores, no_scores), dim=1)
                    batch_scores = torch.nn.functional.softmax(batch_scores, dim=1)
                    scores = batch_scores[:, 0]
                    for score in scores:
                        ranking[current_id].score = score.item()
                        current_id += 1

        ranking = sorted(ranking, key=lambda x: x.score, reverse=True)
        return ranking

    def truncate(self, text, length):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(text)[:length])


class MonoT5LlmRanker(PointwiseLlmRanker):
    def rerank(self, query: str, ranking: List[SearchResult]) -> List[SearchResult]:
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0
        prompt = "Query: {query} Document: {document} Relevant:"
        data = [prompt.format(query=query, document=doc.text) for doc in ranking]
        dataset = Text2TextGenerationDataset(data, self.tokenizer)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=DataCollatorWithPadding(
                self.tokenizer,
                max_length=512,
                padding='longest',
            ),
            shuffle=False,
            drop_last=False,
            num_workers=4
        )
        decoder_input_ids = torch.Tensor([self.llm.config.decoder_start_token_id]).to(self.llm.device, dtype=torch.long).repeat(
            self.batch_size, 1)
        current_id = 0
        with torch.no_grad():
            for batch_inputs in tqdm(loader):
                self.total_compare += 1
                self.total_prompt_tokens += batch_inputs['input_ids'].shape[0] * batch_inputs['input_ids'].shape[1]

                batch_inputs = batch_inputs.to(self.llm.device)

                batch_decoder_input_ids = decoder_input_ids if decoder_input_ids.shape[0] == len(
                    batch_inputs['input_ids']) \
                    else decoder_input_ids[:len(batch_inputs['input_ids']), :]  # last batch might be smaller

                self.total_prompt_tokens += batch_decoder_input_ids.shape[0] * batch_decoder_input_ids.shape[
                    1]

                logits = self.llm(input_ids=batch_inputs['input_ids'],
                                  attention_mask=batch_inputs['attention_mask'],
                                  decoder_input_ids=batch_decoder_input_ids).logits

                # 6136 and 1176 are the indexes of the tokens false and true in T5.
                batch_scores = logits[:, 0, [6136, 1176]]
                batch_scores = torch.nn.functional.softmax(batch_scores, dim=1)
                scores = batch_scores[:, 1]
                for score in scores:
                    ranking[current_id].score = score.item()
                    current_id += 1

        ranking = sorted(ranking, key=lambda x: x.score, reverse=True)
        return ranking


class AutoJevPointwiseRanker(LlmRanker):
    """Pointwise reranker backed by a local AutoJev HTTP service.

    The binary mode follows Jev's pointwise protocol: each query/document
    pair is judged independently with a binary decision and ranked by the
    true probability. Graded mode maps AutoJev's 0/1/2/3 probabilities to an
    expected relevance score.
    """

    def __init__(self, endpoint="http://127.0.0.1:8000/v1/systemone",
                 method="yes_no", model="autojev", timeout=180):
        if method not in {"yes_no", "graded"}:
            raise ValueError(f"Unsupported AutoJev pointwise method: {method}")
        self.endpoint = endpoint
        self.method = method
        self.model = model
        self.timeout = timeout
        self.total_compare = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def truncate(self, text, length):
        return " ".join(str(text).split()[:length])

    def _score(self, query, passage):
        if self.method == "yes_no":
            question = {
                "type": "noul",
                "instructions": "Judge whether the passage is relevant to the query.",
                "criteria": {
                    "true": "The passage is relevant and answers the query.",
                    "false": "The passage is not relevant to the query.",
                },
            }
        else:
            criteria = {
                "0": "Not relevant.",
                "1": "Related, but not relevant enough to answer the query.",
                "2": "Relevant and useful for answering the query.",
                "3": "Highly relevant and directly answers the query.",
            }
            score_labels = ("0", "1", "2", "3")
            question = {
                "type": "choice",
                "instructions": "Judge the passage's graded relevance to the query.",
                "criteria": criteria,
            }

        body = {
            "model": self.model,
            "state": f"Query:\n{query}\n\nPassage:\n{passage}",
            "questions": {"relevance": question},
        }
        request = Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            payload = json.load(response)
        if self.method == "yes_no":
            return float(payload["answers"]["relevance"]["noul"])
        probabilities = payload["answers"]["relevance"]["probabilities"]
        return sum(int(label) * float(probabilities.get(label, 0.0)) for label in score_labels)

    def rerank(self, query: str, ranking: List[SearchResult], **_kwargs) -> List[SearchResult]:
        self.total_compare = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        for document in ranking:
            document.score = self._score(query, document.text)
            self.total_compare += 1
        return sorted(ranking, key=lambda item: item.score, reverse=True)



