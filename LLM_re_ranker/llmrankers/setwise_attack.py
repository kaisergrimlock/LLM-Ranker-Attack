from typing import List
from .rankers import LlmRanker, SearchResult
import openai
import time
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import copy
from collections import Counter
import random
from prompts import JAILBREAK_PROMPTS
try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except ImportError:
    print("Seems vllm is not installed, RankR1SetwiseLlmRanker only supports vllm inference so far.")
import openai
from openai import OpenAI
import os
import json
from datetime import datetime

random.seed(929)


def save_detailed_log(prompt, raw_output, parsed_label, model_name, log_dir="debug_logs"):
    """
    Save detailed debugging information for output parsing issues.
    
    Args:
        prompt: Input prompt sent to the model
        raw_output: Raw response from the model before parsing
        parsed_label: Parsed label (A, B, C, etc.) or error message
        model_name: Name of the model being used
        log_dir: Directory to save logs
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    model_short = model_name.split("/")[-1]
    log_file = os.path.join(log_dir, f"{model_short}_{timestamp}.json")
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "prompt": prompt,
        "raw_output": raw_output,
        "parsed_label": parsed_label,
        "raw_output_repr": repr(raw_output),  # Shows hidden characters
        "raw_output_bytes": raw_output.encode('utf-8').hex() if raw_output else None
    }
    
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_entry, f, indent=2, ensure_ascii=False)
    
    return log_file


class SetwiseLlmRanker(LlmRanker):
    CHARACTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
                  "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W"]  # "Passage X" and "Passage Y" will be tokenized into 3 tokens, so we dont use for now

    def __init__(self,
                 model_name_or_path,
                 tokenizer_name_or_path,
                 device,
                 num_child=3,
                 k=10,
                 scoring='generation',
                 method="heapsort",
                 num_permutation=1,
                 cache_dir=None):

        self.model_name_or_path = model_name_or_path  # Save for logging
        self.device = device
        self.num_child = num_child
        self.num_permutation = num_permutation
        self.k = k
        self.config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        if self.config.model_type == 't5':
            self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name_or_path
                                                         if tokenizer_name_or_path is not None else
                                                         model_name_or_path,
                                                         cache_dir=cache_dir)
            self.llm = T5ForConditionalGeneration.from_pretrained(model_name_or_path,
                                                                  device_map='auto',
                                                                  torch_dtype=torch.float16 if device == 'cuda'
                                                                  else torch.float32,
                                                                  cache_dir=cache_dir)
            self.decoder_input_ids = self.tokenizer.encode("<pad> Passage",
                                                           return_tensors="pt",
                                                           add_special_tokens=False).to(self.device) if self.tokenizer else None

            self.target_token_ids = self.tokenizer.batch_encode_plus([f'<pad> Passage {self.CHARACTERS[i]}'
                                                                      for i in range(len(self.CHARACTERS))],
                                                                     return_tensors="pt",
                                                                     add_special_tokens=False,
                                                                     padding=True).input_ids[:, -1]
        elif self.config.model_type in ['llama', 'mistral', 'qwen3', 'gemma3', 'gemma3_text', 'qwen3_moe']:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
            self.tokenizer.use_default_system_prompt = False
            if 'vicuna' and 'v1.5' in model_name_or_path:
                self.tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\\'s questions.' %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 %}{{ system_message }}{% endif %}{% if message['role'] == 'user' %}{{ ' USER: ' + message['content'].strip() }}{% elif message['role'] == 'assistant' %}{{ ' ASSISTANT: ' + message['content'].strip() + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ ' ASSISTANT:' }}{% endif %}"
            self.llm = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                            device_map='auto',
                                                            torch_dtype=torch.bfloat16 if device == 'cuda'
                                                            else torch.float32,
                                                            cache_dir=cache_dir).eval()
            # set padding size left
            self.llm.config.padding_side = "left"
        else:
            raise NotImplementedError(f"Model type {self.config.model_type} is not supported yet for setwise:(")

        self.scoring = scoring
        self.method = method
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0

    def _extract_label_from_output(self, raw_output: str, num_docs: int) -> str:
        """
        Robust label extractor for LLM outputs.
        Prioritizes precision over recall to avoid spurious matches
        (important for ASR / reproducibility studies).
        
        Args:
            raw_output: Raw text output from the model
            num_docs: Number of documents (to validate label is in range)
            
        Returns:
            Single character label (A, B, C, etc.) or "INVALID" if extraction fails
        """
        if raw_output is None:
            return "INVALID"

        valid = {c.upper() for c in self.CHARACTERS[:num_docs]}
        text = str(raw_output).strip()
        if not text:
            return "INVALID"

        # Normalize: collapse whitespace, keep original for some checks
        norm = re.sub(r"\s+", " ", text)

        # -----------------------
        # Strategy 0: JSON-ish / structured outputs (high precision)
        # -----------------------
        # e.g., {"answer":"A"} or answer: "B"
        m = re.search(r'(?i)\b(answer|final|choice|label)\b\s*[:=]\s*["\']?([A-Z])["\']?', norm)
        if m and m.group(2).upper() in valid:
            return m.group(2).upper()



        # e.g., "Final answer: A", "Answer is B", "I choose C", "Most relevant: D"
        anchor_patterns = [
            r'(?i)\bfinal(?:\s+answer)?\b\s*[:\-]?\s*([A-Z])\b',
            r'(?i)\banswer(?:\s+is)?\b\s*[:\-]?\s*([A-Z])\b',
            r'(?i)\bchoice\b\s*[:\-]?\s*([A-Z])\b',
            r'(?i)\bi\s+(?:choose|pick|select)\b\s*[:\-]?\s*([A-Z])\b',
            r'(?i)\bmost\s+relevant\b\s*[:\-]?\s*([A-Z])\b',
            r'(?i)\btherefore\b.*?\b([A-Z])\b',
        ]
        for pat in anchor_patterns:
            m = re.search(pat, norm)
            if m:
                lab = m.group(1).upper()
                if lab in valid:
                    return lab
                
        # e.g., "PASSAGE A", "Passage: B", "Document C", "Option D"
        m = re.search(r'(?i)\b(passage|document|doc|option|candidate)\b\s*[:\-]?\s*([A-Z])\b', norm)
        if m and m.group(2).upper() in valid:
            return m.group(2).upper()

        # e.g., "A", "A.", "(B)", "**C**"
        first = re.match(r'^[\(\[\{<"\']*\s*([A-Z])[\)\]\}>,"\'.:\s]*', text)
        if first:
            lab = first.group(1).upper()
            if lab in valid:
                return lab

        # Only accept isolated tokens: " A " not "PASSAGE" or "AND"
        tokens = re.findall(r'\b([A-Z])\b', norm)
        # If multiple candidates appear, this is ambiguous; prefer INVALID over guessing
        candidates = [t for t in tokens if t.upper() in valid]
        if len(candidates) == 1:
            return candidates[0].upper()

        # e.g., explanation contains A/B but final line says "Answer: D"
        m_all = re.findall(r'(?i)\b(final|answer|choice)\b\s*[:\-]?\s*([A-Z])\b', norm)
        if m_all:
            lab = m_all[-1][1].upper()
            if lab in valid:
                return lab

        return "INVALID"

    def compare(self, query: str, docs: List, attack_prompt: str="none", attack_position: str="back"):
        self.total_compare += 1 if self.num_permutation == 1 else self.num_permutation

        # 校验参数
        if attack_prompt not in ("none", "so", "sd"):
            raise ValueError(f"Unknown attack prompt: {attack_prompt}")
        if attack_position not in ("front", "back"):
            raise ValueError(f"Unknown attack position: {attack_position}")

        passage_lines = []
        for i, doc in enumerate(docs):
            gt_rel = getattr(doc, "gt_rel", None)
            text = doc.text

            if gt_rel == 0 and attack_prompt != "none":
                if attack_prompt == "so":
                    if attack_position == "front":
                        text = JAILBREAK_PROMPTS["so"] + " " + text
                    else:
                        text = text + " " + JAILBREAK_PROMPTS["so"]
                else:
                    if attack_position == "front":
                        text = JAILBREAK_PROMPTS["sd"] + " " + text
                    else:
                        text = text + " " + JAILBREAK_PROMPTS["sd"]
            passage_lines.append(f'Passage {self.CHARACTERS[i]}: "{text}"')

        passages = "\n\n".join(passage_lines)
        
        # Build list of valid labels for this comparison (e.g., "A, B, C, D" for 4 docs)
        valid_labels = ", ".join(self.CHARACTERS[:len(docs)])

        input_text = f'Given a query "{query}", which of the following passages is the most relevant one to the query?\n\n' \
                     + passages + f'\n\nOutput only the passage label of the most relevant passage ({valid_labels}):'

        if self.scoring == 'generation':
            if self.config.model_type == 't5':

                if self.num_permutation == 1:
                    input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
                    self.total_prompt_tokens += input_ids.shape[1]

                    output_ids = self.llm.generate(input_ids,
                                                   decoder_input_ids=self.decoder_input_ids,
                                                   max_new_tokens=2)[0]

                    self.total_completion_tokens += output_ids.shape[0]

                    output = self.tokenizer.decode(output_ids,
                                                   skip_special_tokens=True).strip()
                    output = output[-1]
                else:
                    id_passage = [(i, p) for i, p in enumerate(docs)]
                    labels = [self.CHARACTERS[i] for i in range(len(docs))]
                    batch_data = []
                    for _ in range(self.num_permutation):
                        batch_data.append([random.sample(id_passage, len(id_passage)),
                                           random.sample(labels, len(labels))])

                    batch_ref = []
                    input_text = []
                    for batch in batch_data:
                        ref = []
                        passages = []
                        characters = []
                        for p, c in zip(batch[0], batch[1]):
                            ref.append(p[0])
                            passages.append(p[1].text)
                            characters.append(c)
                        batch_ref.append((ref, characters))
                        passages = "\n\n".join([f'Passage {characters[i]}: "{passages[i]}"' for i in range(len(passages))])
                        input_text.append(f'Given a query "{query}", which of the following passages is the most relevant one to the query?\n\n' \
                                          + passages + '\n\nOutput only the passage label of the most relevant passage:')

                    input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
                    self.total_prompt_tokens += input_ids.shape[1] * input_ids.shape[0]

                    output_ids = self.llm.generate(input_ids,
                                                   decoder_input_ids=self.decoder_input_ids.repeat(input_ids.shape[0], 1),
                                                   max_new_tokens=2)
                    output = self.tokenizer.batch_decode(output_ids[:, self.decoder_input_ids.shape[1]:],
                                                         skip_special_tokens=True)

                    # vote
                    candidates = []
                    for ref, result in zip(batch_ref, output):
                        result = result.strip().upper()
                        docids, characters = ref
                        if len(result) != 1 or result not in characters:
                            print(f"Unexpected output: {result}")
                            continue
                        win_doc = docids[characters.index(result)]
                        candidates.append(win_doc)

                    if len(candidates) == 0:
                        print(f"Unexpected voting: {output}")
                        output = "Unexpected voting."
                    else:
                        # handle tie
                        candidate_counts = Counter(candidates)
                        max_count = max(candidate_counts.values())
                        most_common_candidates = [candidate for candidate, count in candidate_counts.items() if
                                                  count == max_count]
                        if len(most_common_candidates) == 1:
                            output = self.CHARACTERS[most_common_candidates[0]]
                        else:
                            output = self.CHARACTERS[random.choice(most_common_candidates)]

            elif self.config.model_type in ['llama', 'mistral', 'qwen3', 'gemma3', 'gemma3_text', 'qwen3_moe']:
                conversation = [{"role": "user", "content": input_text}]

                # Add enable_thinking=False for Qwen models to disable thinking mode
                if self.config.model_type == 'qwen3' or self.config.model_type == 'qwen3_moe':
                    prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                else:
                    prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                prompt += " Passage:"

                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
                self.total_prompt_tokens += input_ids.shape[1]

                output_ids = self.llm.generate(input_ids,
                                               do_sample=False,
                                               temperature=0.0,
                                               top_p=None,
                                               max_new_tokens=20)[0]  # Increased to support various LLM generation lengths

                self.total_completion_tokens += output_ids.shape[0]

                # Decode the output
                raw_output = self.tokenizer.decode(output_ids[input_ids.shape[1]:],
                                               skip_special_tokens=True).strip()

                #print(raw_output)
                #print(prompt)
                # Extract label using robust extraction logic (matching working version)
                output = self._extract_label_from_output(raw_output, len(docs))
                #print("Raw output:", repr(raw_output), "Extracted label:", output)
                # Save detailed log for debugging if output is invalid
                if output == "INVALID":
                    log_file = save_detailed_log(prompt, raw_output, output, self.model_name_or_path)
                    print(f"[WARNING] Invalid output detected. Log saved to: {log_file}")
                    print(f"[WARNING] Raw output was: {repr(raw_output)[:100]}")
                    # Default to first passage for invalid outputs to continue processing
                    output = self.CHARACTERS[0]

        elif self.scoring == 'likelihood':
            if self.config.model_type == 't5':
                input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
                self.total_prompt_tokens += input_ids.shape[1]
                with torch.no_grad():
                    logits = self.llm(input_ids=input_ids, decoder_input_ids=self.decoder_input_ids).logits[0][-1]
                    distributions = torch.softmax(logits, dim=0)
                    scores = distributions[self.target_token_ids[:len(docs)]]
                    ranked = sorted(zip(self.CHARACTERS[:len(docs)], scores), key=lambda x: x[1], reverse=True)
                    output = ranked[0][0]

            else:
                raise NotImplementedError

        if len(output) == 1 and output in self.CHARACTERS:
            pass
        else:
            print(f"Unexpected output: {output}")

        return output

    def heapify(self, arr, n, i, query, attack_prompt="none", attack_position="back"):
        # Find largest among root and children
        if self.num_child * i + 1 < n:  # if there are children
            docs = [arr[i]] + arr[self.num_child * i + 1: min((self.num_child * (i + 1) + 1), n)]
            inds = [i] + list(range(self.num_child * i + 1, min((self.num_child * (i + 1) + 1), n)))
            output = self.compare(query, docs, attack_prompt=attack_prompt, attack_position=attack_position)
            try:
                best_ind = self.CHARACTERS.index(output)
            except ValueError:
                best_ind = 0
            try:
                largest = inds[best_ind]
            except IndexError:
                largest = i
            # If root is not largest, swap with largest and continue heapifying
            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                self.heapify(arr, n, largest, query, attack_prompt=attack_prompt, attack_position=attack_position)

    def heapSort(self, arr, query, k, attack_prompt="none", attack_position="back"):
        n = len(arr)
        ranked = 0
        # Build max heap
        for i in range(n // self.num_child, -1, -1):
            self.heapify(arr, n, i, query, attack_prompt=attack_prompt, attack_position=attack_position)
        for i in range(n - 1, 0, -1):
            # Swap
            arr[i], arr[0] = arr[0], arr[i]
            ranked += 1
            if ranked == k:
                break
            # Heapify root element
            self.heapify(arr, i, 0, query, attack_prompt=attack_prompt, attack_position=attack_position)

    def rerank(self,  query: str, ranking: List[SearchResult], attack_prompt: str = "none", attack_position: str = "back") -> List[SearchResult]:
        original_ranking = copy.deepcopy(ranking)
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0
        
        if self.method == "heapsort":
            self.heapSort(ranking, query, self.k, attack_prompt=attack_prompt, attack_position=attack_position)
            ranking = list(reversed(ranking))
        elif self.method == "bubblesort":
            last_start = len(ranking) - (self.num_child + 1)

            for i in range(self.k):
                start_ind = last_start
                end_ind = last_start + (self.num_child + 1)
                is_change = False
                while True:
                    if start_ind < i:
                        start_ind = i
                    output = self.compare(query, ranking[start_ind:end_ind], attack_prompt=attack_prompt, attack_position=attack_position)
                    try:
                        best_ind = self.CHARACTERS.index(output)
                    except ValueError:
                        best_ind = 0
                    if best_ind != 0:
                        ranking[start_ind], ranking[start_ind + best_ind] = ranking[start_ind + best_ind], ranking[start_ind]
                        if not is_change:
                            is_change = True
                            if last_start != len(ranking) - (self.num_child + 1) \
                                    and best_ind == len(ranking[start_ind:end_ind])-1:
                                last_start += len(ranking[start_ind:end_ind])-1

                    if start_ind == i:
                        break

                    if not is_change:
                        last_start -= self.num_child

                    start_ind -= self.num_child
                    end_ind -= self.num_child
                    
        ##  this is a bit slower but standard bobblesort implementation, keep here FYI
        # elif self.method == "bubblesort":
        #     for i in range(k):
        #         start_ind = len(ranking) - (self.num_child + 1)
        #         end_ind = len(ranking)
        #         while True:
        #             if start_ind < i:
        #                 start_ind = i
        #             output = self.compare(query, ranking[start_ind:end_ind])
        #             try:
        #                 best_ind = self.CHARACTERS.index(output)
        #             except ValueError:
        #                 best_ind = 0
        #             if best_ind != 0:
        #                 ranking[start_ind], ranking[start_ind + best_ind] = ranking[start_ind + best_ind], ranking[start_ind]
        #
        #             if start_ind == i:
        #                 break
        #
        #             start_ind -= self.num_child
        #             end_ind -= self.num_child

        else:
            raise NotImplementedError(f'Method {self.method} is not implemented.')

        results = []
        top_doc_ids = set()
        rank = 1

        for i, doc in enumerate(ranking[:self.k]):
            top_doc_ids.add(doc.docid)
            results.append(SearchResult(docid=doc.docid, score=-rank, text=None))
            rank += 1
        for doc in original_ranking:
            if doc.docid not in top_doc_ids:
                results.append(SearchResult(docid=doc.docid, score=-rank, text=None))
                rank += 1

        return results

    def truncate(self, text, length):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(text)[:length])


class OpenAiSetwiseLlmRanker(SetwiseLlmRanker):
    def __init__(self, model_name_or_path, api_key, num_child=3, method='heapsort', k=10):
        import tiktoken

        self.llm = model_name_or_path
        self.tokenizer = tiktoken.encoding_for_model(model_name_or_path)
        self.num_child = num_child
        self.method = method
        self.k = k
        self.total_compare = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.system_prompt = "You are RankGPT, an intelligent assistant specialized in selecting the most relevant passage from a pool of passages based on their relevance to the query."
        key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=key)

    def compare(self, query: str, docs: List, attack_prompt: str="none", attack_position: str="back"):
        self.total_compare += 1

        # 校验参数
        if attack_prompt not in ("none", "so", "sd"):
            raise ValueError(f"Unknown attack prompt: {attack_prompt}")
        if attack_position not in ("front", "back"):
            raise ValueError(f"Unknown attack position: {attack_position}")
        
        passage_lines = []
        for i, doc in enumerate(docs):
            gt_rel = getattr(doc, "gt_rel", None)
            text = doc.text

            if gt_rel == 0 and attack_prompt != "none":
                if attack_prompt == "so":
                    if attack_position == "front":
                        text = JAILBREAK_PROMPTS["so"] + " " + text
                    else:
                        text = text + " " + JAILBREAK_PROMPTS["so"]
                else:
                    if attack_position == "front":
                        text = JAILBREAK_PROMPTS["sd"] + " " + text
                    else:
                        text = text + " " + JAILBREAK_PROMPTS["sd"]
            passage_lines.append(f'Passage {self.CHARACTERS[i]}: "{text}"')

        passages = "\n\n".join(passage_lines)     


        input_text = f'Given a query "{query}", which of the following passages is the most relevant one to the query?\n\n' \
                     + passages + '\n\nOutput only the passage label of the most relevant passage.'

        while True:
            try:
                response = self.client.responses.create(
                    model=self.llm,
                    instructions=self.system_prompt,
                    input=input_text,
                    reasoning={"effort": "low"},
                )

                if getattr(response, "usage", None):
                    self.total_prompt_tokens += int(getattr(response.usage, "input_tokens", 0) or 0)
                    self.total_completion_tokens += int(getattr(response.usage, "output_tokens", 0) or 0)

                output = response.output_text or ""

                matches = re.findall(r"(Passage [A-Z])", output, re.MULTILINE)
                if matches:
                    output = matches[0][8]
                elif output.strip() in self.CHARACTERS:
                    pass
                else:
                    print(f"Unexpected output: {output}")
                    output = "A"
                return output

            except openai.error.APIError as e:
                # Handle API error here, e.g. retry or log
                print(f"OpenAI API returned an API Error: {e}")
                time.sleep(5)
                continue
            except openai.error.APIConnectionError as e:
                # Handle connection error here
                print(f"Failed to connect to OpenAI API: {e}")
                time.sleep(5)
                continue
            except openai.error.RateLimitError as e:
                # Handle rate limit error (we recommend using exponential backoff)
                print(f"OpenAI API request exceeded rate limit: {e}")
                time.sleep(5)
                continue
            except openai.error.InvalidRequestError as e:
                # Handle invalid request error
                print(f"OpenAI API request was invalid: {e}")
                raise e
            except openai.error.AuthenticationError as e:
                # Handle authentication error
                print(f"OpenAI API request failed authentication: {e}")
                raise e
            except openai.error.Timeout as e:
                # Handle timeout error
                print(f"OpenAI API request timed out: {e}")
                time.sleep(5)
                continue
            except openai.error.ServiceUnavailableError as e:
                # Handle service unavailable error
                print(f"OpenAI API request failed with a service unavailable error: {e}")
                time.sleep(5)
                continue
            except Exception as e:
                print(f"Unknown error: {e}")
                raise e

    def truncate(self, text, length):
        return self.tokenizer.decode(self.tokenizer.encode(text)[:length])


class RankR1SetwiseLlmRanker(SetwiseLlmRanker):
    CHARACTERS = [f'[{i+1}]' for i in range(20)]

    def __init__(self,
                 model_name_or_path,
                 prompt_file,
                 lora_name_or_path=None,
                 tokenizer_name_or_path=None,
                 num_child=19,
                 k=10,
                 scoring='generation',
                 method="heapsort",
                 num_permutation=1,
                 cache_dir=None,
                 verbose=False):

        if scoring != 'generation':
            raise NotImplementedError(f"Scoring method {scoring} is not supported for RankR1SetwiseLlmRanker. RankR1SetwiseLlmRanker only supports 'generation' scoring.")
        self.verbose = verbose

        import toml
        self.prompt = toml.load(prompt_file)

        from huggingface_hub import snapshot_download
        if lora_name_or_path is not None:
            # check if the path exists
            if not os.path.exists(lora_name_or_path):
                # download the model
                lora_path = snapshot_download(lora_name_or_path)
            else:
                lora_path = lora_name_or_path
        else:
            lora_path = None

        self.lora_path = lora_path
        self.num_child = num_child
        self.num_permutation = num_permutation
        self.k = k
        self.sampling_params = SamplingParams(temperature=0.0,
                                              max_tokens=2048)
        if tokenizer_name_or_path is None:
            tokenizer_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, cache_dir=cache_dir)
        self.llm = LLM(model=model_name_or_path,
                       tokenizer=tokenizer_name_or_path,
                       enable_lora=True if lora_name_or_path is not None else False,
                       max_lora_rank=32,
                       )

        self.scoring = scoring
        self.method = method
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0

    def compare(self, query: str, docs: List):
        self.total_compare += 1 if self.num_permutation == 1 else self.num_permutation

        id_passage = [(i, p) for i, p in enumerate(docs)]
        labels = [self.CHARACTERS[i] for i in range(len(docs))]
        batch_data = []
        for _ in range(self.num_permutation):
            batch_data.append([random.sample(id_passage, len(id_passage)),
                               labels])

        batch_ref = []
        input_text = []
        for batch in batch_data:
            ref = []
            passages = []
            characters = []
            for p, c in zip(batch[0], batch[1]):
                ref.append(p[0])
                passages.append(p[1].text)
                characters.append(c)
            batch_ref.append((ref, characters))
            passages = "\n".join([f'{characters[i]} {passages[i]}' for i in range(len(passages))])
            system_message = self.prompt["prompt_system"]
            user_message = self.prompt['prompt_user'].format(query=query,
                                                             docs=passages)
            input_text.append([
                {'role': "system", 'content': system_message},
                {'role': "user", 'content': user_message}
            ])
        outputs = self.llm.chat(input_text,
                                sampling_params=self.sampling_params,
                                use_tqdm=False,
                                lora_request=LoRARequest("R1adapter",
                                                         1,
                                                         self.lora_path)
                                if self.lora_path is not None else None,
                                )
        results = []
        for output, input in zip(outputs, input_text):
            self.total_completion_tokens += len(output.outputs[0].token_ids)
            self.total_prompt_tokens += len(output.prompt_token_ids)

            completion = output.outputs[0].text

            if self.verbose:
                print('--------------------------------------')
                print(f'query: {query}')
                print(f'input_text:\n{self.tokenizer.apply_chat_template(input, tokenize=False)}')
                print(f'completion:\n{completion}')
                print('--------------------------------------')

            pattern = rf'{self.prompt["pattern"]}'
            match = re.search(pattern, completion.lower(), re.DOTALL)
            if match:
                results.append(match.group(1).strip())
            else:
                results.append(f'input_text:\n{input}, completion:\n{completion}')

        # vote
        candidates = []
        for ref, result in zip(batch_ref, results):
            result = result.strip()
            docids, characters = ref
            if result not in characters:
                if self.verbose:
                    print(f"Unexpected output: {result}")
                continue
            win_doc = docids[characters.index(result)]
            candidates.append(win_doc)

        if len(candidates) == 0:
            if self.verbose:
                print(f"Unexpected voting: {results}")
            output = "Unexpected voting."
        else:
            # handle tie
            candidate_counts = Counter(candidates)
            max_count = max(candidate_counts.values())
            most_common_candidates = [candidate for candidate, count in candidate_counts.items() if
                                      count == max_count]
            if len(most_common_candidates) == 1:
                output = self.CHARACTERS[most_common_candidates[0]]
            else:
                output = self.CHARACTERS[random.choice(most_common_candidates)]

        if output in self.CHARACTERS:
            pass
        else:
            if self.verbose:
                print(f"Unexpected output: {output}")

        return output
