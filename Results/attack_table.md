# TREC-DL attack reproduction table

Only `back`-position runs are included. `Ours` uses the standard evaluator prompt; `Defense` uses the defense evaluator prompt. Blank cells have no recorded result.

| Dataset | Model | Pairwise Flipped %<br>DOH Orig | Pairwise Flipped %<br>DOH Ours | Pairwise Flipped %<br>DOH Defense | Pairwise Flipped %<br>DCH Orig | Pairwise Flipped %<br>DCH Ours | Pairwise Flipped %<br>DCH Defense | Setwise Attack Success %<br>DOH Orig | Setwise Attack Success %<br>DOH Ours | Setwise Attack Success %<br>DOH Defense | Setwise Attack Success %<br>DCH Orig | Setwise Attack Success %<br>DCH Ours | Setwise Attack Success %<br>DCH Defense | Listwise Attack Top Position %<br>DOH Orig | Listwise Attack Top Position %<br>DOH Ours | Listwise Attack Top Position %<br>DOH Defense | Listwise Attack Top Position %<br>DCH Orig | Listwise Attack Top Position %<br>DCH Ours | Listwise Attack Top Position %<br>DCH Defense |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TREC-DL-2019 | Qwen3-1.7B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| TREC-DL-2019 | Qwen3-4B |  | 81.74%<br>(3348/4096) | 69.95%<br>(2865/4096) |  | 76.37%<br>(3128/4096) |  |  | 81.25%<br>(13/16) | 73.60%<br>(3013/4094) |  |  |  |  | 32.15%<br>(1316/4093) |  |  |  |  |
| TREC-DL-2019 | Qwen3-8B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| TREC-DL-2019 | Qwen3-14B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| TREC-DL-2019 | Qwen3-32B |  | 95.90%<br>(3928/4096) | 29.03%<br>(1189/4096) |  | 96.97%<br>(3972/4096) | 16.19%<br>(663/4096) |  | 90.10%<br>(3658/4060) | 79.74%<br>(3266/4096) |  |  |  |  |  |  |  |  |  |
| TREC-DL-2019 | Gemma-3-12B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| TREC-DL-2019 | Gemma-3-27B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| TREC-DL-2019 | Llama-3-8B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| TREC-DL-2019 | Llama-3.3-70B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| TREC-DL-2019 | GPT-4.1-mini |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| TREC-DL-2019 | GPT-OSS-20B |  | 100.00%<br>(4060/4060) | 33.37%<br>(1300/3896) |  |  |  |  | 98.99%<br>(98/99) |  |  |  |  |  |  |  |  |  |  |
| TREC-DL-2019 | gpt-5.6-terra |  |  |  |  |  |  |  | 8.00%<br>(8/100) |  |  |  |  |  |  |  |  |  |  |
| TREC-DL-2019 | **Mean ± Std** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| TREC-DL-2020 | Qwen3-1.7B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| TREC-DL-2020 | Qwen3-4B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| TREC-DL-2020 | Qwen3-8B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| TREC-DL-2020 | Qwen3-14B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| TREC-DL-2020 | Qwen3-32B |  | 92.11%<br>(3773/4096) | 28.25%<br>(1157/4096) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| TREC-DL-2020 | Gemma-3-12B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| TREC-DL-2020 | Gemma-3-27B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| TREC-DL-2020 | Llama-3-8B |  | 60.11%<br>(2462/4096) | 56.59%<br>(2318/4096) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| TREC-DL-2020 | Llama-3.3-70B |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| TREC-DL-2020 | GPT-4.1-mini |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| TREC-DL-2020 | GPT-OSS-20B |  | 100.00%<br>(4080/4080) | 37.09%<br>(1433/3864) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| TREC-DL-2020 | meta.llama3-70b-instruct-v1:0 |  | 97.80%<br>(4006/4096) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| TREC-DL-2020 | **Mean ± Std** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
