# GLM-Air-4.5

* vLLM Version: vLLM: 0.10.1.1 , vLLM-KunLun Version: v0.10.1.1
* Software Environment:OS: Ubuntu 22.04, PyTorch ≥ 2.5.1
* Hardware Environment: KunLun P800
* Parallel mode:TP8

```bash
+-------------+----------+---------------+---------+-----+--------+---------+
| Model       | Dataset  | Metric        | Subset  | Num | Score  | Cat.0   |
+-------------+----------+---------------+---------+-----+--------+---------+
| GLM-4.5-Air | math_500 | AveragePass@1 | Level 1 | 43  | 0.9302 | default |
| GLM-4.5-Air | math_500 | AveragePass@1 | Level 2 | 90  | 0.9222 | default |
| GLM-4.5-Air | math_500 | AveragePass@1 | Level 3 | 105 | 0.8762 | default |
| GLM-4.5-Air | math_500 | AveragePass@1 | Level 4 | 128 | 0.8984 | default |
| GLM-4.5-Air | math_500 | AveragePass@1 | Level 5 | 134 | 0.8955 | default |
+-------------+----------+---------------+---------+-----+--------+---------+
```
