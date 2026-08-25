import re
import sys

p = sys.argv[1]
src = open(p).read()
if "__GDNT_" in src:
    print("already instrumented")
    raise SystemExit(0)

anchor = "logger = init_logger(__name__)"
assert anchor in src
acc = anchor + '''

# ---- GDN layer-0 wall-clock timing (temporary) ----
import time as __gdnt_time
__GDNT_ACC = {"n": 0, "conv": 0.0, "gate": 0.0, "rec": 0.0, "tail": 0.0, "tot": 0.0}


def __gdnt_report():
    a = __GDNT_ACC
    n = a["n"]
    if n > 0 and n % 10 == 0:
        print(f"[GDNT n={n}] conv={a['conv']/n:.2f} gate={a['gate']/n:.2f} rec={a['rec']/n:.2f} tail={a['tail']/n:.2f} TOTAL={a['tot']/n:.2f} ms (layer0 spec)", flush=True)
# ---- end instrumentation ----
'''
src = src.replace(anchor, acc, 1)

m = "assert isinstance(attn_metadata, dict)\n        attn_metadata = attn_metadata[self.prefix]"
assert m in src
src = src.replace(
    m,
    m + "\n        __gdnt_do = \".layers.0.\" in self.prefix and attn_metadata.spec_sequence_masks is not None\n"
    "        if __gdnt_do:\n            __gdnt_t0 = __gdnt_time.time()",
    1,
)

m1 = "        # 1.1: Process the multi-query part"
assert m1 in src
src = src.replace(
    m1,
    "        if __gdnt_do:\n            torch.cuda.synchronize()\n            __gdnt_t1 = __gdnt_time.time()\n" + m1,
    1,
)

m2 = "        beta = b.sigmoid()"
assert m2 in src
src = src.replace(
    m2,
    "        if __gdnt_do:\n            torch.cuda.synchronize()\n            __gdnt_t2 = __gdnt_time.time()\n" + m2,
    1,
)

m3 = "        # 2.1: Process the multi-query part"
assert m3 in src
src = src.replace(
    m3,
    "        if __gdnt_do:\n            torch.cuda.synchronize()\n            __gdnt_t3 = __gdnt_time.time()\n" + m3,
    1,
)

m4 = "        # 3. Merge core attention output"
assert m4 in src
src = src.replace(
    m4,
    "        if __gdnt_do:\n            torch.cuda.synchronize()\n            __gdnt_t4 = __gdnt_time.time()\n" + m4,
    1,
)

tail = "        else:\n            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)\n"
assert tail in src
rep = tail + '''
        if __gdnt_do:
            torch.cuda.synchronize()
            __gdnt_t5 = __gdnt_time.time()
            __GDNT_ACC["n"] += 1
            __GDNT_ACC["conv"] += (__gdnt_t2 - __gdnt_t1) * 1000
            __GDNT_ACC["gate"] += (__gdnt_t3 - __gdnt_t2) * 1000
            __GDNT_ACC["rec"] += (__gdnt_t4 - __gdnt_t3) * 1000
            __GDNT_ACC["tail"] += (__gdnt_t5 - __gdnt_t4) * 1000
            __GDNT_ACC["tot"] += (__gdnt_t5 - __gdnt_t0) * 1000
            __gdnt_report()
'''
src = src.replace(tail, rep, 1)

open(p + ".instr", "w").write(src)
print("written", p + ".instr")
