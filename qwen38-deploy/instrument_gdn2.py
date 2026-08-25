import re
import sys

p = sys.argv[1]
src = open(p).read()
if "_GDN_T" in src:
    print("already instrumented")
    raise SystemExit(0)

anchor = "logger = init_logger(__name__)"
assert anchor in src
acc = anchor + '''

# ---- GDN layer-0 wall-clock timing (temporary) ----
import time as _gdn_time
_GDN_T = {"n": 0, "conv": 0.0, "gate": 0.0, "rec": 0.0, "tail": 0.0, "tot": 0.0}


def _gdn_t_report():
    a = _GDN_T
    n = a["n"]
    if n > 0 and n % 10 == 0:
        print(f"[GDN_T n={n}] conv={a['conv']/n:.2f} gate={a['gate']/n:.2f} rec={a['rec']/n:.2f} tail={a['tail']/n:.2f} TOTAL={a['tot']/n:.2f} ms (layer0, spec)", flush=True)
# ---- end instrumentation ----
'''
src = src.replace(anchor, acc, 1)

m = "assert isinstance(attn_metadata, dict)\n        attn_metadata = attn_metadata[self.prefix]"
assert m in src
src = src.replace(
    m,
    m + "\n        _gdn_do = attn_metadata.spec_sequence_masks is not None and self.prefix.endswith('.layers.0')\n"
    "        if _gdn_do:\n            _gdn_t0 = _gdn_time.time()",
    1,
)

# conv boundary: before "# 1.1"
m1 = "        # 1.1: Process the multi-query part"
assert m1 in src
src = src.replace(
    m1,
    "        if _gdn_do:\n            torch.cuda.synchronize()\n            _gdn_t1 = _gdn_time.time()\n" + m1,
    1,
)

# gate boundary: before "beta = b.sigmoid()"
m2 = "        beta = b.sigmoid()"
assert m2 in src
src = src.replace(
    m2,
    "        if _gdn_do:\n            torch.cuda.synchronize()\n            _gdn_t2 = _gdn_time.time()\n" + m2,
    1,
)

# recurrent boundary: before "# 2.1"
m3 = "        # 2.1: Process the multi-query part"
assert m3 in src
src = src.replace(
    m3,
    "        if _gdn_do:\n            torch.cuda.synchronize()\n            _gdn_t3 = _gdn_time.time()\n" + m3,
    1,
)

# after recurrent: before "# 3. Merge"
m4 = "        # 3. Merge core attention output"
assert m4 in src
src = src.replace(
    m4,
    "        if _gdn_do:\n            torch.cuda.synchronize()\n            _gdn_t4 = _gdn_time.time()\n" + m4,
    1,
)

tail = "        else:\n            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)\n"
assert tail in src
rep = tail + '''
        if _gdn_do:
            torch.cuda.synchronize()
            _gdn_t5 = _gdn_time.time()
            _GDN_T["n"] += 1
            _GDN_T["conv"] += (_gdn_t2 - _gdn_t1) * 1000
            _GDN_T["gate"] += (_gdn_t3 - _gdn_t2) * 1000
            _GDN_T["rec"] += (_gdn_t4 - _gdn_t3) * 1000
            _GDN_T["tail"] += (_gdn_t5 - _gdn_t4) * 1000
            _GDN_T["tot"] += (_gdn_t5 - _gdn_t0) * 1000
            _gdn_t_report()
'''
src = src.replace(tail, rep, 1)

open(p + ".instr", "w").write(src)
print("written", p + ".instr")
