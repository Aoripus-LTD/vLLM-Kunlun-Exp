import sys

p = sys.argv[1]
src = open(p).read()
if "_GDN_ACC" in src:
    print("already instrumented")
    raise SystemExit(0)

anchor = "logger = init_logger(__name__)"
assert anchor in src, "logger anchor missing"
acc = anchor + '''

# ---- GDN timing instrumentation (temporary) ----
_GDN_ACC = {"n": 0, "conv_s": 0.0, "conv_ns": 0.0, "gate": 0.0, "rec_s": 0.0, "rec_ns": 0.0, "merge": 0.0, "tot": 0.0}


def _gdn_timer_report():
    a = _GDN_ACC
    n = a["n"]
    if n > 0 and n % 240 == 0:
        print(f"[GDN_TIMER n={n}] conv_s={a['conv_s']/n:.2f} conv_ns={a['conv_ns']/n:.2f} gate={a['gate']/n:.2f} rec_s={a['rec_s']/n:.2f} rec_ns={a['rec_ns']/n:.2f} merge={a['merge']/n:.2f} TOTAL={a['tot']/n:.2f} ms/layer", flush=True)
# ---- end instrumentation ----
'''
src = src.replace(anchor, acc, 1)

m = "assert isinstance(attn_metadata, dict)\n        attn_metadata = attn_metadata[self.prefix]"
assert m in src, "attn_metadata anchor missing"
src = src.replace(m, m + "\n        _t0 = torch.xpu.Event(enable_timing=True); _t0.record()", 1)

m1 = "        # 1.1: Process the multi-query part"
assert m1 in src, "m1 missing"
src = src.replace(m1, "        _t1 = torch.xpu.Event(enable_timing=True); _t1.record()\n" + m1, 1)

m2 = "        # 1.2: Process the remaining part"
assert m2 in src, "m2 missing"
src = src.replace(m2, "        _t2 = torch.xpu.Event(enable_timing=True); _t2.record()\n" + m2, 1)

m3 = "        beta = b.sigmoid()"
assert m3 in src, "m3 missing"
src = src.replace(m3, "        _t3 = torch.xpu.Event(enable_timing=True); _t3.record()\n" + m3, 1)

m4 = "        # 2.1: Process the multi-query part"
assert m4 in src, "m4 missing"
src = src.replace(m4, "        _t4 = torch.xpu.Event(enable_timing=True); _t4.record()\n" + m4, 1)

m5 = "        # 2.2: Process the remaining part"
assert m5 in src, "m5 missing"
src = src.replace(m5, "        _t5 = torch.xpu.Event(enable_timing=True); _t5.record()\n" + m5, 1)

m6 = "        # 3. Merge core attention output"
assert m6 in src, "m6 missing"
src = src.replace(m6, "        _t6 = torch.xpu.Event(enable_timing=True); _t6.record()\n" + m6, 1)

tail = "        else:\n            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)\n"
assert tail in src, "tail missing"
rep = tail + '''
        if spec_sequence_masks is not None:
            _t7 = torch.xpu.Event(enable_timing=True); _t7.record()
            torch.xpu.synchronize()
            _GDN_ACC["n"] += 1
            _GDN_ACC["conv_s"] += _t0.elapsed_time(_t1)
            _GDN_ACC["conv_ns"] += _t1.elapsed_time(_t2)
            _GDN_ACC["gate"] += _t2.elapsed_time(_t3)
            _GDN_ACC["rec_s"] += _t3.elapsed_time(_t4)
            _GDN_ACC["rec_ns"] += _t4.elapsed_time(_t5)
            _GDN_ACC["merge"] += _t5.elapsed_time(_t6)
            _GDN_ACC["tot"] += _t0.elapsed_time(_t6)
            _gdn_timer_report()
'''
src = src.replace(tail, rep, 1)

open(p + ".instr", "w").write(src)
print("written", p + ".instr")
