# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Baidu, Inc. and/or its affiliates
"""Ordinary DP2 request concurrency patches for vLLM-Omni on Kunlun.

vLLM-Omni (0.26.0 and 0.28.0rc1) only enables ``dp_concurrent`` request
batching when *distributed layerwise offload* (DLO) is active.  On Kunlun we
serve MiniMax H3 with ordinary data parallelism (two TP4 replicas on eight
XPUs, no DLO), so the engine refuses to schedule more than one request and
the executor routes multi-request waves to the fused request-batch RPC that
MiniMax H3 does not support.

These patches generalise the existing (DLO-only) DP machinery to ordinary
data parallelism:

* ``DiffusionEngine``: treat ``data_parallel_size > 1`` as DP-concurrent so
  ``max_num_running_reqs`` is raised to ``dp_size`` and the request scheduler
  coalesces up to ``dp_size`` requests per wave.
* ``MultiprocDiffusionExecutor``: route multi-request waves through the
  per-replica ``execute_model`` broadcast (each DP rank picks the envelope for
  its replica) instead of the fused ``execute_model_batch`` RPC.
* ``MiniMaxH3Pipeline``: scope the DiT-level broadcasts and the text-encoder
  process group to the tensor-parallel replica instead of the whole default
  world, so each DP replica tokenises and encodes its own prompt.

Every patch is idempotent and failure-tolerant: if the installed vLLM-Omni
layout does not match, the patch is skipped and the platform keeps working.
"""

from __future__ import annotations

import logging
import os
import time

_KUNLUN_DP2_PATCHES_APPLIED = False

_logger = logging.getLogger("vllm_kunlun.omni.dp2")


def _kunlun_resolve_dp_size(parallel_config) -> int:
    dp_size = getattr(parallel_config, "data_parallel_size", None)
    try:
        dp_size = int(dp_size) if dp_size is not None else 1
    except (TypeError, ValueError):
        dp_size = 1
    if dp_size > 1:
        return dp_size
    # Last-resort inference from the resolved world layout.
    try:
        import torch.distributed as _dist

        tp_size = int(getattr(parallel_config, "tensor_parallel_size", 1) or 1)
        if _dist.is_available() and _dist.is_initialized() and tp_size > 0:
            dp_size = int(_dist.get_world_size()) // tp_size
    except Exception:
        dp_size = 1
    return max(1, dp_size)


def _patch_parallel_config_dp_env_override() -> None:
    """Inject ``VLLM_KUNLUN_DP_SIZE`` into DiffusionParallelConfig.

    vLLM-Omni 0.26.0 has no ``--data-parallel-size`` CLI flag and does not
    propagate the deploy-YAML ``data_parallel_size`` into the engine's
    ``DiffusionParallelConfig``.  Wrapping the model constructor covers both
    the YAML dict branch (``from_dict`` -> ``cls(**data)``) and the CLI kwargs
    branch (direct construction), so DP2 layouts can be requested purely via
    the environment.
    """
    try:
        from vllm_omni.diffusion.data import DiffusionParallelConfig as _DPC
    except Exception:
        return
    if getattr(_DPC, "_kunlun_dp2_env_patched", False):
        return

    _orig_init = _DPC.__init__

    def _kunlun_dpc_init(self, **data):
        try:
            _env_dp = os.environ.get("VLLM_KUNLUN_DP_SIZE")
            if _env_dp:
                data["data_parallel_size"] = int(_env_dp)
        except Exception:
            pass
        _orig_init(self, **data)

    _DPC.__init__ = _kunlun_dpc_init
    _DPC._kunlun_dp2_env_patched = True
    _logger.info("DP2: DiffusionParallelConfig VLLM_KUNLUN_DP_SIZE override installed")


def _patch_executor_dp_rpc_routing() -> None:
    """Route DP multi-reply waves through the sync wave-collection path.

    Request-mode ``execute_model`` / ``execute_model_batch`` normally go
    through the ``rpc_id`` future pump (async output path).  That pump only
    resolves futures for ``AsyncDiffusionOutput`` messages, but a DP wave
    worker reply is a plain dict envelope (``{"dp_rank": ..., "output": ...}``
    produced by ``DiffusionWorker.execute_model`` for batch input), so the
    future would hang until timeout.  Re-route DP waves
    (``unique_reply_rank=None`` + ``exec_all_ranks=True``) through the
    synchronous collection loop that understands those dict replies.
    """
    try:
        import vllm_omni.diffusion.executor.multiproc_executor as _exec_mod
    except Exception:
        return
    if getattr(_exec_mod, "_kunlun_dp2_rpc_patched", False):
        return

    from vllm_omni.diffusion.ipc import unpack_diffusion_output_shm

    _exec_logger = _exec_mod.logger
    _orig_collective_rpc = _exec_mod.MultiprocDiffusionExecutor.collective_rpc

    def _kunlun_collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
        exec_all_ranks: bool = False,
    ):
        kwargs = kwargs or {}
        is_dp_multi_reply = (
            method in ("execute_model", "execute_model_batch")
            and unique_reply_rank is None
            and exec_all_ranks
            and not self.od_config.step_execution
        )
        if not is_dp_multi_reply:
            return _orig_collective_rpc(
                self,
                method,
                timeout=timeout,
                args=args,
                kwargs=kwargs,
                unique_reply_rank=unique_reply_rank,
                exec_all_ranks=exec_all_ranks,
            )

        # ── Sync wave collection (mirrors upstream Path 2) ──
        self._ensure_open()
        deadline = None if timeout is None else time.monotonic() + timeout
        self._rpc_wave_id += 1
        wave_id = self._rpc_wave_id
        rpc_request = {
            "type": "rpc",
            "method": method,
            "args": args,
            "kwargs": kwargs,
            "output_rank": None,
            "exec_all_ranks": True,
            "collect_rank_status": True,
            "wave_id": wave_id,
        }

        try:
            self._broadcast_mq.enqueue(rpc_request)

            dp_size = _kunlun_resolve_dp_size(
                getattr(self.od_config, "parallel_config", None)
            )
            num_responses = max(1, dp_size)
            tagged: list[tuple[int, object]] = []
            collected_errors: list[str] = []
            for _ in range(num_responses):
                response = self._dequeue_one_with_failure_polling(deadline, method)
                response = self._validate_wave_id(response, wave_id, deadline, method)
                try:
                    unpack_diffusion_output_shm(response)
                except Exception as exc:
                    _exec_logger.warning(
                        "SHM unpack failed (data may already be inline): %s", exc
                    )
                if isinstance(response, dict) and response.get("status") == "error":
                    collected_errors.append(str(response.get("error", "unknown")))
                else:
                    response = (
                        _exec_mod.MultiprocDiffusionExecutor._handle_rpc_response(
                            response
                        )
                    )
                    if isinstance(response, dict) and "dp_rank" in response:
                        tagged.append((response["dp_rank"], response["output"]))
                    else:
                        tagged.append((len(tagged), response))
            if collected_errors:
                raise RuntimeError(f"Worker error: {collected_errors[0]}")
            tagged.sort(key=lambda item: item[0])
            return [payload for _, payload in tagged]
        except Exception as exc:
            _exec_logger.error("RPC call failed: %s", exc)
            raise

    _exec_mod.MultiprocDiffusionExecutor.collective_rpc = _kunlun_collective_rpc
    _exec_mod._kunlun_dp2_rpc_patched = True
    _logger.info("DP2: MultiprocDiffusionExecutor collective_rpc DP-wave routing installed")


def _patch_engine_dp_concurrency() -> None:
    """Enable dp_concurrent for ordinary (non-DLO) data parallelism."""
    try:
        import vllm_omni.diffusion.diffusion_engine as _engine_mod
    except Exception:
        return
    if getattr(_engine_mod, "_kunlun_dp2_patched", False):
        return

    _engine_logger = _engine_mod.logger

    # 0.28.0rc1 routes several decisions through this helper; replace it when
    # present so the resolution-mode guard and KV-profile path also see DP>1.
    if hasattr(_engine_mod, "_uses_dlo_dp_concurrency"):

        def _kunlun_uses_dlo_dp_concurrency(od_config) -> bool:
            parallel_config = getattr(od_config, "parallel_config", None)
            return _kunlun_resolve_dp_size(parallel_config) > 1

        _engine_mod._uses_dlo_dp_concurrency = _kunlun_uses_dlo_dp_concurrency

    def _kunlun_init_runtime_state(self) -> None:
        import asyncio as _asyncio
        import queue as _queue
        import threading as _threading

        parallel_config = getattr(self.od_config, "parallel_config", None)
        dp_size = _kunlun_resolve_dp_size(parallel_config)
        if dp_size > 1:
            self.scheduler.max_num_running_reqs = dp_size
            self.dp_concurrent = True
            # 0.26.0: the scheduler reads this from od_config; make sure
            # concurrent requests actually accumulate before scheduling.
            if float(getattr(self.od_config, "request_batch_max_wait_ms", 0) or 0) == 0:
                self.od_config.request_batch_max_wait_ms = 500.0
            _engine_logger.info(
                "dp_concurrent: max_num_running_reqs=%d, batch_wait=%sms",
                dp_size,
                self.od_config.request_batch_max_wait_ms,
            )
        else:
            self.dp_concurrent = False
        self.main_loop = None
        self.stop_event = None
        self.worker_thread = None
        self._loop_started = False
        self._init_lock = _asyncio.Lock()
        self._rpc_lock = _threading.RLock()
        self._cv = _threading.Condition(self._rpc_lock)
        self._out_streams: dict = {}
        self._closed = False
        self._shutdown_complete = False
        self.abort_queue = _queue.Queue()
        self._rpc_queue = _queue.Queue()
        # 0.28.0rc1 keeps this metric; harmless on 0.26.0.
        self._scheduler_num_waiting_reqs = 0

    _engine_mod.DiffusionEngine._init_runtime_state = _kunlun_init_runtime_state
    _engine_mod._kunlun_dp2_patched = True


def _patch_executor_dp_result_queues() -> None:
    """Backport vLLM-Omni 0.28's per-worker result queues to 0.26.0.

    In 0.26.0 the executor only creates a result-queue reader for worker 0,
    so replies from replica 1+ primary ranks never reach the engine and a DP
    wave hangs waiting for them.  vLLM-Omni 0.28.0rc1 collects every worker's
    ``result_handle`` and runs one pump thread per queue; backport that here.
    """
    try:
        import vllm_omni.diffusion.executor.multiproc_executor as _exec_mod
    except Exception:
        return
    if getattr(_exec_mod, "_kunlun_dp2_queues_patched", False):
        return

    _MDE = _exec_mod.MultiprocDiffusionExecutor
    _exec_logger = _exec_mod.logger

    # ── 1. _launch_workers: return every worker's result_handle ──
    def _kunlun_launch_workers(self, broadcast_handle, wake_events):
        od_config = self.od_config
        _exec_logger.info("Starting server...")
        num_gpus = int(od_config.num_gpus)
        _exec_mod.mp.set_start_method("spawn", force=True)
        processes = []
        worker_extension_cls = od_config.worker_extension_cls
        custom_pipeline_args = getattr(od_config, "custom_pipeline_args", None)
        scheduler_pipe_readers = []
        scheduler_pipe_writers = []
        for i in range(num_gpus):
            reader, writer = _exec_mod.mp.Pipe(duplex=False)
            scheduler_pipe_writers.append(writer)
            process = _exec_mod.mp.Process(
                target=_exec_mod.WorkerProc.worker_main,
                args=(
                    i,
                    od_config,
                    writer,
                    broadcast_handle,
                    wake_events[i],
                    worker_extension_cls,
                    custom_pipeline_args,
                ),
                name=f"DiffusionWorker-{i}",
                daemon=True,
            )
            scheduler_pipe_readers.append(reader)
            process.start()
            processes.append(process)

        result_handles = []
        for writer in scheduler_pipe_writers:
            writer.close()

        for i, reader in enumerate(scheduler_pipe_readers):
            try:
                data = reader.recv()
            except EOFError:
                _exec_logger.error(
                    "Rank %i scheduler is dead. Please check if there are relevant logs.",
                    i,
                )
                processes[i].join()
                _exec_logger.error("Exit code: %s", processes[i].exitcode)
                raise
            if data["status"] != "ready":
                raise RuntimeError(
                    "Initialization failed. Please see the error messages above."
                )
            result_handles.append(data.get("result_handle"))
            reader.close()

        _exec_logger.debug("All workers are ready")
        return processes, result_handles

    # ── 2. _init_result_queue: accept a list of handles ──
    _orig_init_result_queue = _MDE._init_result_queue

    def _kunlun_init_result_queue(self, result_handle):
        if isinstance(result_handle, list):
            queues = [_orig_init_result_queue(self, handle) for handle in result_handle]
            self._result_mqs = queues
            return queues[0] if queues else None
        queue = _orig_init_result_queue(self, result_handle)
        self._result_mqs = [queue]
        return queue

    # ── 3. result pump: one thread per queue ──
    def _kunlun_result_pump(self, result_mq=None) -> None:
        result_mq = result_mq if result_mq is not None else self._result_mq
        while not self._pump_stop.is_set():
            try:
                msg = result_mq.dequeue(timeout=1.0)
            except TimeoutError:
                if self._is_failed:
                    break
                continue
            except Exception:
                _exec_logger.exception("Result pump dequeue failed")
                if self._is_failed:
                    break
                continue

            if not isinstance(msg, _exec_mod.AsyncDiffusionOutput):
                self._sync_result_buffer.put(msg)
                continue

            if msg.kind in (
                _exec_mod.AsyncOutputKind.RPC_RESULT,
                _exec_mod.AsyncOutputKind.COMPUTE_DONE,
            ):
                with self._futures_lock:
                    fut = self._rpc_futures.pop(msg.rpc_id, None) if msg.rpc_id else None
                if fut is not None and not fut.done():
                    if msg.error:
                        fut.set_exception(RuntimeError(msg.error))
                    else:
                        fut.set_result(msg)
            elif msg.kind == _exec_mod.AsyncOutputKind.OUTPUT_READY:
                batch_id = msg.async_output_id
                with self._futures_lock:
                    per_req_map = self._batch_split_map.pop(batch_id, None) if batch_id else None
                if per_req_map is not None:
                    try:
                        _exec_mod.unpack_diffusion_output_shm(msg.output)
                    except Exception:
                        _exec_logger.exception(
                            "SHM unpack failed for batch %s", batch_id
                        )
                    batch_output = msg.output
                    for per_req_id, req_id in per_req_map.items():
                        req_output = batch_output.get_request_output(req_id)
                        if req_output is not None and req_output.result is not None:
                            per_req_result = req_output.result
                        elif msg.error:
                            per_req_result = _exec_mod.DiffusionOutput(error=msg.error)
                        else:
                            per_req_result = _exec_mod.DiffusionOutput(
                                error="No output result for batch request"
                            )
                        fut = _exec_mod.concurrent.futures.Future()
                        fut.set_result(per_req_result)
                        with self._futures_lock:
                            pending = self._output_futures.pop(per_req_id, None)
                            if pending is not None and not pending.done():
                                pending.set_result(per_req_result)
                            else:
                                self._completed_outputs[per_req_id] = fut
                else:
                    with self._futures_lock:
                        fut = self._output_futures.pop(batch_id, None) if batch_id else None
                    if fut is not None and not fut.done():
                        if msg.error:
                            fut.set_exception(RuntimeError(msg.error))
                        else:
                            try:
                                _exec_mod.unpack_diffusion_output_shm(msg.output)
                            except Exception as exc:
                                _exec_logger.exception(
                                    "SHM unpack failed in result pump"
                                )
                                fut.set_exception(exc)
                                continue
                            fut.set_result(msg.output)
                    elif batch_id:
                        fut = _exec_mod.concurrent.futures.Future()
                        if msg.error:
                            fut.set_exception(RuntimeError(msg.error))
                        else:
                            try:
                                _exec_mod.unpack_diffusion_output_shm(msg.output)
                            except Exception as exc:
                                _exec_logger.exception(
                                    "SHM unpack failed in result pump (cached)"
                                )
                                fut.set_exception(exc)
                            else:
                                fut.set_result(msg.output)
                        with self._futures_lock:
                            self._completed_outputs[batch_id] = fut

    def _kunlun_start_result_pump(self) -> None:
        self._pump_running = True
        self._pump_stop.clear()
        queues = getattr(self, "_result_mqs", None) or [self._result_mq]
        self._result_pump_threads = []
        for idx, result_mq in enumerate(queues):
            thread = _exec_mod.threading.Thread(
                target=self._result_pump,
                args=(result_mq,),
                daemon=True,
                name=f"DiffusionResultPump-{idx}",
            )
            thread.start()
            self._result_pump_threads.append(thread)
        _exec_logger.info("Async result pumps started (%d)", len(queues))

    _MDE._launch_workers = _kunlun_launch_workers
    _MDE._init_result_queue = _kunlun_init_result_queue
    _MDE._result_pump = _kunlun_result_pump
    _MDE._start_result_pump = _kunlun_start_result_pump
    _exec_mod._kunlun_dp2_queues_patched = True
    _logger.info("DP2: per-worker result queues installed")


def _patch_executor_dp_routing() -> None:
    """Route multi-request DP waves to the per-replica execute_model RPC."""
    try:
        import vllm_omni.diffusion.executor.multiproc_executor as _exec_mod
    except Exception:
        return
    if getattr(_exec_mod, "_kunlun_dp2_patched", False):
        return

    import json as _json

    from vllm_omni.diffusion.data import (
        AsyncDiffusionOutput,
        AsyncOutputKind,
        DiffusionOutput,
    )

    _executor_class = _exec_mod.MultiprocDiffusionExecutor
    _logger = _exec_mod.logger

    if hasattr(_executor_class, "_fail_closed_on_dp_wave_timeout"):
        # ── vLLM-Omni 0.28.0rc1 layout ────────────────────────────────
        _is_empty_dp_prompt = _exec_mod._is_empty_dp_prompt
        _DLO_DP_WAVE_TIMEOUT_S = _exec_mod._DLO_DP_WAVE_TIMEOUT_S

        def _kunlun_execute_request(self, scheduler_output):
            from vllm_omni.diffusion.sched.interface import (
                validate_new_request_data_identity,
            )
            from vllm_omni.diffusion.sched.request_scheduler import (
                build_request_batch_sampling_params_key,
            )
            from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput

            self._ensure_open()
            new_reqs = scheduler_output.scheduled_new_reqs
            runner_outputs: list = []
            for new_req in new_reqs:
                validate_new_request_data_identity(new_req)

            parallel_config = getattr(self.od_config, "parallel_config", None)
            dp_size = int(getattr(parallel_config, "data_parallel_size", 1) or 1)

            if len(new_reqs) > 1 and dp_size > 1:
                compatibility_keys = [
                    build_request_batch_sampling_params_key(nr.req) for nr in new_reqs
                ]
                if any(key != compatibility_keys[0] for key in compatibility_keys[1:]):
                    raise ValueError(
                        "DP multi-concurrency requires compatible shape, CFG, "
                        "denoise schedule, output count, and LoRA settings for all "
                        "requests in one collective wave."
                    )
                extra_args_signatures: set = set()
                for nr in new_reqs:
                    ea = getattr(nr.req.sampling_params, "extra_args", None)
                    extra_args_signatures.add(
                        _json.dumps(ea, sort_keys=True, default=repr)
                        if ea is not None
                        else None
                    )
                if len(extra_args_signatures) > 1:
                    raise ValueError(
                        "DP multi-concurrency requires all concurrent requests to "
                        "share identical extra_args. Different extra_args can change "
                        "the forward schedule and cause a collective deadlock."
                    )
                empty_prompt_ids = [
                    nr.request_id
                    for nr in new_reqs
                    if _is_empty_dp_prompt(nr.req.prompt)
                ]
                if empty_prompt_ids:
                    raise ValueError(
                        "DP multi-concurrency requires a non-empty prompt for every request; "
                        f"empty prompt request IDs: {empty_prompt_ids}."
                    )

                try:
                    results = self.collective_rpc(
                        "execute_model",
                        timeout=_DLO_DP_WAVE_TIMEOUT_S,
                        args=(new_reqs, self.od_config, scheduler_output.kv_prefetch_job),
                        unique_reply_rank=None,
                        exec_all_ranks=True,
                    )
                    results = results if isinstance(results, list) else [results]
                    for i, new_req in enumerate(new_reqs):
                        res = results[i] if i < len(results) else results[0]
                        if isinstance(res, DiffusionOutput):
                            runner_outputs.append(
                                RunnerOutput(
                                    request_id=new_req.request_id,
                                    step_index=None,
                                    finished=True,
                                    result=res,
                                )
                            )
                        else:
                            raise RuntimeError(
                                f"Unexpected response type [{i}]: {type(res)!r}"
                            )
                except Exception as exc:
                    if isinstance(exc, TimeoutError):
                        self._fail_closed_on_dp_wave_timeout(exc)
                    for new_req in new_reqs:
                        runner_outputs.append(
                            RunnerOutput(
                                request_id=new_req.request_id,
                                step_index=None,
                                finished=True,
                                result=DiffusionOutput(error=str(exc)),
                            )
                        )
                return BatchRunnerOutput.from_list(runner_outputs)

            for new_req in new_reqs:
                req = new_req.req
                try:
                    args: tuple = (
                        req,
                        self.od_config,
                        scheduler_output.kv_prefetch_job,
                    )
                    if new_req.diffusion_kv_metadata is not None:
                        args += (new_req.diffusion_kv_metadata,)
                    result = self.collective_rpc(
                        "execute_model",
                        args=args,
                        unique_reply_rank=0,
                        exec_all_ranks=True,
                    )
                    if (
                        isinstance(result, AsyncDiffusionOutput)
                        and result.kind == AsyncOutputKind.COMPUTE_DONE
                    ):
                        runner_outputs.append(
                            RunnerOutput(
                                request_id=new_req.request_id,
                                step_index=None,
                                finished=True,
                                result=None,
                                async_output_id=result.async_output_id,
                            )
                        )
                    elif isinstance(result, DiffusionOutput):
                        runner_outputs.append(
                            RunnerOutput(
                                request_id=new_req.request_id,
                                step_index=None,
                                finished=True,
                                result=result,
                            )
                        )
                    else:
                        raise RuntimeError(
                            f"Unexpected response type: {type(result)!r}"
                        )
                except Exception as exc:
                    runner_outputs.append(
                        RunnerOutput(
                            request_id=new_req.request_id,
                            step_index=None,
                            finished=True,
                            result=DiffusionOutput(error=str(exc)),
                        )
                    )
            return BatchRunnerOutput.from_list(runner_outputs)

        def _kunlun_execute_batch(self, scheduler_output):
            from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput

            self._ensure_open()
            if len(scheduler_output.scheduled_new_reqs) <= 1:
                return self.execute_request(scheduler_output)

            parallel_config = getattr(self.od_config, "parallel_config", None)
            dp_size = int(getattr(parallel_config, "data_parallel_size", 1) or 1)
            if dp_size > 1:
                return self.execute_request(scheduler_output)

            result = self.collective_rpc(
                "execute_model_batch",
                args=(scheduler_output, self.od_config),
                unique_reply_rank=0,
                exec_all_ranks=True,
            )
            if (
                isinstance(result, AsyncDiffusionOutput)
                and result.kind == AsyncOutputKind.COMPUTE_DONE
            ):
                batch_id = result.async_output_id
                per_req_map: dict = {}
                runner_outputs: list = []
                for new_req in scheduler_output.scheduled_new_reqs:
                    per_req_id = f"{batch_id}/{new_req.request_id}"
                    per_req_map[per_req_id] = new_req.request_id
                    runner_outputs.append(
                        RunnerOutput(
                            request_id=new_req.request_id,
                            step_index=None,
                            finished=True,
                            result=None,
                            async_output_id=per_req_id,
                        )
                    )
                with self._futures_lock:
                    early = self._completed_outputs.pop(batch_id, None)
                    if early is None:
                        self._batch_split_map[batch_id] = per_req_map
                if early is not None:
                    _logger.debug(
                        "Batch %s output arrived before split map; splitting now",
                        batch_id,
                    )
                    try:
                        batch_output = early.result()
                        error = None
                    except Exception as exc:
                        batch_output = None
                        error = str(exc)
                    self._deliver_batch_split(per_req_map, batch_output, error)
                return BatchRunnerOutput.from_list(runner_outputs)
            if not isinstance(result, BatchRunnerOutput):
                raise RuntimeError(
                    f"Unexpected response type for execute_batch: {type(result)!r}"
                )
            return result

    else:
        # ── vLLM-Omni 0.26.0 layout ───────────────────────────────────

        def _kunlun_execute_request(self, scheduler_output):
            from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput

            self._ensure_open()
            new_reqs = scheduler_output.scheduled_new_reqs
            runner_outputs: list = []

            parallel_config = getattr(self.od_config, "parallel_config", None)
            dp_size = int(getattr(parallel_config, "data_parallel_size", 1) or 1)

            if len(new_reqs) > 1 and dp_size > 1:
                step_counts = {
                    nr.req.sampling_params.num_inference_steps
                    for nr in new_reqs
                    if nr.req.sampling_params.num_inference_steps is not None
                }
                has_none = any(
                    nr.req.sampling_params.num_inference_steps is None
                    for nr in new_reqs
                )
                if (len(step_counts) > 1) or has_none:
                    raise ValueError(
                        "DP multi-concurrency requires all concurrent requests to have "
                        "the same explicit num_inference_steps (None is not allowed), got "
                        f"{[nr.req.sampling_params.num_inference_steps for nr in new_reqs]}."
                    )
                extra_args_signatures: set = set()
                for nr in new_reqs:
                    ea = getattr(nr.req, "extra_args", None)
                    if ea and isinstance(ea, dict):
                        extra_args_signatures.add(_json.dumps(ea, sort_keys=True))
                    else:
                        extra_args_signatures.add(None)
                if len(extra_args_signatures) > 1:
                    raise ValueError(
                        "DP multi-concurrency requires all concurrent requests to "
                        "share identical extra_args. Different extra_args can change "
                        "the forward schedule and cause AllGather deadlock."
                    )

                reqs_list = [nr.req for nr in new_reqs]
                try:
                    results = self.collective_rpc(
                        "execute_model",
                        args=(
                            reqs_list,
                            self.od_config,
                            scheduler_output.kv_prefetch_job,
                        ),
                        unique_reply_rank=None,
                        exec_all_ranks=True,
                    )
                    results = results if isinstance(results, list) else [results]
                    for i, new_req in enumerate(new_reqs):
                        res = results[i] if i < len(results) else results[0]
                        if isinstance(res, DiffusionOutput):
                            runner_outputs.append(
                                RunnerOutput(
                                    request_id=new_req.request_id,
                                    step_index=None,
                                    finished=True,
                                    result=res,
                                )
                            )
                        else:
                            raise RuntimeError(
                                f"Unexpected response type [{i}]: {type(res)!r}"
                            )
                except Exception as exc:
                    if isinstance(exc, TimeoutError) and hasattr(
                        self, "_fail_closed_on_dp_wave_timeout"
                    ):
                        self._fail_closed_on_dp_wave_timeout(exc)
                    for new_req in new_reqs:
                        runner_outputs.append(
                            RunnerOutput(
                                request_id=new_req.request_id,
                                step_index=None,
                                finished=True,
                                result=DiffusionOutput(error=str(exc)),
                            )
                        )
                return BatchRunnerOutput.from_list(runner_outputs)

            for new_req in new_reqs:
                req = new_req.req
                try:
                    result = self.collective_rpc(
                        "execute_model",
                        args=(req, self.od_config, scheduler_output.kv_prefetch_job),
                        unique_reply_rank=0,
                        exec_all_ranks=True,
                    )
                    if (
                        isinstance(result, AsyncDiffusionOutput)
                        and result.kind == AsyncOutputKind.COMPUTE_DONE
                    ):
                        runner_outputs.append(
                            RunnerOutput(
                                request_id=new_req.request_id,
                                step_index=None,
                                finished=True,
                                result=None,
                                async_output_id=result.async_output_id,
                            )
                        )
                    elif isinstance(result, DiffusionOutput):
                        runner_outputs.append(
                            RunnerOutput(
                                request_id=new_req.request_id,
                                step_index=None,
                                finished=True,
                                result=result,
                            )
                        )
                    else:
                        raise RuntimeError(
                            f"Unexpected response type: {type(result)!r}"
                        )
                except Exception as exc:
                    runner_outputs.append(
                        RunnerOutput(
                            request_id=new_req.request_id,
                            step_index=None,
                            finished=True,
                            result=DiffusionOutput(error=str(exc)),
                        )
                    )
            return BatchRunnerOutput.from_list(runner_outputs)

        def _kunlun_execute_batch(self, scheduler_output):
            from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput

            self._ensure_open()
            if len(scheduler_output.scheduled_new_reqs) <= 1:
                return self.execute_request(scheduler_output)

            parallel_config = getattr(self.od_config, "parallel_config", None)
            dp_size = int(getattr(parallel_config, "data_parallel_size", 1) or 1)
            if dp_size > 1:
                return self.execute_request(scheduler_output)

            result = self.collective_rpc(
                "execute_model_batch",
                args=(scheduler_output, self.od_config),
                unique_reply_rank=0,
                exec_all_ranks=True,
            )
            if (
                isinstance(result, AsyncDiffusionOutput)
                and result.kind == AsyncOutputKind.COMPUTE_DONE
            ):
                batch_id = result.async_output_id
                per_req_map: dict = {}
                runner_outputs: list = []
                for new_req in scheduler_output.scheduled_new_reqs:
                    per_req_id = f"{batch_id}/{new_req.request_id}"
                    per_req_map[per_req_id] = new_req.request_id
                    runner_outputs.append(
                        RunnerOutput(
                            request_id=new_req.request_id,
                            step_index=None,
                            finished=True,
                            result=None,
                            async_output_id=per_req_id,
                        )
                    )
                with self._futures_lock:
                    self._batch_split_map[batch_id] = per_req_map
                return BatchRunnerOutput.from_list(runner_outputs)
            if not isinstance(result, BatchRunnerOutput):
                raise RuntimeError(
                    f"Unexpected response type for execute_batch: {type(result)!r}"
                )
            return result

    _executor_class.execute_request = _kunlun_execute_request
    _executor_class.execute_batch = _kunlun_execute_batch
    _exec_mod._kunlun_dp2_patched = True


def _patch_minimax_h3_dp_aware_encode() -> None:
    """Scope MiniMax H3 tokenisation/encode to the DP replica (TP group)."""
    try:
        import vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 as _h3
    except Exception:
        return
    if getattr(_h3, "_kunlun_dp2_patched", False):
        return

    _orig_dit_rank_world = _h3._dit_rank_world

    def _kunlun_dit_rank_world():
        """Return the DiT-level group/rank/world scoped to the TP replica.

        The upstream implementation used the whole default world group, so with
        data_parallel_size > 1 every DP replica believed the DiT group spanned
        all ranks and rank-0-only work (tokenisation, reference preparation,
        tensor broadcasts) was performed once globally.  The tensor-parallel
        group is the correct DiT scope for every DP>1 layout; for DP=1 it is
        identical to the world group.
        """
        try:
            import vllm.distributed.parallel_state as _vllm_ps

            tp_group = getattr(_vllm_ps, "_TP", None)
            device_group = getattr(tp_group, "device_group", None)
            if device_group is not None:
                return (
                    device_group,
                    int(getattr(tp_group, "rank_in_group", 0)),
                    int(getattr(tp_group, "world_size", 1)),
                )
        except Exception:
            pass
        return _orig_dit_rank_world()

    def _kunlun_broadcast_tensor(tensor, *, dtype, device):
        """Replica-aware version of the module-level ``_broadcast_tensor``.

        Upstream hardcodes ``src=0``, which is only valid for the first DP
        replica.  Once ``_dit_rank_world`` is scoped to the TP replica group,
        the source rank must be the first global rank of that replica
        (``global_rank - rank_in_group``).
        """
        import torch
        import torch.distributed as _dist

        group, rank, world_size = _h3._dit_rank_world()
        if world_size == 1:
            if tensor is None:
                raise ValueError("source tensor is required for single-rank execution")
            return tensor.to(device=device, dtype=dtype)

        shape = torch.zeros(5, dtype=torch.long, device=device)
        if rank == 0:
            if tensor is None:
                raise ValueError("rank 0 must provide a tensor to broadcast")
            shape[0] = tensor.ndim
            shape[1 : tensor.ndim + 1] = torch.tensor(
                tensor.shape,
                device=device,
            )
        try:
            src = int(_dist.get_rank()) - int(rank)
        except Exception:
            src = 0
        _dist.broadcast(shape, src=src, group=group)
        ndim = int(shape[0].item())
        tensor_shape = tuple(int(v) for v in shape[1 : ndim + 1].tolist())
        if rank == 0:
            output = tensor.to(device=device, dtype=dtype).contiguous()
        else:
            output = torch.empty(tensor_shape, device=device, dtype=dtype)
        _dist.broadcast(output, src=src, group=group)
        return output

    def _kunlun_build_text_encoder_group(self, text_encoder_tp_size: int):
        """Build one text-encoder TP group per DP replica.

        Upstream only covers global ranks ``[0, text_encoder_tp_size)``.  With
        DP2 the second replica must own its own encoder group.  Constructing
        per-replica ``new_group`` calls with mismatched rank lists deadlocks,
        so instead every rank constructs the *same ordered list* of
        replica-local rank sets through one GroupCoordinator; the new_group
        collectives then complete and every rank joins exactly one group.
        """
        if int(text_encoder_tp_size) == 1:
            return _h3._SingleRankEncoderGroup(rank=self._dit_rank)
        _, _, dit_world = _h3._dit_rank_world()
        dit_world = int(dit_world)
        if int(text_encoder_tp_size) != dit_world:
            raise ValueError(
                "Kunlun DP2 MiniMax H3 requires text_encoder_tp_size == "
                f"tensor_parallel_size ({dit_world}), got {text_encoder_tp_size}"
            )
        import torch.distributed as _dist

        from vllm_omni.diffusion.distributed.group_coordinator import (
            GroupCoordinator,
        )

        world_size = int(_dist.get_world_size())
        num_replicas = max(1, world_size // dit_world)
        group_ranks = [
            list(
                range(
                    rep * dit_world,
                    rep * dit_world + int(text_encoder_tp_size),
                )
            )
            for rep in range(num_replicas)
        ]
        return GroupCoordinator(
            group_ranks=group_ranks,
            local_rank=_h3.envs.LOCAL_RANK,
            torch_distributed_backend=_h3.current_omni_platform.dist_backend,
        )

    _h3._dit_rank_world = _kunlun_dit_rank_world
    _h3._broadcast_tensor = _kunlun_broadcast_tensor
    _h3.MiniMaxH3Pipeline._build_text_encoder_group = (
        _kunlun_build_text_encoder_group
    )
    _h3._kunlun_dp2_patched = True


def _patch_minimax_h3_vae_dp_parallel() -> None:
    """Scope MiniMax H3 video VAE patch parallelism to the DP replica.

    ``MiniMaxH3VideoVAE.set_parallel_size`` derives its process group from
    ``get_dit_group()``, which under DP2 spans every replica (8 ranks).  The
    native VAE then rejects any ``vae_patch_parallel_size`` other than 1 or
    the full 8-rank group, forcing single-rank decode on Kunlun.  Scope the
    group to the tensor-parallel replica so ``vae_patch_parallel_size ==
    tensor_parallel_size`` (4) is accepted and decode parallelises inside
    each replica.
    """
    try:
        import vllm_omni.diffusion.models.minimax_h3.vae as _vae
    except Exception:
        return
    if getattr(_vae, "_kunlun_dp2_vae_patched", False):
        return

    _orig_set_parallel_size = _vae.MiniMaxH3VideoVAE.set_parallel_size

    def _kunlun_set_parallel_size(self, parallel_size: int, mode: str = "tile"):
        if mode != "tile":
            raise ValueError(
                f"MiniMax H3 VAE supports its native tile parallel mode only, got {mode!r}"
            )
        try:
            import vllm.distributed.parallel_state as _vllm_ps

            tp_group = getattr(_vllm_ps, "_TP", None)
            device_group = getattr(tp_group, "device_group", None)
            if device_group is None:
                return _orig_set_parallel_size(self, parallel_size, mode=mode)
            group = device_group
            world_size = int(getattr(tp_group, "world_size", 1))
            rank = int(getattr(tp_group, "rank_in_group", 0))
        except Exception:
            return _orig_set_parallel_size(self, parallel_size, mode=mode)

        import importlib

        parallel_size = int(parallel_size)
        if parallel_size not in (1, world_size):
            raise ValueError(
                "MiniMax H3 native VAE patch parallelism currently requires "
                "vae_patch_parallel_size=1 or the full DiT group size "
                f"({world_size}), got {parallel_size}"
            )
        self.parallel_size = parallel_size
        enabled = parallel_size > 1

        package = self.remote.__class__.__module__.rsplit(".", 1)[0]
        parallel_module = importlib.import_module(f"{package}.parallel")
        state = parallel_module.get_parallel_state()
        state.clear()
        state.update(
            group_size=parallel_size,
            group_rank=rank if enabled else 0,
            local_process_group=group if enabled else None,
            sp_size=parallel_size,
            sp_rank=rank if enabled else 0,
            sp_enabled=enabled,
            sp_process_group=group if enabled else None,
            tp_size=1,
            tp_rank=0,
        )
        self.model.parallel_tiling = enabled

    _vae.MiniMaxH3VideoVAE.set_parallel_size = _kunlun_set_parallel_size
    _vae._kunlun_dp2_vae_patched = True
    _logger.info("DP2: MiniMaxH3VideoVAE replica-scoped patch parallelism installed")


def apply_kunlun_dp2_patches() -> None:
    """Apply all DP2 concurrency patches (idempotent, failure-tolerant)."""
    global _KUNLUN_DP2_PATCHES_APPLIED
    if _KUNLUN_DP2_PATCHES_APPLIED:
        return
    _patch_parallel_config_dp_env_override()
    _patch_engine_dp_concurrency()
    _patch_executor_dp_result_queues()
    _patch_executor_dp_rpc_routing()
    _patch_executor_dp_routing()
    _patch_minimax_h3_dp_aware_encode()
    _patch_minimax_h3_vae_dp_parallel()
    _KUNLUN_DP2_PATCHES_APPLIED = True
