from contextlib import contextmanager
from time import perf_counter

import torch
import drjit as dr


class TrainingProfiler:
    """Synchronous stage timer used only by the optional profiling mode."""

    def __init__(self, steps):
        self.steps = steps
        self.step_times = []
        self._stage_step_times = []
        self._stage_order = []
        self._step = None

    @property
    def active(self):
        return self._step is not None and self._step < self.steps

    def start_step(self, step):
        self._step = step
        self.step_times.append(0.0)
        self._stage_step_times.append({})

    def _sync(self):
        # Dr.Jit and PyTorch may enqueue work on different CUDA streams.
        dr.sync_thread()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _record(self, name, elapsed, top_level):
        stage_times = self._stage_step_times[-1]
        stage_times[name] = stage_times.get(name, 0.0) + elapsed
        if name not in self._stage_order:
            self._stage_order.append(name)
        if top_level:
            self.step_times[-1] += elapsed

    @contextmanager
    def measure(self, name):
        if not self.active:
            yield
            return

        self._sync()
        start = perf_counter()
        try:
            yield
        finally:
            self._sync()
            elapsed = perf_counter() - start
            self._record(name, elapsed, True)

    def report(self):
        if not self.step_times:
            return

        print("\n=== Training profile ===", flush=True)
        print(f"Profiled steps: {len(self.step_times)}", flush=True)
        print(
            f"Total step time: {sum(self.step_times):.3f} s "
            f"({sum(self.step_times) / len(self.step_times):.4f} s/step)",
            flush=True,
        )
        if len(self.step_times) > 1:
            steady = self.step_times[1:]
            print(
                f"Steady step time (excluding step 0): "
                f"{sum(steady) / len(steady):.4f} s/step",
                flush=True,
            )

        print("\nStage                            all steps     steady     first", flush=True)
        print("-------------------------------- ---------- ---------- ----------", flush=True)
        for name in self._stage_order:
            values = [
                stage_times.get(name, 0.0)
                for stage_times in self._stage_step_times
            ]
            all_avg = sum(values) / len(values)
            steady_values = values[1:] if len(values) > 1 else values
            steady_avg = sum(steady_values) / len(steady_values)
            first = values[0]
            print(
                f"{name:32} {all_avg:8.4f}s {steady_avg:8.4f}s {first:8.4f}s",
                flush=True,
            )


class RenderProfiler:
    """Synchronous profiler for one-shot setup and repeated render passes."""

    def __init__(self, passes):
        self.passes = passes
        self._pass_stage_times = []
        self._pass_total_times = []
        self._pass_stage_order = []
        self._one_shot_order = []
        self._one_shot_times = {}
        self._segments = {}
        self._current_pass = None

    def _sync(self):
        # Dr.Jit and PyTorch can enqueue work on different CUDA streams.
        dr.sync_thread()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _record(self, target, name, elapsed):
        target[name] = target.get(name, 0.0) + elapsed

    @contextmanager
    def measure(self, name):
        self._sync()
        start = perf_counter()
        try:
            yield
        finally:
            self._sync()
            elapsed = perf_counter() - start
            if self._current_pass is None:
                self._record(self._one_shot_times, name, elapsed)
                if name not in self._one_shot_order:
                    self._one_shot_order.append(name)
            else:
                self._record(self._pass_stage_times[-1], name, elapsed)
                if name not in self._pass_stage_order:
                    self._pass_stage_order.append(name)

    def start_segment(self, name):
        self._sync()
        self._segments[name] = perf_counter()

    def stop_segment(self, name):
        segment = self._segments.pop(name, None)
        if segment is None:
            return
        self._sync()
        elapsed = perf_counter() - segment
        self._record(self._pass_stage_times[-1], name, elapsed)
        if name not in self._pass_stage_order:
            self._pass_stage_order.append(name)

    @contextmanager
    def pass_scope(self):
        self._sync()
        start = perf_counter()
        self._pass_stage_times.append({})
        self._current_pass = len(self._pass_stage_times) - 1
        try:
            yield
        finally:
            self._sync()
            self._pass_total_times.append(perf_counter() - start)
            self._segments.clear()
            self._current_pass = None

    def report(self):
        print("\n=== Render profile ===", flush=True)
        print(f"Render passes: {len(self._pass_stage_times)}", flush=True)

        if self._pass_stage_times:
            pass_totals = self._pass_total_times
            print(
                f"Total pass time: {sum(pass_totals):.3f} s "
                f"({sum(pass_totals) / len(pass_totals):.4f} s/pass)",
                flush=True,
            )
            if len(pass_totals) > 1:
                steady = pass_totals[1:]
                print(
                    f"Steady pass time (excluding pass 0): "
                    f"{sum(steady) / len(steady):.4f} s/pass",
                    flush=True,
                )

            print(
                "\nRepeated stage                 all passes    steady     first      last",
                flush=True,
            )
            print(
                "----------------------------- ----------- ---------- ---------- ----------",
                flush=True,
            )
            for name in self._pass_stage_order:
                values = [
                    stage_times.get(name, 0.0)
                    for stage_times in self._pass_stage_times
                ]
                all_avg = sum(values) / len(values)
                steady_values = values[1:] if len(values) > 1 else values
                steady_avg = sum(steady_values) / len(steady_values)
                print(
                    f"{name:29} {all_avg:9.4f}s {steady_avg:9.4f}s "
                    f"{values[0]:9.4f}s {values[-1]:9.4f}s",
                    flush=True,
                )

        if self._one_shot_times:
            print("\nOne-shot stage", flush=True)
            print("----------------------------- ----------", flush=True)
            for name in self._one_shot_order:
                elapsed = self._one_shot_times[name]
                print(f"{name:29} {elapsed:9.4f}s", flush=True)
