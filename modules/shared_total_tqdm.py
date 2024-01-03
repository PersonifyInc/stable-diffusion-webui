import tqdm

from modules import shared


class TotalTQDM:
    def __init__(self):
        self._tqdm = None
        self.step = 0
        self.extra_steps = 0

    def reset(self):
        self._tqdm = tqdm.tqdm(
            desc="Total progress",
            total=shared.state.job_count * shared.state.sampling_steps + self.extra_steps,
            position=1,
            file=shared.progress_print_out
        )
        self.step = 0

    def update(self):
        if not shared.opts.multiple_tqdm or shared.cmd_opts.disable_console_progressbars:
            return
        if self._tqdm is None:
            self.reset()
        self._tqdm.update()
        self.step += 1

    def updateTotal(self, new_total):
        if not shared.opts.multiple_tqdm or shared.cmd_opts.disable_console_progressbars:
            return
        if self._tqdm is None:
            self.reset()
        self._tqdm.total = new_total + self.extra_steps

    def updateExtraSteps(self, extra_steps):
        self.extra_steps = extra_steps

    def clear(self):
        if self._tqdm is not None:
            self._tqdm.refresh()
            self._tqdm.close()
            self._tqdm = None
        self.step = 0
        self.extra_steps = 0

    def get_total_steps(self):
        if self._tqdm is not None:
            return self._tqdm.total
        return 0
