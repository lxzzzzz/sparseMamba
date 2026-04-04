try:
    from torch.utils.tensorboard import SummaryWriter as _TorchSummaryWriter

    class SummaryWriter(_TorchSummaryWriter):
        pass

except Exception:
    class SummaryWriter:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            self.log_dir = kwargs.get('log_dir', None)

        def add_scalar(self, *args, **kwargs):
            return None

        def add_scalars(self, *args, **kwargs):
            return None

        def add_text(self, *args, **kwargs):
            return None

        def flush(self):
            return None

        def close(self):
            return None
