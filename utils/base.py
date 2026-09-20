from collections import OrderedDict
from hooks.hook import Hook
from hooks.priority import get_priority
from torch import nn


class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()
        # set common hooks during training
        self._hooks = []  # record underlying hooks
        self.hooks_dict = OrderedDict()  # actual object to be used to call hooks
        # self.set_hooks()

    def process_out_dict(self, out_dict = None, **kwargs):
        """
        process the out_dict as return of train_step
        """
        if out_dict is None:
            out_dict = {}

        for arg, var in kwargs.items():
            out_dict[arg] = var

        # process res_dict, add output from res_dict to out_dict if necessary
        return out_dict

    def process_log_dict(self, log_dict = None, prefix = 'train', **kwargs):
        """
        process the tb_dict as return of train_step
        """
        if log_dict is None:
            log_dict = {}

        for arg, var in kwargs.items():
            log_dict[f'{prefix}/' + arg] = var
        return log_dict

    def get_save_dict(self, load_path):
        """
        Create a dictionary of additional arguments to save for when saving the model.
        """
        # Base arguments for all models
        save_dict = {
            "model": self.model.state_dict(),
            "ema_model": self.ema_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "loss_scaler": self.loss_scaler.state_dict(),
            "it": self.it + 1,
            "epoch": self.epoch + 1,
            "best_it": self.best_it,
        }
        if self.task_type == 'cls':
            save_dict['best_eval_acc'] = self.best_eval_metric
        else:
            save_dict['best_eval_mse'] = self.best_eval_metric
        if self.scheduler is not None:
            # Using a scheduler, so save its state dict
            save_dict["scheduler"] = self.scheduler.state_dict()

        if hasattr(self, "aim_run_hash"):
            # Using Aim, so save the run hash so that tracking can be resumed
            save_dict["aim_run_hash"] = self.aim_run_hash

        return save_dict

    def register_hook(self, hook, name = None, priority = 'NORMAL'):
        """
        Ref: https://github.com/open-mmlab/mmcv/blob/a08517790d26f8761910cac47ce8098faac7b627/mmcv/runner/base_runner.py#L263
        Register a hook into the hook list.
        The hook will be inserted into a priority queue, with the specified
        priority (See :class:`Priority` for details of priorities).
        For hooks with the same priority, they will be triggered in the same
        order as they are registered.
        Args:
            hook (:obj:`Hook`): The hook to be registered.
            hook_name (:str, default to None): Name of the hook to be registered. Default is the hook class name.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        # assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority  # type: ignore
        hook.name = name if name is not None else type(hook).__name__

        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:  # type: ignore
                self._hooks.insert(i + 1, hook)
                inserted = True
                break

        if not inserted:
            self._hooks.insert(0, hook)

        # call set hooks
        self.hooks_dict = OrderedDict()
        for hook in self._hooks:
            self.hooks_dict[hook.name] = hook

    def call_hook(self, fn_name, hook_name = None, *args, **kwargs):
        """Call all hooks.
        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
            hook_name (str): The specific hook name to be called, such as
                "param_update" or "dist_align", uesed to call single hook in train_step.
        """

        if hook_name is not None:
            return getattr(self.hooks_dict[hook_name], fn_name)(self, *args, **kwargs)

        for hook in self.hooks_dict.values():
            if hasattr(hook, fn_name):
                getattr(hook, fn_name)(self, *args, **kwargs)
