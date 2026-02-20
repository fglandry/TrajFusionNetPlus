from transformers import TrainerCallback

class IgnoreEarlyBestModelCallback(TrainerCallback):
    def __init__(self, min_epoch=5):
        self.min_epoch = min_epoch

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # state.epoch is a float (e.g. 3.0)
        if state.epoch is not None and state.epoch < self.min_epoch:
            # remove the metric used for best model selection
            metric_key = f"eval_{args.metric_for_best_model}"
            if metric_key in metrics:
                if args.greater_is_better:
                    worst_value = float("-inf")
                else:
                    worst_value = float("inf")
                metrics[metric_key] = worst_value