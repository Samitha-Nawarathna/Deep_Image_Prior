from dip_core.abstractions.logger import Logger
from dip_core.base_implementations.default_logger import DefaultLogger

class DenoisingLogger(DefaultLogger):
    def __init__(self, gt, metrics_list=None, log_dir="./logs", image_per=100):
        super().__init__(log_dir)
        self.gt = gt
        self.image_per = image_per
        self.metrics_list = metrics_list or []

    def log_results(self, metrics, i, **kwargs):
        # log loss from step
        self.log_scalar("loss", metrics["loss"], i)

        # log images periodically
        if i % self.image_per == 0:
            self.log_image("output", metrics["output"][0], i)

        # compute and log additional metrics
        output = metrics["output"][0]  # or adjust if batch
        for metric in self.metrics_list:
            try:
                value = metric(gt=self.gt, output=output, **kwargs)
                self.log_scalar(metric.name(), value, i)
            except Exception as e:
                print(f"Metric {metric.name()} failed at step {i}: {e}")

    def gt_metrics(self, input, **kwargs):
        results = {}
        for metric in self.metrics_list:
            try:
                value = metric(self.gt, input, **kwargs)
                results[metric.name()] = value
            except Exception as e:
                print(f"Metric {metric.name()} failed for ground truth: {e}")
                return
            
            self.log_str("".join([f"GT {metric}: {value}\n" for metric, value in results.items()]))
