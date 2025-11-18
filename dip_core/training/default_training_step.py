import torch

class DefaultTrainingStep:
    def __call__(self, model, input_noise, target, operator, loss_fn, optimizer, logger, iteration):
        optimizer.zero_grad()
        output = model(input_noise)
        loss = loss_fn(operator.forward(output), target)
        loss.backward()
        optimizer.step()
        return {"loss": loss.item(), "output": output}
