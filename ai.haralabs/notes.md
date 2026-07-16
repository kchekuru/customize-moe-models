Pass `model_name="thinkingmachines/Inkling"` anywhere the cookbook accepts a
model name. The appropriate renderer and tokenizer are selected automatically.

--

```python
import tinker
service_client = tinker.ServiceClient()
training_client = service_client.create_lora_training_client(
  base_model="thinkingmachines/Inkling", rank=32,
)
training_client.forward_backward(...)
training_client.optim_step(...)
training_client.save_state(...)
training_client.load_state(...)

sampling_client = training_client.save_weights_and_get_sampling_client()
sampling_client.sample(...)
```
Inkling is not the strongest overall model available today, open or closed. Instead, a combination of qualities makes it a good open-weights base for customization: multimodal capabilities, efficient thinking, and availability on Tinker for fine-tuning.