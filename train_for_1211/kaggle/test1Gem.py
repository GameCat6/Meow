import keras
import keras_nlp
import numpy as np

gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_1.1_instruct_2b_en")
gemma_lm.generate("Keras is a", max_length=30)

# Generate with batched prompts.
gemma_lm.generate(["Keras is a", "I want to say"], max_length=30)