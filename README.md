# GPT2-Fine-Tuning-With-Custom-Data

**Since this is a hobby project, it was not written with a clean code approach. Moreover, of the studies mentioned below, only some files from the works have been added to the repo.**

This project is for our graduation project at Ankara University. We planned to integrate a chatbot into our tourist app project. To this end, we experimented with some different LLMs such as `GPT2`, `BLOOM`, `GPT-J`, `GPT-Neo` which are free to use in `HuggingFace Trasformers Library`.

We decided it was most efficient to organize the custom dataset as follows:

> What is Furkan's favourite food? <QUESTION\>

> Furkan's favourite food is unknown for now. <END\>

With the `<QUESTION>` tag, we aimed for the model to prioritize the custom dataset while generating the output. Since the words with the `<QUESTION>` tag will have the higher probabilities while model calculating the probabilities for the output, the model will generate the outputs from custom dataset mostly.

We used `<END>` tag because we needed a tag to get only the part we were interested in from the output of the model. Because the model can not know where it should stop generating the output and it continues to generate outputs after generating `<END>` tags till reaching the output limit which we gave the model as a hyperparameter.

We also used following part because we aimed for the model to generate this output for the questions that are out of the custom dataset.

> <QUESTION\>

> Sorry, I only have some information about Furkan and I don't have knowledge about this. <END\>

Since we do not have large individual hardware (GPU, TPU etc.) today, we can not use the models that have high parameters. Freezing the first layers, using fp16, using adafactor optimizer and using checkpoint gradient techniques allowed us to work with models with a little more parameters but still it was not enough.

After all these experiments, we realized that the parameters of the model are not sufficient for professional outputs. While a model like GPT-3.5 had 170 billion parameters, our models had at most 1 billion parameters, which shows that GPT-3.5 model has hundreds of times more understanding capacity with hundreds of times more parameters.

In order to be able to use the models with such few parameters, especially for the question-answering purposes, the only way is to prepare a very professional custom dataset. The model's scores will increase if there is every possible question and answer in the custom dataset.

Whats's more, we only used `AutoModelForCausalLM()` class for casual language models. It can be tried to use `AutoModelForQuestionAnswering()` classes for LLMs or different models which have only encoder part of transformer arthitecture such as BERT, for the question-answering purpose.

