# Replication scripts

First, note the dependencies in [README_dependencies.md](README_dependencies.md).

## Online Appendix

Separately, replication scripts for the additional results in the Online Appendix appear in [README_replication_appendix.md](README_replication_appendix.md). It may be easier to go through these initially to get an overall sense of how the various pieces fit together, since it is a much smaller set of experiments. The FCE and BEA 2019 scripts illustrate the core zero-shot detection functionality, and the CoNLL scripts additionally show how to fine-tune the base model for supervised labeling and how to approximate the model with a K-NN, as with the experiments in the main text, below. (The only main methods not demonstrated in the above scripts are the generation of aggregate feature summaries and updating the support set, both of which are demonstrated below.)


## Main Text

Replication scripts for the experiments in the main text appear in [README_replication_main_text.md](README_replication_main_text.md). We illustrate the core behavior and options, from which running any additional experiments not explicitly included with its own code block then just amounts to choosing the particular input files and model configurations which should then be clear. There are quite a few experiments, so we have aimed to arrange these in an accessible manner. These scripts are a mix of tutorial and replication notes. To make our internal scripts easier to follow, we have in many cases duplicated code blocks that have differing environment variable dependencies; however, in some cases, such as changing the dataset splits, you will still need to comment/uncomment the applicable variables. As mentioned above, the notes for the experiments in the Online Appendix are likely easier to follow as an initial first pass.
