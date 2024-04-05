# Instructions

For details about installation and data tokenization go to: https://github.com/huggingface/brrr/tree/main/examples/fineweb-runs

These are scripts adapted for my 1B runs, the cosmo-1B config and training slurm are available under `brrr/cosmopedia` folder but that config may not work out of the box because of some breaking changes made in `brrr`.

After adapting the paths and env in `launch_training_1b.py`  and the evaluation jinja file, you can launch the training using:

```bash
python launch_training_1b.py $local_data_path  $run_name
```

The dataset needs to be tokenized with `datatrove` as explained in https://github.com/huggingface/brrr/tree/main/examples/fineweb-runs 