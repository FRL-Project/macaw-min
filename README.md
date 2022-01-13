# MACAW with Meta-World

The original code is from Eric Mitchell's repository https://github.com/eric-mitchell/macaw-min.
This code is an adapted version wich allows to train MACAW with Meta-World version 2. The main pyhton file is `impl.py`. 

## Run a job
The script `submit_jobs.py` can generate the jog commands automatically.

As an example to run a metaworld ml10 job following command can be used:
```
python ./impl.py inner_policy_lr=0.1 inner_value_lr=0.1 task_config=task_config/metaworld_ml10.json
```
