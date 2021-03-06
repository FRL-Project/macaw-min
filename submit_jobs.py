"""
Main class to speed ub job creation on the cluster.
"""
import itertools
import os

import click

task_config_file = {
    # env name - file name component
    "ml1": 'metaworld_ml1.json',
    "ml10": 'metaworld_ml10.json',
    "ml45": 'metaworld_ml45.json'
}

experiments_params = {
    "macaw": {
        "outer_lr": {
            "outer_policy_lr": [1e-3, 1e-4, 1e-5],
            "outer_value_lr": [1e-3, 1e-4, 1e-5]
        },
        "inner_lr": {
            'inner_policy_lr': [1e-1, 1e-2, 1e-3],
            'inner_value_lr': [1e-1, 1e-2, 1e-3]
        }
    }
}


def product_dict(**kwargs):
    # https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


@click.command()
@click.option('--algo', default="macaw")
@click.option('--env', default="ml10")
@click.option('--exp', default="outer_lr")
@click.option('--time', default="120:00")
@click.option('--n_cpus', default="10")
@click.option('--mem', default="6144")
@click.option('--gpu', default=None)
def submit_job(algo, env, exp, time, n_cpus, mem, gpu=None, path="."):
    task_config_path = os.path.join("task_config/", task_config_file[env])

    python_file_path = os.path.join(path, "impl.py")

    experiment_parameter = experiments_params[algo][exp]

    for params in product_dict(**experiment_parameter):

        command = ''

        # use 4 cpus
        command += 'bsub -n ' + n_cpus
        command += ' -J "' + algo + '-' + env + '-' + exp + ':' + str(list(params.values())).replace("[", "").replace("]", "") + '"'
        # job time
        command += ' -W ' + time
        # memory per cpu
        command += ' -R "rusage[mem=' + mem
        if gpu is None:
            command += ']"'
        elif gpu is not None:
            command += ', ngpus_excl_p=1]"'
            if gpu == 1:
                command += ' -R "select[gpu_model0==GeForceGTX1080Ti]"'
            elif gpu == 2:
                command += ' -R "select[gpu_model0==GeForceRTX2080Ti]"'
            else:
                command += ' -R "select[gpu_mtotal0>=10240]"'  # GPU memory more then 10GB

        command += ' "python ' + str(python_file_path)
        for par in params:
            command += ' ' + par + '=' + str(params[par])

        command += ' ' + 'task_config=' + task_config_path

        command += '"'

        print(command)
        os.system(command)


submit_job()
