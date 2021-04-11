
# Utils
import gc
import sys
import time
import json
import torch
import queue
import warnings
import argparse
import traceback
import numpy as np
import multiprocessing
from hashlib import md5
# Data
from graphnas.utils.data_utils import load_data
# Metrics
from graphnas.utils.model_utils import test
from graphnas.utils.model_utils import train
from graphnas.utils.model_utils import build_gnn
# XMLRPC Server
from xmlrpc.server import SimpleXMLRPCServer


warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_pipeline_results_dir(id_, train, test, extra, error):
    return {
        'id': id_,
        'train': train,
        'test': test,
        'extra': extra,
        'error': error
    }


def evaluate_pipeline(inputs, evaluated_list):
    while True:
        try:
            print("Waiting for input in evaluate_pipeline...")
            pipeline_id, pipeline = inputs.get(block=True)
            if pipeline_id == 'exit':
                inputs.close()  # Close queue: clean exit
                return 1
            print("GOT INPUT in evaluate_pipeline...", pipeline_id)
            # Evaluate pipeline and add results to evaluated list
            evaluated_list.append(
                evaluate_pipeline_sync(pipeline_id, pipeline))
            # Collect cached objects
            gc.collect()
            # Clear GPU cache
            torch.cuda.empty_cache()
        except Exception:
            time.sleep(1)


def evaluate_pipeline_sync(pipeline_id, pipeline):
    try:
        # Load data
        data = load_data(pipeline['dataset_name'],
                         pipeline['random_seed'],
                         pipeline['number_of_folds'])
        num_classes = data.num_classes
        num_features = data.num_features
        candidate_arch = pipeline['candidate_arch']
        # Set random seed for this arch
        torch.manual_seed(pipeline['random_seed'])
        np.random.seed(pipeline['random_seed'])
        # Build the pytorch model from specification
        print("Building GNN model")
        model = build_gnn(candidate_arch, pipeline['search_space'],
                          num_features, num_classes, pipeline['in_drop'])
        print('model: ', model)
        # Print number of parameters and dimensions
        pytorch_total_params = sum(p.numel()
                                   for p in model.parameters())
        print('Number of params: ', pytorch_total_params)
        extra_info = {
            'model_str': str(model),
            'num_params': pytorch_total_params,
            'params_shape': '',
        }
        # Calculate and save the shape of the model's parameters
        for parameter in model.parameters():
            if parameter.requires_grad:
                extra_info['params_shape'] += str(parameter.shape)
        print('Parameter shape: ', extra_info['params_shape'])
        # Move model to device
        print('Move model to device')
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=pipeline['learning_rate'],
                                     weight_decay=pipeline['weight_decay'])
        if pipeline['batches'] > 0:
            train_metrics, test_metrics = \
                evaluate_model(model=model,
                               data=data,
                               optimizer=optimizer,
                               epochs=pipeline['epochs'],
                               n_batches=pipeline['batches'],
                               show_info=False)
            # Add results to evaluated list
            del model
            del optimizer
            del data
            gc.collect()
            torch.cuda.empty_cache()
            return build_pipeline_results_dir(
                id_=pipeline_id, train=train_metrics, test=test_metrics,
                extra=extra_info, error={})
        else:
            print('Invalid number of batches!! batches <= 0')
            error_dict = {
                'type': '',
                'msg': 'Invalid number of batches!! batches <= 0',
                'stack': '',
            }
            return build_pipeline_results_dir(
                id_=pipeline_id, train={}, test={},
                extra={}, error=error_dict)
    except Exception as e:
        error_dict = {
            'type': repr(e),
            'msg': str(e),
            'stack': traceback.format_exc(),
        }
        print('ERROR: on evaluate_pipeline_sync', '\n', error_dict)
        return build_pipeline_results_dir(
            id_=pipeline_id, train={}, test={}, extra={}, error=error_dict)


def evaluate_model(model, data, optimizer, epochs=300,
                   n_batches=1, show_info=False):
    # Train
    model, epoch_metrics = \
        train(model, data, optimizer, epochs, n_batches, show_info)
    # Test
    test_metrics = test(model, data, n_batches, show_info)
    return epoch_metrics, test_metrics


class EvaluatorServer(object):

    def __init__(self, n_cpus=1):
        self.manager = multiprocessing.Manager()
        # List containing results as dictionaries
        self.evaluated_list = self.manager.list()
        self.n_cpus = n_cpus if n_cpus >= 0 else multiprocessing.cpu_count()
        self.unprocessed_pipelines = queue.Queue()
        self.start_workers()

    def start_workers(self):
        self.inputs = multiprocessing.Queue()
        # Move elements from unprocessed queue to input queue
        while not self.unprocessed_pipelines.empty():
            elem = self.unprocessed_pipelines.get(block=False)
            self.inputs.put(elem)
        self.processes = [
            multiprocessing.Process(target=evaluate_pipeline,
                                    args=(self.inputs,
                                          self.evaluated_list)
                                    ) for _ in range(self.n_cpus)]
        for p in self.processes:
            p.start()

    def kill_workers(self):
        # Move all unprocessed pipelines from input queue to
        # temporary unprocessed queue
        while True:
            try:
                elem = self.inputs.get(block=False)
                self.unprocessed_pipelines.put(elem)
            except queue.Empty:
                break
        # Join workers
        for p in self.processes:
            # Put in the input queue an exit signal for the worker
            self.inputs.put(['exit', {}])
            p.join()

    def submit(self, pipeline_string):
        # Convert string to dictionary
        pipeline = json.loads(pipeline_string)
        # Generate pipeline ID
        m = md5()
        m.update((pipeline_string).encode())
        pipeline_id = m.hexdigest()
        # Send for processing
        if pause_server:
            print('Server is paused, putting pipeline ', pipeline_id,
                  'in unprocessed_pipelines list')
            self.unprocessed_pipelines.put([pipeline_id, pipeline])
        else:
            print('Putting pipeline ', pipeline_id, 'in input queue')
            self.inputs.put([pipeline_id, pipeline])
        sys.stderr.flush()
        return pipeline_id

    def get_evaluated(self, pipeline_id):
        sys.stderr.flush()
        for pipeline in self.evaluated_list:
            if pipeline['id'] == pipeline_id:
                self.evaluated_list.remove(pipeline)
                pipeline_str = json.dumps(pipeline)
                del pipeline
                gc.collect()
                torch.cuda.empty_cache()
                return pipeline_str

        return json.dumps({})  # If not found, return empty dict

    def get_multiple_evaluated(self, pipeline_ids):
        sys.stderr.flush()
        found_pipelines = []
        for pipeline in self.evaluated_list:
            if pipeline['id'] in pipeline_ids:  # inneficient! O(n)
                found_pipelines.append(json.dumps(pipeline))
                self.evaluated_list.remove(pipeline)
                del pipeline
                gc.collect()
                torch.cuda.empty_cache()

        return json.dumps(found_pipelines)  # If not found, return empty list

    def get_core_count(self):
        return json.dumps(self.n_cpus)

    def get_all_evaluated_pipelines(self):
        return json.dumps(list(map(json.dumps, self.evaluated_list)))

    def resume(self):
        global pause_server
        pause_server = False
        self.start_workers()
        return json.dumps("OK")

    def pause(self):
        global pause_server
        pause_server = True
        self.kill_workers()
        return json.dumps('OK')

    def quit(self):
        global stop_server
        self.kill_workers()
        stop_server = True
        return json.dumps('OK')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Start server that evaluates GNN pipelines.')
    parser.add_argument('-p', '--port-number',
                        required=True, type=int, default=80,
                        help='Port in which the server is supposed to run.')
    parser.add_argument('-n', '--n_cpus',
                        type=int, default=1,
                        help='Number of CPUs to use.')
    return parser.parse_args()


if __name__ == '__main__':
    global stop_server
    global pause_server
    stop_server = False
    pause_server = False
    torch.multiprocessing.set_start_method("forkserver")
    args = parse_args()
    n_cpus = args.n_cpus
    port_number = args.port_number
    server_url = '0.0.0.0'
    # Server Settings
    print('=============== Server Settings:')
    print('Server URL: ', server_url)
    print('Port Number:', port_number)
    print('Num CPUs:', n_cpus)
    print()
    # Initiate variables and start server
    eval_server = EvaluatorServer(n_cpus)
    server = SimpleXMLRPCServer((server_url, port_number))
    server.register_instance(eval_server)
    # Run untill cancellation
    try:
        while not stop_server:
            server.handle_request()
    except Exception as e:
        server.quit()
        print("ERROR: ", str(e))
        print(repr(e))
