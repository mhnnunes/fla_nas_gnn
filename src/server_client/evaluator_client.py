import json
import time
from xmlrpc.client import ServerProxy


class EvaluatorClient(object):
    """Opens an HTTP connection with the pipeline evaluator server.

    Arguments:
        server_url (str): URL, including port, of the server.
    """

    def __init__(self, server_url):
        self.server_url = server_url
        self.server_proxy = ServerProxy(server_url)

    def build_pipeline(self, actions, search_space, dataset_name,
                       random_seed, n_folds, n_batches,
                       lr=0.005, in_drop=0.6, weight_decay=5e-4, layers=2,
                       epochs=300):
        pipeline = {}
        pipeline['candidate_arch'] = actions
        pipeline['search_space'] = search_space
        pipeline['dataset_name'] = dataset_name
        pipeline['random_seed'] = random_seed
        pipeline['number_of_folds'] = n_folds
        pipeline['batches'] = n_batches
        pipeline['learning_rate'] = lr
        pipeline['in_drop'] = in_drop
        pipeline['weight_decay'] = weight_decay
        pipeline['layers'] = layers
        pipeline['epochs'] = epochs
        return pipeline

    def evaluate_pipeline_async(self, pipeline):
        pipeline_string = json.dumps(pipeline)
        return self._submit(pipeline_string)

    def evaluate_pipeline(self, pipeline):
        pipeline_string = json.dumps(pipeline)
        pipeline_id = self._submit(pipeline_string)
        return self._get_evaluated(pipeline_id)

    def _submit(self, pipeline_string):
        return self.server_proxy.submit(pipeline_string)

    def _get_evaluated(self, pipeline_id):
        attempts = 0
        step = 2
        results = {}
        results = json.loads(self.server_proxy.get_evaluated(pipeline_id))
        while 'id' not in results or results['id'] != pipeline_id:
            time.sleep(2)
            attempts += step
            results = json.loads(self.server_proxy.get_evaluated(pipeline_id))
        if results['error'] != {}:  # error dictionary is not empty == ERROR
            print("ERROR evaluating pipeline:", results['error'])
        return results

    def _get_evaluated_multiple(self, pipeline_ids, verbose=False):
        results = []
        query_pipelines = pipeline_ids.copy()
        if verbose:
            print('querying pipelines: ', query_pipelines)
        query_results = list(
            map(json.loads, json.loads(
                self.server_proxy.get_multiple_evaluated(
                    json.dumps(query_pipelines)))))
        if verbose:
            print('query results: ', query_results)
        while len(query_pipelines) > 0:
            if len(query_results) > 0:
                for pipeline in query_results:
                    # delete pipeline from query pipelines
                    for pipeline_id in query_pipelines:
                        if pipeline_id == pipeline['id']:
                            query_pipelines.remove(pipeline_id)
                    # move pipeline to results
                    results.append(pipeline)
                    del pipeline
            time.sleep(2)
            if verbose:
                print('querying pipelines: ', query_pipelines)
            query_results = list(
                map(json.loads, json.loads(
                    self.server_proxy.get_multiple_evaluated(
                        json.dumps(query_pipelines)))))
            if verbose:
                print('query results: ', query_results)
        return results

    def _get_evaluated_async(self, pipeline_id):
        results = {}
        results = json.loads(self.server_proxy.get_evaluated(pipeline_id))
        if results['error'] != {}:  # error dictionary is not empty == ERROR
            print("ERROR evaluating pipeline:", results['error'])
        return results

    def _get_all_evaluated_pipelines(self):
        return list(map(json.loads, json.loads(
            self.server_proxy.get_all_evaluated_pipelines())))

    def quit(self):
        return self.server_proxy.quit()

    def pause_server(self):
        return self.server_proxy.pause()

    def resume_server(self):
        return self.server_proxy.resume()
