"""Methods for communicating with psnc quantum api"""
import time
import requests


class OrcaTask:
    """
    Represents a single task for orca on psnc quantum api.
    """

    API_BASE_URL = 'https://api.quantum.psnc.pl/api/client'

    def __init__(
            self,
            input_state,
            bs_angles,
            loop_lengths,
            machine,
            auth_token,
            n_samples=200,
            postselection=False,
            postselection_threshold=None,
            **kwargs) -> None:

        self.machine = machine
        self.auth = auth_token
        self.task_payload = {
            'input_state': input_state,
            'bs_angles': bs_angles,
            'n_samples': n_samples,
            'loop_lengths': loop_lengths,
            'postselection': postselection,
            'postselection_threshold': postselection_threshold,
            'machine': machine,
            'extra_options': kwargs
        }

        self.uid: str
        self._job_ids = []
        self._results = []

        self._created_time = None

        self._submitted = False

        self._create_on_remote()

    def _full_url(self, rel):
        return f"{OrcaTask.API_BASE_URL}/{rel}"

    def _get_headers(self):
        return {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.auth}'
        }

    def _create_on_remote(self):
        response = requests.post(
            self._full_url('tasks'),
            headers=self._get_headers(),
            json={
                'machine': self.machine,
                'payload': self.task_payload
            },
            timeout=5
        )

        response.raise_for_status()

        response_data = response.json()

        self.uid = response_data.get('uid', None)
        self._created_time = response_data.get('created', None)

        if self.uid is None:
            raise ValueError('API did not return task UID')

    def _try_get_results(self):

        response = requests.get(
            self._full_url(f'tasks/{self.uid}/results'),
            headers=self._get_headers(),
            timeout=5
        )

        if not response.ok:
            return {}

        response_data = response.json()

        return response_data

    def results(self) -> list[str]:
        """
        Get task results. Blocks the thread until results are available.

        Returns:
            list[str]: List of bitstrings obtained from runs.
        """
        if self._results != []:
            return self._results

        res = self._try_get_results()
        while res == {}:
            time.sleep(0.05)
            res = self._try_get_results()

        if not isinstance(res, list):
            raise ValueError(f'invalid result type: {type(res)}, {res}')

        self._results = res
        return res

    @property
    def status(self):
        """Task status"""
        response = requests.get(
            self._full_url(f'tasks/{self.uid}/status'),
            headers=self._get_headers(),
            timeout=5
        )

        response.raise_for_status()

        response_data = response.json()

        return response_data.get('status', 'Unknown')
