"""asdfasdfasdf"""
import time
import requests


class OrcaTask:
    """
    Represents a single task for orca on pcss quantum api

    __init__
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
            'machine': None,
            'extra_options': kwargs
        }

        self.uid = None
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
            raise ValueError("API did not return task UID")

    def submit(self):
        """asdfasdfasdf"""

        if self._submitted:
            raise ValueError(f"Cannot submit the same task twice, {self.uid}")

        response = requests.post(
            self._full_url(f'tasks/{self.uid}/submit'),
            headers=self._get_headers(),
            timeout=5
        )
        response.raise_for_status()
        response_data = response.json()

        response_job_ids = response_data.get('job_ids', {'ids': []})['ids']
        if len(response_job_ids) == 0:
            raise RuntimeError("Failed to create job for task.")

        self._submitted = True
        self._job_ids += response_job_ids

    def _try_get_results(self):
        if not self._submitted:
            raise ValueError(f"Submit a task first to get results. {self.uid}")

        response = requests.get(
            self._full_url(f'tasks/{self.uid}/results'),
            headers=self._get_headers(),
            timeout=5
        )

        response.raise_for_status()
        response_data = response.json()

        return response_data

    def results(self):
        """asdfasdfasdf"""

        if self._results != []:
            return self.results

        res = self._try_get_results()
        while res == {}:
            time.sleep(0.05)
            res = self._try_get_results()

        self._results = res
        return res

    @property
    def status(self):
        """asdfasdfasdf"""

        if not self._submitted:
            return 'NEW'
        if self._submitted and not len(self._results) > 0:
            return 'RUNNING'
        return 'COMPLETED'
