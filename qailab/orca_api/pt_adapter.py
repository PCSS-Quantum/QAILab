""" PT tbi Adapter """
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAlias

from ptseries.tbi.pt import PT
from ptseries.tbi.tbi_abstract import TBIDevice
from qailab.orca_api.networking import OrcaTask

if TYPE_CHECKING:
    FILE_LIKE: TypeAlias = str | os.PathLike


class PTAdapter(PT):
    """ Adapter for PT tbi """
    # pylint:disable=too-few-public-methods

    def __init__(  # pylint: disable=super-init-not-called
        self,
        n_loops: int = 1,
        loop_lengths: list[int] | tuple[int, ...] | npt.NDArray[np.int_] | None = None,
        postselected: bool | None = None,
        postselection: bool = True,
        postselection_threshold: int | None = None,
        ip_address: str | None = None,
        url: str | None = None,
        machine: str | None = None,
        secret_key: str = '',
        **kwargs,
    ):
        TBIDevice.__init__(  # pylint: disable=non-parent-init-called
            self,
            n_loops=n_loops,
            loop_lengths=loop_lengths,
            postselected=postselected,
            postselection=postselection,
            postselection_threshold=postselection_threshold,
            ip_address=ip_address,
            url=url,
            machine=machine,
        )

        self.pt_kwargs = kwargs
        self.secret_key = secret_key

        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        self.sample_async_flag = False

    def _submit_job_async(
        self,
        input_state: list[int] | tuple[int, ...],
        bs_angles: list[float] | tuple[float, ...],
        n_samples: int,
    ) -> str:
        """Prepares and sends sample request to PT.

        Args:
            input_state: description of input modes. The left-most entry corresponds to the first mode entering the loop.
            bs_angles: list of beam splitter angles
            n_samples: number of samples to draw. Defaults to 1.
        """
        task = OrcaTask(input_state, bs_angles,
                        self.loop_lengths, self.secret_key, self.machine, n_samples,
                        self.postselection, self.postselection_threshold,
                        **self.pt_kwargs)
        return task.uid

    def _request_samples(
        self,
        input_state: list[int] | tuple[int, ...],
        bs_angles: list[float] | tuple[float, ...],
        n_samples: int,
        # TODO: figure out if we should do anything with this arg
        save_dir: FILE_LIKE | None = None,  # pylint:disable=unused-argument
    ) -> npt.NDArray[np.int_]:
        """Prepares and sends sample request to PT.

        Args:
            input_state: description of input modes.
                The left-most entry corresponds to the first mode entering the loop.
            bs_angles: list of beam splitter angles
            n_samples: number of samples to draw. Defaults to 1.
            save_dir: Path to the directory in which to save results. If set to None the results are not saved. Defaults
                to None.
        """
        task = OrcaTask(input_state, bs_angles,
                        self.loop_lengths, self.machine, self.secret_key, n_samples,
                        self.postselection, self.postselection_threshold,
                        **self.pt_kwargs)
        result_json = task.results()
        samples = result_json
        samples = self._reformat_samples(samples)
        return samples
