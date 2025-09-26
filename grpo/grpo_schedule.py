# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
from deepspeed.runtime.utils import call_to_str
# from ..utils import call_to_str

from abc import ABC, abstractmethod


class PipeSchedule(ABC):
    """Directs the execution of a pipeline engine by generating sequences of
    :class:`PipeInstruction`.

    Schedules are generators that yield sequences of
    :class:`PipeInstruction` to process the micro-batches in one batch.
    Each yielded step is atomic in the sense that a barrier
    synchronization can be placed between successive steps without
    deadlock.

    Below is an example schedule that implements data parallelism with gradient accumulation:

    .. code-block:: python

        class DataParallelSchedule(PipeSchedule):
            def steps(self):
                for step_id in range(self.micro_batches):
                    cmds = [
                        LoadMicroBatch(buffer_id=0),
                        ForwardPass(buffer_id=0),
                        BackwardPass(buffer_id=0),
                    ]
                    if step_id == self.micro_batches - 1:
                        cmds.extend([
                            ReduceGrads(),
                            OptimizerStep(),
                        ])
                    yield cmds

            def num_pipe_buffers(self):
                return 1

    Args:
        micro_batches (int): The number of micro-batches that comprise a batch.
        stages (int): The number of pipeline stages.
        stage_id (int): The pipe stage that will execute the generated schedule.
    """

    def __init__(self, micro_batches, stages, stage_id):
        super().__init__()
        self.micro_batches = micro_batches
        self.stages = stages
        self.stage_id = stage_id
        self.prev_stage = self.stage_id - 1
        self.next_stage = self.stage_id + 1

    @abstractmethod
    def steps(self):
        """Yield a list of :class:`PipeInstruction` for each step in the schedule.

        .. note::
            Schedules must implement ``steps()`` to define the schedule.

        Returns:
            Instructions to be executed as one step of the pipeline
        """
        pass

    def num_pipe_buffers(self):
        """The number of pipeline buffers that will be used by this stage.

        .. note::
            Schedules should specialize ``num_pipe_buffers()`` for memory savings at scale.

        Returns:
            The number of buffers for the engine to allocate.
        """
        return self.micro_batches

    def _valid_micro_batch(self, micro_batch_id):
        return 0 <= micro_batch_id < self.micro_batches

    def _valid_stage(self, stage_id):
        return 0 <= stage_id < self.stages

    @property
    def stage(self):
        """Stage index used to configure this schedule."""
        return self.stage_id

    @property
    def num_stages(self):
        """The number of total pipeline stages used to configure this schedule."""
        return self.stages

    @property
    def num_micro_batches(self):
        """The number of total micro_batches used to configure this schedule."""
        return self.micro_batches

    @property
    def is_first_stage(self):
        """True if the configured ``stage_id`` is the first stage in the pipeline."""
        return self.stage_id == 0

    @property
    def is_last_stage(self):
        """True if the configured ``stage_id`` is the last stage in the pipeline."""
        return self.stage_id == self.stages - 1

    def _buffer_idx(self, micro_batch_id):
        """Map a micro-batch index to a pipeline buffer index.

        This method uses a cyclic allocation strategy.

        Args:
            micro_batch_id (int): The micro-batch index relative to the beginning of the schedule.

        Returns:
            int: The index of the buffer that should store data.
        """
        assert self._valid_micro_batch(micro_batch_id)
        return micro_batch_id % self.num_pipe_buffers()

    def __iter__(self):
        self.it = None
        return self

    def __next__(self):
        if self.it is None:
            self.it = self.steps()
        return next(self.it)

#
# class InferenceSchedule_bk(PipeSchedule):
#     """A schedule for inferencing batches using pipeline parallelism.
#     """
#
#     def steps(self):
#         """"""
#         prev_micro_batch_id = -1
#         total_steps = self.micro_batches + self.stages - 1
#         for step_id in range(total_steps):
#             cmds = []
#             micro_batch_id = step_id - self.stage_id
#
#             # Alternate send/recv buffers
#             if _is_even(self.stage_id):
#                 recv_buf = step_id % 2
#                 send_buf = (step_id + 1) % 2
#             else:
#                 recv_buf = (step_id + 1) % 2
#                 send_buf = step_id % 2
#
#             if self.is_first_stage or self.is_last_stage:
#                 if self._valid_micro_batch(micro_batch_id):
#                     cmds.append(LoadMicroBatch(recv_buf))
#
#             if _is_even(self.stage_id):
#                 if self._valid_stage(self.next_stage):
#                     if self._valid_micro_batch(micro_batch_id - 1):
#                         cmds.append(SendActivation(send_buf))
#                 if self._valid_stage(self.prev_stage):
#                     if self._valid_micro_batch(micro_batch_id):
#                         cmds.append(RecvActivation(recv_buf))
#             else:
#                 if self._valid_stage(self.prev_stage):
#                     if self._valid_micro_batch(micro_batch_id):
#                         cmds.append(RecvActivation(recv_buf))
#
#                 if self._valid_stage(self.next_stage):
#                     if self._valid_micro_batch(micro_batch_id - 1):
#                         cmds.append(SendActivation(send_buf))
#
#             if self._valid_micro_batch(micro_batch_id):
#                 cmds.append(ForwardPass(recv_buf))
#
#             yield cmds
#
#     def num_pipe_buffers(self):
#         """Only two pipeline buffers are required for inferencing.
#
#         Returns:
#             ``2``
#         """
#         return 2


class TrainSchedule(PipeSchedule):
    """A schedule for training a batch using hybrid parallelism.

    Pipeline parallelism is extracted through gradient accumulation and thus
    convergence follows that of a data parallel approach with the same batch
    size.
    """

    def steps(self):
        """"""
        prev_micro_batch_id = -1
        total_steps = 2 * (self.micro_batches + self.stages - 1)
        for step_id in range(total_steps):
            # Map the step of the pipeline to the micro-batch id and also whether it is a
            # forward or backward pass step.
            micro_batch_id, is_forward = self._step_to_micro_batch(step_id)

            # print(
            #     f"TrainSchedule_1: {self.stage_id}, step_id: {step_id}, micro_batch_id: {micro_batch_id}, is_forward:{is_forward}")

            #for reward
            if self.stage_id == 0:
                lag_micro_batch_id, lag_is_forward = self._step_to_micro_batch(step_id - self.num_stages + 1)
            # if self.stage_id == self.stages - 1:
            #     future_micro_batch_id, future_is_forward = self._step_to_micro_batch(step_id + self.num_stages - 1)

            if self._valid_micro_batch(prev_micro_batch_id):
                prev_buffer = self._buffer_idx(prev_micro_batch_id)
            if self._valid_micro_batch(micro_batch_id):
                curr_buffer = self._buffer_idx(micro_batch_id)

            #for reward
            if self.stage_id == 0:
                if self._valid_micro_batch(lag_micro_batch_id):
                    lag_curr_buffer = self._buffer_idx(lag_micro_batch_id)

            # if self.stage_id == self.stages - 1:
            #     if self._valid_micro_batch(future_micro_batch_id):
            #         future_curr_buffer = self._buffer_idx(future_micro_batch_id)

            cmds = []

            # last stage pre-fetch receive labels from stage0
            #for reward
            if self.stage_id == 0:
                if lag_is_forward and self._valid_micro_batch(lag_micro_batch_id):
                    cmds.append(SendRwd(lag_curr_buffer))


            # if self.stage_id == self.stages - 1:
            #     if future_is_forward and self._valid_micro_batch(future_micro_batch_id):
            #         cmds.append(RecvRwd(future_curr_buffer))


            # Exchange activations
            if is_forward:
                if self._valid_micro_batch(prev_micro_batch_id) and self._valid_stage(self.prev_stage):
                    cmds.append(SendGrad(prev_buffer))
                if self._valid_micro_batch(micro_batch_id) and self._valid_stage(self.prev_stage):
                    cmds.append(RecvActivation(curr_buffer))
            else:
                if self._valid_micro_batch(micro_batch_id) and self._valid_stage(self.next_stage):
                    cmds.append(RecvGrad(curr_buffer))
                if self._valid_micro_batch(prev_micro_batch_id) and self._valid_stage(self.next_stage):
                    cmds.append(SendActivation(prev_buffer))

            # First/last stage loads
            if self.stage_id == 0 or self.stage_id == self.stages - 1:
                if is_forward and self._valid_micro_batch(micro_batch_id):
                    cmds.append(LoadMicroBatch(curr_buffer))

            if self.stage_id == self.stages - 1:
                if is_forward and self._valid_micro_batch(micro_batch_id):
                    cmds.append(RecvRwd(curr_buffer))

            # #for reward
            # if self.stage_id == self.stages - 1:
            #     if is_forward and self._valid_micro_batch(micro_batch_id):
            #         cmds.append(SendRwd(curr_buffer))

            # Computation
            if self._valid_micro_batch(micro_batch_id):
                if is_forward:
                    cmds.append(ForwardPass(curr_buffer))
                else:
                    cmds.append(BackwardPass(curr_buffer))

            # Model step at the end of the batch
            if step_id == total_steps - 1:
                cmds.append(ReduceTiedGrads())
                cmds.append(ReduceGrads())
                cmds.append(OptimizerStep())

            # # First stage send advantage(labels) data to last stage
            # if self.stage_id == 0:
            #     if is_forward and self._valid_micro_batch(micro_batch_id):
            #         cmds.append(SendRwd(curr_buffer))

            # need to determine receive



            # if self.stage_id >= 0:
            #     print(f"TrainSchedule_2  stage: {self.stage_id}, step: {step_id}, cmds: {cmds}")
            # Prepare state for next time
            prev_micro_batch_id = micro_batch_id
            yield cmds

    def num_pipe_buffers(self):
        """Return the number of pipeline buffers required for this stage.

        This is equivalent to the maximum number of in-flight forward passes,
        since we need to remember the activations of forward passes in order
        to run backpropagation. For synchronous 1F1B, this is equivalent to
        the index difference between this stage and the last stage.
        """
        buffers = min(self.stages - self.stage_id, self.micro_batches)
        return max(2, buffers)

    def _step_to_micro_batch(self, step_id):
        if _is_even(step_id) and _is_even(self.stage_id):
            micro_batch_id = self._even_step_forward_id(step_id)
            is_forward = True

        elif _is_odd(step_id) and _is_odd(self.stage_id):
            micro_batch_id = self._odd_step_forward_id(step_id)
            is_forward = True

        elif _is_even(step_id) and _is_odd(self.stage_id):
            micro_batch_id = self._even_step_backward_id(step_id)
            is_forward = False

        elif _is_odd(step_id) and _is_even(self.stage_id):
            micro_batch_id = self._odd_step_backward_id(step_id)
            is_forward = False

        else:
            assert False

        return micro_batch_id, is_forward

    def _even_step_forward_id(self, step_id):
        base = step_id // 2
        micro_batch_id = int(base - self.stage_id // 2)
        return micro_batch_id

    def _odd_step_forward_id(self, step_id):
        base = (step_id - 1) // 2
        micro_batch_id = int(base - self.stage_id // 2)
        return micro_batch_id

    def _even_step_backward_id(self, step_id):
        base = step_id // 2
        micro_batch_id = int(base - self.stages + (self.stage_id + 1) // 2)
        return micro_batch_id

    def _odd_step_backward_id(self, step_id):
        base = ((step_id - 1) // 2) - self.stages + 1
        micro_batch_id = int(base + self.stage_id // 2)
        return micro_batch_id


class DataParallelSchedule(PipeSchedule):
    """An example schedule that trains using traditional data parallelism with gradient
    accumulation.
    """

    def steps(self):
        """"""
        for step_id in range(self.micro_batches):
            cmds = [
                LoadMicroBatch(buffer_id=0),
                ForwardPass(buffer_id=0),
                BackwardPass(buffer_id=0),
            ]
            if step_id == self.micro_batches - 1:
                cmds.extend([
                    ReduceGrads(),
                    OptimizerStep(),
                ])
            yield cmds

    def num_pipe_buffers(self):
        """Only one pipeline buffer needed.
        """
        return 1


class PipeInstruction:
    """Base class for all instructions to be executed by the pipeline engine.

    All keyword arguments are stored as members similar to a ``namedtuple``. These are
    then accessible to the :class:`PipeEngine` during execution.

    Args:
        kwargs (optional): keyword arguments to store as members
    """

    def __init__(self, **kwargs):
        self.name = self.__class__.__name__
        self.kwargs = kwargs
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __repr__(self):
        return call_to_str(self.name, **self.kwargs)


class OptimizerStep(PipeInstruction):
    """Performs one step with the optimizer and zeros gradients.

    .. note:: Should be issued after :class:`ReduceGrads` and :class:`ReduceTiedGrads`.

    .. note:: Can be a synchronization point among data-parallel ranks.
    """
    pass


class ReduceGrads(PipeInstruction):
    """Reduce the computed gradients among data-parallel processes within the stage.
    """
    pass


class ReduceTiedGrads(PipeInstruction):
    """Reduce the computed gradients of tied modules within a pipeline-parallel group.

    .. warning::
        The stages included in this synchronization point are not known until
        the model is partitioned among pipeline stages. In the worst case, it
        includes all pipeline stages. This instruction should be scheduled
        carefully to avoid deadlocks.
    """
    pass


class BufferOpInstruction(PipeInstruction):
    """A pipeline instruction that operates on pipeline buffer(s).

    Args:
        buffer_id (int): the index of the pipeline buffer() to modify.
    """

    def __init__(self, buffer_id, **kwargs):
        super().__init__(buffer_id=buffer_id, **kwargs)


# IO
class LoadMicroBatch(BufferOpInstruction):
    """Load a micro-batch into a buffer.

    Roughly:

    .. code-block:: python

        buffers['inputs'][buffer_id] = next(data_iter)
    """
    pass


# Compute
class ForwardPass(BufferOpInstruction):
    """Compute a forward pass.

    Roughly:

    .. code-block:: python

        buffers['outputs'][buffer_id] = forward(buffers['inputs'][buffer_id])
    """
    pass


class BackwardPass(BufferOpInstruction):
    """Compute a backward pass and accumulate gradients.

    Roughly:

    .. code-block:: python

        outputs = buffers['outputs'][buffer_id]
        gradients = buffers['gradients'][buffer_id]
        torch.autograd.backward(tensors=outputs,
                                grad_tensors=gradients)
    """
    pass


# Communication
class SendActivation(BufferOpInstruction):
    """Send activations to the next stage in the pipeline.

    Roughly:

    .. code-block:: python

        send(buffers['outputs'][buffer_id])

    .. note::
        The communication is blocking and must be paired with a :class:`RecvActivation`
        on the next pipeline stage to avoid deadlock.
    """
    pass


class RecvActivation(BufferOpInstruction):
    """Receive activations from the previous stage in the pipeline.

    Roughly:

    .. code-block:: python

        buffers['inputs'][buffer_id] = recv()

    .. note::
        The communication is blocking and must be paired with a :class:`SendActivation`
        on the previous pipeline stage to avoid deadlock.
    """
    pass


class SendGrad(BufferOpInstruction):
    """Send computed gradients to the previous pipeline stage.
    with respect to the received activations

    .. note::
        Only received tensors with ``requires_grad==True`` will produce gradients.
        Missing gradients will be replaced with ``None`` on the receiving stage.

    .. note::
        The communication is blocking and must be paired with a :class:`RecvGrad`
        on the previous pipeline stage to avoid deadlock.
    """
    pass


class RecvGrad(BufferOpInstruction):
    """Receive computed gradients the next pipeline stage.

    .. note::
        Only activations with ``requires_grad==True`` will produce gradients.
        Missing gradients will be replaced with ``None``.

    .. note::
        The communication is blocking and must be paired with a :class:`SendGrad`
        on the next pipeline stage to avoid deadlock.
    """
    pass

class SendRwd(BufferOpInstruction):
    pass

class RecvRwd(BufferOpInstruction):
    pass




def _is_even(x):
    return x % 2 == 0


def _is_odd(x):
    return x % 2 != 0




class InferenceSchedule(PipeSchedule):
    """A schedule for inferencing batches using pipeline parallelism.
    """

    def steps(self):
        """"""
        prev_micro_batch_id = -1
        total_steps = self.micro_batches + self.stages - 1
        for step_id in range(total_steps):
            cmds = []
            micro_batch_id = step_id - self.stage_id

            #for reward
            if self.stage_id == 0:
                lag_micro_batch_id = step_id - self.num_stages + 1 - self.stage_id

            # Alternate send/recv buffers
            if _is_even(self.stage_id):
                recv_buf = step_id % 2
                send_buf = (step_id + 1) % 2
            else:
                recv_buf = (step_id + 1) % 2
                send_buf = step_id % 2

            if self.stage_id == 0:
                first_stage_recv_buf = step_id % self.num_pipe_buffers()
                first_stage_send_buf = (step_id - 1) % self.num_pipe_buffers()
                first_stage_lag_reward_buf = (step_id - self.num_stages + 1) % self.num_pipe_buffers()


            # if _is_even(self.stage_id):
            #     lag_recv_buf = (step_id - self.num_stages + 1) % 2
            #     lag_send_buf = (step_id - self.num_stages + 1 + 1) % 2
            # else:
            #     lag_recv_buf = (step_id - self.num_stages + 1 + 1) % 2
            #     lag_send_buf = (step_id - self.num_stages + 1 )% 2

            # print(
            #     f"InferSchedule_1: {self.stage_id}, step_id: {step_id}, micro_batch_id: {micro_batch_id}, recv_buf:{recv_buf}, send_buf:{send_buf}, lag_recv_buf:{lag_recv_buf}, lag_send_buf:{lag_send_buf}")


            #send reward
            if self.is_first_stage:
                if self._valid_micro_batch(lag_micro_batch_id):
                    cmds.append(SendRwd(first_stage_lag_reward_buf))

            if self.is_last_stage:
                if self._valid_micro_batch(micro_batch_id):
                    cmds.append(LoadMicroBatch(recv_buf))

            #newly added for reward send
            if self.is_first_stage:
                if self._valid_micro_batch(micro_batch_id):
                    cmds.append(LoadMicroBatch(first_stage_recv_buf))



            #receive reward
            if self.stage_id == self.stages - 1:
                if self._valid_micro_batch(micro_batch_id):
                    cmds.append(RecvRwd(recv_buf))

            if _is_even(self.stage_id):
                if self._valid_stage(self.next_stage):
                    if self._valid_micro_batch(micro_batch_id - 1) and self.stage_id > 0: #change by luke
                        cmds.append(SendActivation(send_buf))
                if self._valid_stage(self.prev_stage):
                    if self._valid_micro_batch(micro_batch_id):
                        cmds.append(RecvActivation(recv_buf))
            else:
                if self._valid_stage(self.prev_stage):
                    if self._valid_micro_batch(micro_batch_id):
                        cmds.append(RecvActivation(recv_buf))

                if self._valid_stage(self.next_stage):
                    if self._valid_micro_batch(micro_batch_id - 1):
                        cmds.append(SendActivation(send_buf))

            #newly added
            if self.stage_id == 0:
                if self._valid_micro_batch(micro_batch_id - 1) and self.stage_id == 0:
                    cmds.append(SendActivation(first_stage_send_buf))

            if self._valid_micro_batch(micro_batch_id) and self.stage_id > 0:  #chaned by luke
                cmds.append(ForwardPass(recv_buf))
            elif self._valid_micro_batch(micro_batch_id) and self.stage_id == 0:
                cmds.append(ForwardPass(first_stage_recv_buf))


            # print(f"InferSchedule_2, stage: {self.stage_id}, step:{step_id}, cmd: {cmds}")

            yield cmds

    def num_pipe_buffers_bk(self):
        """Only two pipeline buffers are required for inferencing.

        Returns:
            ``2``
        """

        return 2

    # def num_pipe_label_reward_buffers(self):
    #     if self.stage_id == 0:
    #         buffers = min(self.stages, self.micro_batches)
    #         num = max(2, buffers)
    #
    #     else:
    #         num = 2
    #
    #     #but only first and last stage is effective, none sense for other stages
    #     return num

    def num_pipe_buffers(self):
        """Only two pipeline buffers are required for inferencing.

        Returns:
            ``2``
        """
        if self.stage_id == 0:
            buffers = min(self.stages, self.micro_batches)
            num = max(2, buffers)

        else:
            num = 2

        #but only first and last stage is effective, none sense for other stages
        return num
