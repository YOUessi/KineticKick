"""Keep the frozen running/strike/recovery acceptance conjunction intact."""


def full_task_success(running,stride,sequence,handoffs):
    return bool(running['running_success'] and handoffs and stride['continuous_running_contact'] and sequence['sequence_success'])
