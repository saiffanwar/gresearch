def assign_tasks(factor, arrival, bonus, reward, duration, time_bonus):
    """Assign tasks to processors."""
    """Simple dumb strategy, guaranteed to produce valid solutions."""
    """All tasks are assigned to processor 0, each are scheduled"""
    """max(duration) ticks apart."""
    num_tasks = len(factor)
    max_arrival = max(arrival)
    max_duration = max(duration)
    schedule = []
    for i in range(num_tasks):
        schedule.append((0, max_arrival + i * max_duration))
    return schedule

