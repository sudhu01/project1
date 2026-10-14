"""Timed raw observation windows. Recovery starts after historical collection."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from time import monotonic, sleep
from threading import Event
import requests
from .backends.evidence_v2 import collect_logs


def collect_window(driver, workload, stats, registry, seconds=120, interval=2, expected_states=None, ingestion_seconds=10):
    if seconds != 120 or interval not in (.1, 1, 2):
        raise ValueError('unregistered capture window or workload interval')
    start = datetime.now(timezone.utc)
    end = start + timedelta(seconds=seconds)
    deadline = monotonic() + seconds
    cancelled = Event()
    workload.cart_observations = []
    raw = {'stats': [], 'requests': [], 'status': {}, 'workload_errors': []}
    if expected_states is None:
        expected_states = driver.states()
    for resource in registry:
        row = driver.state(resource)
        if start <= datetime.fromisoformat(row['sample_time']) <= end:
            raw['status'][resource] = row
    with ThreadPoolExecutor(max_workers=1) as executor:
        def traffic():
            while monotonic() < deadline and not cancelled.is_set():
                try:
                    row = workload.checkout()
                    if start <= datetime.fromisoformat(row['sample_time']) <= end:
                        raw['requests'].append(row)
                except (RuntimeError, requests.RequestException) as exc:
                    # Preparation failures are collector/workload errors, not invented checkout 500s.
                    raw['workload_errors'].append({'sample_time': datetime.now(timezone.utc).isoformat(), 'reason': type(exc).__name__})
                cancelled.wait(min(interval, max(0, deadline - monotonic())))
        future = executor.submit(traffic)
        try:
            while monotonic() < deadline:
                began = monotonic()
                states = driver.states()
                if states != expected_states:
                    raise RuntimeError('resource state changed during raw capture')
                rows = stats.poll()
                raw['stats'].extend(row for row in rows if start <= datetime.fromisoformat(row['sample_time']) <= end)
                sleep(min(max(0, 2 - (monotonic() - began)), max(0, deadline - monotonic())))
            future.result()
        finally:
            cancelled.set()
    raw['requests'].extend(row for row in workload.cart_observations if start <= datetime.fromisoformat(row['sample_time']) <= end)
    raw['workload_errors'] = [row for row in raw['workload_errors'] if start <= datetime.fromisoformat(row['sample_time']) <= end]
    if type(ingestion_seconds) is not int or not 0 <= ingestion_seconds <= 15:
        raise ValueError('unbounded telemetry ingestion allowance')
    sleep(ingestion_seconds)
    with requests.Session() as session:
        raw['logs'] = collect_logs(session, registry, start.isoformat(), end.isoformat())
    raw['collected_at'] = datetime.now(timezone.utc).isoformat()
    raw['registry'] = registry
    return {'window_start': start.isoformat(), 'window_end': end.isoformat()}, raw
