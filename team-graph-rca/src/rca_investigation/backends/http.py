import json


def bounded_json(session, url, params, maximum_bytes=1048576):
    response = session.get(url, params=params, timeout=(3, 5), stream=True)
    try:
        response.raise_for_status()
        content = bytearray()
        for chunk in response.iter_content(chunk_size=8192):
            content.extend(chunk)
            if len(content) > maximum_bytes:
                raise ValueError('telemetry response exceeded byte limit')
        return json.loads(content)
    finally:
        response.close()
