'''

Integration Testing Script

Launch the server before running this.

'''

import requests
import json

PORT = 8000
URL = f'http://localhost:{PORT}'


def test_local_llm_stream():
    url = f"{URL}/local_llm_stream"
    data = {
        "text": "How are you?",
        "max_length": 50,
        "do_sample": True,
        "top_k": 10,
        "num_return_sequences": 1
    }
    response = requests.post(url, json=data, stream=True)
    assert response.status_code == 200
    for line in response.iter_lines():
        if line:  # filter out keep-alive new lines
            print(json.loads(line))


test_local_llm_stream()
