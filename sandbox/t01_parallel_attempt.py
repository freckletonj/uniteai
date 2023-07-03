import requests
import time


# replace 'localhost:8000' with your server address if it's different
url_start = "http://localhost:8000/start_stream"
url_stop = "http://localhost:8000/stop_stream"
url_inf = "http://localhost:8000/infinite"

# Test streaming endpoint
def test_stream_endpoint():
    response = requests.get(url_start, stream=True)

    # Check status code
    assert response.status_code == 200

    # Check data streaming
    data = []
    for line in response.iter_lines():
        if line:
            x = line.decode()
            print(x)
            data.append(x)
    assert len(data) > 0, "No data streamed"

# Test stop stream endpoint
def test_stop_stream():
    response = requests.get(url_stop)

    # Check status code
    assert response.status_code == 200

    # Check response body
    assert response.json() == {"message": "Stream stopped"}

# Test behavior
def test_behavior():
    print('Test Stream Endpoint')
    test_stream_endpoint()
    time.sleep(2)  # let's wait for the stream to flow for some time

    print('Test Stop Endpoint')
    test_stop_stream()  # stopping the stream
    time.sleep(2)  # let's wait for the stop signal to take effect


    print('Test Stream Endpoint')
    test_stream_endpoint()
    time.sleep(10)  # let's wait for the stream to flow for some time

    print('Test Stop Endpoint')
    test_stop_stream()  # stopping the stream
    # time.sleep(2)  # let's wait for the stop signal to take effect

def test_cancel():
    response = requests.get(url_inf, stream=True)
    for line in response.iter_lines():
        if line:
            x = line.decode()
            print(x)


# Go
test_behavior()
