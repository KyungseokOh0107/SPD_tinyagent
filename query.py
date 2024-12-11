import requests

query = "Create a meeting with Sid and Lutfi for tomorrow 2pm to discuss the meeting notes."
response = requests.post('http://127.0.0.3:50001/generate', json={'query': query})