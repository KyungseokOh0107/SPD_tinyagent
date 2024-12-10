import requests

query = "Add date schedule to 2024.12.25"
response = requests.post('http://127.0.0.1:50001/generate', json={'query': query})