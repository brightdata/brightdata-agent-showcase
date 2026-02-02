import requests
import json
import os
from time import sleep

POST_TIMEOUT = (10, 600)  # connect, read
GET_TIMEOUT  = (10, 60)
OUTPUT_FOLDER = "output"

class AIModelRetriever:
    
    def __init__(self, api_token: str):
        self.api_token = api_token

    def get_model_response(self, model: AIModel, prompt: str):
        snapshot_id = self.trigger_prompt_collection(model, prompt)
        if not snapshot_id:
            raise RuntimeError(f"{model.name}: failed to trigger snapshot. Please wait and try again.")
        llm_response = self.collect_snapshot(model, snapshot_id)
        if not llm_response:
            raise RuntimeError(f"Failed to collect snapshot {snapshot_id} for {model.name}. Please wait and try again")
        self.write_model_output(model, llm_response)


    def trigger_prompt_collection(self, model: AIModel, prompt: str, country: str = ""):
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }
        data = json.dumps(
            {"input": 
                [
                    {
                        "url": model.url,
                        "prompt": prompt,
                        #"country":country,
                    }
                ],
            })
        tries = 3

        while tries > 0:
            response = None
            try:
                response = requests.post(
                    f"https://api.brightdata.com/datasets/v3/trigger?dataset_id={model.dataset_id}",
                    headers=headers,
                    data=data,
                    timeout=POST_TIMEOUT
                )
                response.raise_for_status()
                payload = response.json()
                snapshot_id = payload["snapshot_id"]
                return snapshot_id

            except (ValueError, KeyError, TypeError, requests.RequestException) as e:
                print(f"failed to trigger {model.name} snapshot: {e}")
                tries -= 1
                if response is not None and response.status_code >= 400:
                    print(f"Status: {response.status_code}")
                    print(response.text)
        
        print("retries exceeded")
        return
    
    
    def collect_snapshot(self, model: AIModel, snapshot_id: str):
        snapshot_url = f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}"
        headers = {"Authorization": f"Bearer {self.api_token}"}

        print(f"Waiting for {model.name} snapshot {snapshot_id}")
        max_errors = 3

        with requests.Session() as session:
            while max_errors > 0:
                try:
                    resp = session.get(snapshot_url, headers=headers, timeout=GET_TIMEOUT)
                    resp.raise_for_status()
                    json_response = resp.json()
                    status = json_response.get("status")
                    print(f"{model.name} Status: {status}")
                except (requests.RequestException, ValueError) as e:
                    max_errors -= 1
                    print(f"{model.name}: polling error ({e[:500]})")
                    continue

                if "answer_text" in json_response.keys():
                    print(f"{model.name} snapshot {snapshot_id} is ready!")
                    # For many datasets, this same endpoint with ?format=json returns the data
                    data_resp = session.get(
                        f"{snapshot_url}?format=json",
                        headers=headers,
                        timeout=GET_TIMEOUT,
                    )
                    try:
                        data_resp.raise_for_status()
                    except requests.HTTPError as e:
                        print(f"Download error for {model.name} {snapshot_id}: {e[:500]}")
                        return None

                    llm_response = data_resp.json()
                    return llm_response

                elif status in ("building", "collecting", "running", "starting"):
                    # Snapshot still in progress
                    sleep(10)

                elif status == "failed":
                    print(f"{model.name} snapshot {snapshot_id} failed")
                    return None

                else:
                    max_errors -= 1
                    print(f"Unexpected status for {model.name} {snapshot_id}: {status}")
                    
        print(f"Max errors exceeded, {model.name} snapshot {snapshot_id} could not be collected")
        return None

    
    def write_model_output(self, model: AIModel, llm_response: dict):
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        path = os.path.join(OUTPUT_FOLDER, f"{model.name}-output.json")

        with open(path, "w", encoding="utf-8") as file:
            json.dump(llm_response, file, indent=4, ensure_ascii=False)
            print(f"Finished generating report from {model.name} → {path}")

    def run(self, model: "AIModel", prompt: str, country: str = "") -> dict:
        tries = 3
        snapshot_id = None
        while not snapshot_id and tries > 0:
            try:
                snapshot_id = self.trigger_prompt_collection(model, prompt, country=country)
                if not snapshot_id:
                    raise RuntimeError(f"{model.name}: failed to trigger snapshot. Please retry.")
            except Exception as e:
                tries-=1
        
        llm_response = None
        tries = 3
        while not llm_response and tries > 0:
            try:
                llm_response = self.collect_snapshot(model, snapshot_id)
                if not llm_response:
                    raise RuntimeError(f"{model.name}: failed to collect snapshot {snapshot_id}. Please retry.")
            except Exception as e:
                tries-=1
        return llm_response
    


class AIModel:
    def __init__(self, name: str, dataset_id: str, url: str):
        self.name = name
        self.dataset_id = dataset_id
        self.url = url    
    


chatgpt = AIModel(
    name="ChatGPT",
    dataset_id="gd_m7aof0k82r803d5bjm",
    url="https://chatgpt.com/"
)

perplexity = AIModel(
    name="Perplexity",
    dataset_id="gd_m7dhdot1vw9a7gc1n",
    url="https://www.perplexity.ai"
)

gemini = AIModel(
    name="Gemini",
    dataset_id="gd_mbz66arm2mf9cu856y",
    url="https://gemini.google.com/"
)

grok = AIModel(
    name="Grok",
    dataset_id="gd_m8ve0u141icu75ae74",
    url="https://grok.com/"
)

copilot = AIModel(
    name="CoPilot",
    dataset_id="gd_m7di5jy6s9geokz8w",
    url="https://copilot.microsoft.com/chats"
)