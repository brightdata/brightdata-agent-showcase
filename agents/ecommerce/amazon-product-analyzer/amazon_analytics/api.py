"""Bright Data API integration with retry logic."""
import requests
import json
import logging
from typing import Dict, List, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import BRIGHT_DATA_TOKEN, BRIGHT_DATA_DATASET_ID


class BrightDataAPIError(Exception):
    """Custom exception for Bright Data API errors."""
    pass


class BrightDataAPI:
    """Handles all interactions with the Bright Data API."""
    
    def __init__(self, token: Optional[str] = None):
        self.token = token or BRIGHT_DATA_TOKEN
        if not self.token:
            raise BrightDataAPIError("BRIGHT_DATA_TOKEN not set in environment or provided")
        
        self.base_url = "https://api.brightdata.com/datasets/v3"
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
        self.logger = logging.getLogger(__name__)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((requests.RequestException, BrightDataAPIError))
    )
    def trigger_search(self, keyword: str, amazon_url: str, pages_to_search: str = "") -> str:
        """Trigger a search and return snapshot ID."""
        payload = [{
            "keyword": keyword,
            "url": amazon_url,
            "pages_to_search": pages_to_search
        }]
        
        url = f"{self.base_url}/trigger"
        params = {
            "dataset_id": BRIGHT_DATA_DATASET_ID,
            "include_errors": "true",
            "limit_multiple_results": "150"
        }
        
        try:
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            if "snapshot_id" not in data:
                raise BrightDataAPIError(f"No snapshot_id in response: {data}")
            
            return data["snapshot_id"]
            
        except requests.RequestException as e:
            raise BrightDataAPIError(f"Failed to trigger search: {str(e)}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=5),
        retry=retry_if_exception_type((requests.RequestException, BrightDataAPIError))
    )
    def check_status(self, snapshot_id: str) -> str:
        """Check the status of a snapshot."""
        url = f"{self.base_url}/progress/{snapshot_id}"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            status = data.get("status", "unknown")
            
            if status not in ["running", "ready", "failed"]:
                raise BrightDataAPIError(f"Unknown status: {status}")
            
            return status
            
        except requests.RequestException as e:
            raise BrightDataAPIError(f"Failed to check status: {str(e)}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((requests.RequestException, BrightDataAPIError))
    )
    def download_results(self, snapshot_id: str) -> List[Dict[str, Any]]:
        """Download the results from a ready snapshot."""
        url = f"{self.base_url}/snapshot/{snapshot_id}"
        params = {"format": "json"}
        
        try:
            response = requests.get(
                url,
                headers=self.headers,
                params=params,
                timeout=60
            )
            response.raise_for_status()
            
            results = response.json()
            
            if not isinstance(results, list):
                raise BrightDataAPIError(f"Expected list, got {type(results)}")
            
            return results
            
        except requests.RequestException as e:
            raise BrightDataAPIError(f"Failed to download results: {str(e)}")
        except json.JSONDecodeError as e:
            raise BrightDataAPIError(f"Failed to parse JSON response: {str(e)}")
    
    def wait_for_results(
        self, 
        snapshot_id: str, 
        max_wait_time: int = 900,  # 15min timeout balances user experience vs server load
        poll_interval: int = 15   # 15s intervals minimize API quota usage while providing feedback
    ) -> List[Dict[str, Any]]:
        """
        Wait for the snapshot to be ready and return results (synchronous).
        
        Args:
            snapshot_id: The snapshot ID to monitor
            max_wait_time: Maximum time to wait in seconds
            poll_interval: How often to check status in seconds
        """
        import time
        start_time = time.time()
        
        while True:
            try:
                status = self.check_status(snapshot_id)
                self.logger.info(f"Snapshot status: {status}")
                
                if status == "ready":
                    self.logger.info("Data processing completed, downloading results")
                    return self.download_results(snapshot_id)
                elif status == "failed":
                    raise BrightDataAPIError("Snapshot processing failed")
                elif status == "running":
                    elapsed = time.time() - start_time
                    if elapsed > max_wait_time:
                        raise BrightDataAPIError(f"Timeout after {max_wait_time} seconds. Snapshot ID: {snapshot_id}")
                    
                    self.logger.info("Processing in progress...")
                    
                    if elapsed > 120:
                        self.logger.warning("Extended processing time - servers may be experiencing high load")
                    if elapsed > 300:
                        self.logger.warning(f"High server load detected. Snapshot ID: {snapshot_id}")
                        
                    time.sleep(poll_interval)
                else:
                    raise BrightDataAPIError(f"Unexpected status: {status}")
                    
            except BrightDataAPIError:
                raise
            except Exception as e:
                raise BrightDataAPIError(f"Unexpected error while waiting: {str(e)}")
    
