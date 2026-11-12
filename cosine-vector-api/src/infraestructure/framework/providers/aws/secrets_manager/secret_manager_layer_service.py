"""
AWS Secrets Manager Lambda Layer Service.

Implementation using AWS Parameters and Secrets Lambda Extension HTTP endpoint.
"""
import json
import os
from typing import Any, Dict

import requests
from reactivex import Observable, of, throw

from .secret_manager_interface import ISecretManager


class SecretManagerLayerService(ISecretManager):
    """
    AWS Secrets Manager implementation using Lambda extension HTTP endpoint.
    
    The AWS Parameters and Secrets Lambda Extension provides a local HTTP 
    endpoint for fetching secrets with caching, reducing latency and costs.
    
    Best suited for:
    - AWS Lambda functions with the extension layer installed
    """
    
    def __init__(self, port: int = 2773):
        """
        Initialize the Lambda extension client.
        
        Args:
            port: Local port for the Lambda extension. Default is 2773.
        """
        self.port = port
        self.endpoint = f"http://localhost:{port}/secretsmanager/get"
        self.session_token = os.getenv("AWS_SESSION_TOKEN", "")
    
    def get_secret(self, secret_name: str) -> Observable[str]:
        """
        Fetch a secret from AWS Secrets Manager via Lambda extension.
        
        Args:
            secret_name: The name or ARN of the secret.
            
        Returns:
            Observable emitting the secret string value.
        """
        def _fetch_secret() -> Observable[str]:
            try:
                headers = {
                    "X-Aws-Parameters-Secrets-Token": self.session_token
                }
                params = {
                    "secretId": secret_name
                }
                
                response = requests.get(
                    self.endpoint,
                    headers=headers,
                    params=params,
                    timeout=5
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return of(data.get("SecretString", ""))
                elif response.status_code == 404:
                    return throw(ValueError(f"Secret '{secret_name}' not found"))
                elif response.status_code == 403:
                    return throw(PermissionError(f"Access denied to secret '{secret_name}'"))
                else:
                    return throw(RuntimeError(
                        f"Failed to fetch secret: HTTP {response.status_code}"
                    ))
                    
            except requests.exceptions.ConnectionError:
                return throw(ConnectionError(
                    "Lambda extension not available. Is the layer installed?"
                ))
            except Exception as e:
                return throw(e)
        
        return _fetch_secret()
    
    def get_secret_as_dict(self, secret_name: str) -> Observable[Dict[str, Any]]:
        """
        Fetch and parse a JSON secret.
        
        Args:
            secret_name: The name or ARN of the secret.
            
        Returns:
            Observable emitting the parsed JSON as a dictionary.
        """
        def parse_json(secret_string: str) -> Dict[str, Any]:
            return json.loads(secret_string)
        
        return self.get_secret(secret_name).pipe(
            lambda obs: of(parse_json(obs)) if isinstance(obs, str) else obs
        )
