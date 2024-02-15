# Chain Runner

Tooling for Multithreaded and Async Langchain operations.

# Quick start

- install Python 3.11 (if not installed)
    ```shell
    brew install python@3.11
    ```
- install Poetry (if not installed)
    ```shell
    brew install poetry
    ```
- add a local `.env` file to project root (will NOT be pushed to Gitlab)
- add your openai api key
    ```shell
    OPENAI_API_KEY=...
    ```
- install dependencies
  ```shell
  poetry lock
  poetry install
  ```
  
# Structure
 
`/dist` final output artifacts  
`/src` project code

## /src

[processing.py](src/chain_runner/processing.py) - utils for parallel processing and batching
[openai.py](src/chain_runner/openai.py) [DEPRECATED] - raw openai client utils for parallel processing

## Build and Release

```shell
poetry lock; poetry build
```

## Release Process

TBD
