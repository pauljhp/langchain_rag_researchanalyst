# Langchain Research Assistant

API for investment research automation, built with langchain, langgraph and Azure OpenAI. 

## Overview

The source code includes 5 modules. The main UI is through the api, run with `uvicorn main:app --host=<host> --port=<port>`

### api:

This module is directly imported by main.py and contains 3 key functions:

- Automated data ingestion (`data_ingestion.py`);
- Automated information retrieval from both data ingested into the back-end, and searching on internet/available APIs;
- Task automation, which includes report-writing (implemented), charting (not fully implemented), and more

### agents:

This is where the key chains and graphs are implemented. the `api` module imports from this module. Implement new functionalities with agents on this graph

### tools:

This is where agents get their key tools. Some agent-specific tools are implemented in the agent modules, but shared tools should be implemented here.

### drivers:

The drivers to databases should be implemented here. Currently only Chromadb is implemented. 

### utils:

Shared utilitity functions and constructs, such as key data structures.
