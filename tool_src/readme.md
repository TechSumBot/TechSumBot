# TechSumBot Installation

## Pre-requisites

+ clone the whole project
+ implement the techsumbot summarization project
+ install streamlit (pip install)

## Usage

### Using TechSumBot Online

access to the [website](techsumbot.com)

### Using TechSumBot Locally

Run the project under 'tool_src/' folder:

```
streamlit run app.py
```

Set the local port by adding below code in the 'tool_src/.streamlit/config.toml:

```
[server]
port=a specific number, e.g., 12345 # change port number. By default streamlit uses 8501
```

Then you can access the web-based service via the link: `localhost: 12345` (suppose your port number is 12345).

Please kindly refer to the guideline page of TechSumBot website for the detailed usage.
