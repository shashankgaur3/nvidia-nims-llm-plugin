# NVIDIA NIM LLM Plugin 

With this plugin, you can leverage LLMs hosted as NIM Microservice as part of Dataiku LLM Mesh

# Capabilities

- NIM hosted chat completion models in Dataiku Prompt Studios, LLM powered recipes, and via the LLM Mesh python/REST APIs
- NIM hosted embedding endpoint use in Dataiku Embed recipe for Retrieval Augmented Generation (RAG), and via the LLM Mesh python/REST APIs
- Custom Chat and Embed models hosted on NIM

# Limitations

- Must use Dataiku >= v12.5.2

# Setup

## Install Plugin

1. Install the plugin - Go to Plugins -> Add Plugin -> Fetch from Git repository, then enter this repo URL
2. In the plugin settings, add a “NIM API Key” preset (per user credential)
3. Go to your Profile -> Credentials --> Look for the Preset name under Plugin Credentials and add youu API key


