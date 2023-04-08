# Source: https://llamahub.ai/l/github_repo

import argparse
from llama_index import GPTSimpleVectorIndex, download_loader, node_parser
from llama_index.readers.llamahub_modules.github_repo import GithubClient, GithubRepositoryReader
import os
import pickle

# Create ArgumentParser & parse
parser = argparse.ArgumentParser(description="Query the @onflow/flow-nft repo docs")
parser.add_argument("--query", help="What would you like to know about the NFT standard?")
args = parser.parse_args()

# Make sure we have an OPENAI API key
assert (
    os.environ["OPENAI_API_KEY"] is not None
), "Please set the OPENAI_API_KEY environment variable."

# Create the document loader
download_loader("GithubRepositoryReader")

# Check if docs have been pickled
docs = None
if os.path.exists("docs.pkl"):
    with open("docs.pkl", "rb") as f:
        docs = pickle.load(f)

# Get the docs if they're not already stored
if docs is None:
    github_client = GithubClient(os.environ["GITHUB_TOKEN"])
    loader = GithubRepositoryReader(
        github_client,
        owner =                  "onflow",
        repo =                   "flow-nft",
        filter_directories =     (["docs"], GithubRepositoryReader.FilterType.INCLUDE),
        filter_file_extensions = ([".md"], GithubRepositoryReader.FilterType.INCLUDE),
        verbose =                True,
        concurrent_requests =    10,
    )

    docs = loader.load_data(branch="main")
    # docs = loader.load_data(commit_sha="65b538b26207392592d397e4d6c1f08119006efd")

    with open("docs.pkl", "wb") as f:
        pickle.dump(docs, f)

# Index the docs
parser = node_parser.SimpleNodeParser()
docs_as_nodes = parser.get_nodes_from_documents(documents=docs)
index = GPTSimpleVectorIndex(nodes=docs_as_nodes)

# Print the query results
print(index.query(args.query))