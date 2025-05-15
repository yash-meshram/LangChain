from langserve import RemoteRunnable

chain = RemoteRunnable("http://localhost:8000/api/c/N4XyA")
chain.invoke({ ... })