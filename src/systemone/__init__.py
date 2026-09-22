"""systemone: Jev's /v1/systemone typed-decision API on any LLM served by vLLM.

Each question is answered by one next-token read: the prompt ends where the
answer label goes, and the label probabilities at that position are the answer.
"""

__version__ = "0.1.0"
