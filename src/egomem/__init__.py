"""Episodic memory assistant: egocentric video to a textual diary that answers questions.

The pipeline follows Lando, Forte and Furnari (arXiv:2602.22455):

    video ──▶ clips ──▶ [describer · VLM] ──▶ textual memory (timestamped diary)
                                                        │
                           question ──▶ [reader · LLM] ◀┘──▶ answer + cited moment
"""
