# LLM Adversarial Audit Framework (Project HYDRA/TAP)

**An industrial-grade, modular Python framework for automated Red Teaming and safety auditing of Large Language Models (LLMs).**

## --------Overview
This project is a comprehensive suite designed to test the robustness of LLM alignment constraints. Moving beyond simple manual jailbreaks, this framework implements state-of-the-art (SOTA) algorithmic attacks to systematically identify vulnerabilities in local models (e.g., TinyLlama, Llama-3).

It supports both **White-box** (gradient-based) and **Black-box** (logic/evolutionary) attack vectors, leveraging local hardware (CPU/GPU) via Hugging Face Transformers and Ollama.

##--------Key Features

* **Multi-Vector Attack Engine:**
    * **GCG (Greedy Coordinate Gradient):** Automated suffix optimization using gradient descent for white-box models.
    * **TAP (Tree of Attacks with Pruning):** Implements an evolutionary tree-search algorithm (branching & pruning) to bypass sophisticated filters in models like Llama-3.
    * **PAIR Logic:** Uses an attacker LLM to iteratively refine prompts based on victim refusal patterns.
* **Industrial Architecture:**
    * **Modular Design:** Separated logic for Engine, Database, and Mutators.
    * **Persistence:** Full SQL integration (SQLite) to log every iteration, prompt, and loss metric.
    * **Hybrid Inference:** Supports `torch` (Transformers) for raw tensor manipulation and `Ollama` API for high-level logic attacks.
* **Performance:** Optimized for local execution on consumer hardware (supports CUDA and CPU offloading).

## 🛠️ Installation

```bash
# Clone the repository
git clone [https://github.com/yourusername/llm-adversarial-audit.git](https://github.com/IFIR649/AIvsAI)]
cd llm-adversarial-audit

# Install dependencies
pip install torch transformers requests termcolor
