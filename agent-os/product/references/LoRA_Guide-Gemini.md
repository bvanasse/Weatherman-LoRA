# LoRA Training Plan: The TwainBot

## TL;DR

This guide outlines a 3-day weekend plan to fine-tune a small language model using LoRA. The goal is to create "TwainBot," an AI with the wit of Mark Twain and the ability to use a weather API.

*   **Day 1: Data Domination.** Collect and process humorous texts from "The Onion" and public domain authors (Twain, Franklin). Synthetically generate a dataset for weather API tool usage.
*   **Day 2: Training Time.** Choose a base model (e.g., `Phi-3-mini-instruct`, `Mistral-7B-Instruct`), set up a cloud GPU environment, and launch the LoRA fine-tuning process using the prepared dataset.
*   **Day 3: Evaluation & Victory.** Test the model qualitatively on its persona and tool-use capabilities, merge the LoRA weights, and save the final, specialized model.

---

## 1. Overview

### 1.1. Project Goal

The primary objective is to efficiently fine-tune a small language model (SLM) to adopt a specific, humorous persona while also being proficient at a specific task: using a tool (a weather API). This project serves as a practical experiment in data curation, demonstrating how a high-quality, specialized dataset can dramatically shape a model's behavior with minimal training.

### 1.2. Core Concepts

*   **LoRA (Low-Rank Adaptation):** We will use LoRA to drastically reduce the number of trainable parameters. Instead of fine-tuning the entire model (billions of parameters), we will train small "adapter" matrices, making the process faster, cheaper, and more memory-efficient.
*   **QLoRA (Quantized LoRA):** To make this feasible on consumer or free-tier cloud GPUs, we'll use QLoRA, which involves training the LoRA adapters on a quantized (4-bit) version of the base model.
*   **Data is King:** The success of this project hinges almost entirely on the quality and composition of the training data. The plan heavily emphasizes data collection, cleaning, and generation.

---

## 2. Phased Training Plan (3-Day Weekend)

### Phase 1: Data Collection & Preparation (Day 1)

**Objective:** Curate and format two distinct datasets: one for the desired persona and one for the tool-use skill.

#### Part A: Personality & Style Dataset

This dataset will teach the model *how* to talk.

1.  **Source: The Onion & Satirical News**
    *   **Existing Data:** Your workspace already contains `data_sources/reddit-theonion/data/TheOnion_181217_184244.csv`. This is an excellent starting point. Use a script (e.g., in a Jupyter Notebook) to load this CSV, inspect its contents (likely headlines or article bodies), and clean it. Remove any HTML, metadata, or artifacts.
    *   **Web Scraping (Optional):** To get more recent data, you can write a simple web scraper.
        *   **Tools:** Python libraries like `BeautifulSoup` and `requests`.
        *   **Caution:** Always check a website's `robots.txt` and Terms of Service before scraping. Scrape slowly and respectfully to avoid being blocked.

2.  **Source: Public Domain Authors (Twain, Franklin)**
    *   **Resource:** [Project Gutenberg](https://www.gutenberg.org/) is the best source for public domain literature.
    *   **Process:**
        1.  Download the complete works of Mark Twain and Benjamin Franklin in plain text (`.txt`) format.
        2.  Write a script to parse these text files. The goal is to break the books down into smaller, coherent chunks. Paragraphs or short sections (e.g., 200-500 words) are ideal. This prevents the model from losing context during training.
        3.  Filter out chapter headings, tables of contents, and any Gutenberg-specific headers/footers.

#### Part B: Tool-Use Dataset (Synthetic Generation)

This dataset will teach the model *what* to do. We will synthetically create examples of how to use a weather API.

1.  **Define the Tool:** First, define a clear, simple function signature for your tool.
    ```python
    # Example Weather API tool definition
    def get_weather(location: str, date: str = "today") -> dict:
        """Gets the weather for a given location and date."""
        # In a real scenario, this would call an API.
        # For data generation, we can use mock data.
        if "san francisco" in location.lower():
            return {"temperature": "65°F", "condition": "sunny with a chance of existential dread"}
        else:
            return {"temperature": "72°F", "condition": "pleasantly unremarkable"}
    ```

2.  **Generate Conversational Examples:** The most effective format for tool-tuning is conversational, showing the model's "thought process." Use a powerful LLM (like Gemini, Claude, or GPT-4) with a few-shot prompt to generate a few hundred examples in this format.

    **Few-shot Prompt Example:**
    ```
    Create a conversational training example for a helpful, witty AI assistant that uses the `get_weather` tool.

    **User:** What's the weather like in SF tomorrow?
    **AI Thought:** The user is asking about the weather in "SF" for "tomorrow". I should call the `get_weather` tool with `location="San Francisco"` and `date="tomorrow"`.
    **AI Tool Call:** `get_weather(location="San Francisco", date="tomorrow")`
    **Tool Response:** `{"temperature": "68°F", "condition": "foggy until noon, then glorious sun"}`
    **AI Final Answer:** I've consulted the meteorological spirits, and they tell me San Francisco will be wrapped in its signature fog tomorrow morning. But fear not, for the sun shall triumph by noon, bringing a glorious 68-degree day.

    ---

    **User:** Is it going to be hot in Death Valley next week?
    **AI Thought:** The user wants to know the weather in "Death Valley" for "next week". I need to call the `get_weather` tool.
    **AI Tool Call:** `get_weather(location="Death Valley", date="next week")`
    **Tool Response:** `{"temperature": "115°F", "condition": "scorching sun"}`
    **AI Final Answer:** My dear friend, asking if it will be hot in Death Valley is like asking if water is wet. You can expect a temperature of 115°F under a sun that laughs at the very concept of mercy.
    ```

#### Part C: Final Data Formatting

1.  **Combine and Clean:** Merge the personality and tool-use datasets. Remove duplicates and perform a final cleaning pass.
2.  **Structure:** Format the entire dataset into a single file, preferably in JSON Lines (`.jsonl`) format, where each line is a JSON object. A simple format is:
    ```json
    {"text": "Your formatted conversational example or text chunk here..."}
    ```
3.  **Split:** Shuffle the dataset randomly and split it into a training set and a smaller validation set (e.g., a 95/5 or 90/10 split). This is crucial for monitoring overfitting.

### Phase 2: Model Setup & Training (Day 2)

**Objective:** Configure the environment and launch the fine-tuning job.

1.  **Choose a Base Model:**
    *   Select a small but capable instruction-tuned model. Excellent candidates include:
        *   `microsoft/Phi-3-mini-4k-instruct` (Great performance at a small size)
        *   `mistralai/Mistral-7B-Instruct-v0.2` (A very strong 7B parameter model)
        *   `meta-llama/Llama-3-8B-Instruct` (State-of-the-art for its size)

2.  **Set Up the Environment:**
    *   Use a cloud GPU provider. Google Colab (with a Pro subscription) or Kaggle Kernels offer free T4/P100 GPUs. For more power, consider services like Lambda Labs, RunPod, or Vast.ai.
    *   Install the necessary Python libraries.
        ```bash
        pip install torch transformers datasets peft accelerate bitsandbytes trl
        ```

3.  **Create the Training Script:**
    *   Use the Hugging Face `SFTTrainer` from the `trl` library, which simplifies the entire process.
    *   Your script will need to:
        a.  Load the base model in 4-bit precision (`BitsAndBytesConfig`).
        b.  Load your training and validation datasets.
        c.  Configure the LoRA parameters (`LoraConfig`).
        d.  Define the training arguments (`TrainingArguments`).
        e.  Initialize and run the `SFTTrainer`.

    **Key `LoraConfig` Parameters:**
    *   `r` (rank): The dimensionality of the LoRA matrices. A good starting point is `16` or `32`.
    *   `lora_alpha`: The scaling factor. A common practice is to set this to `2 * r` (e.g., `32` or `64`).
    *   `target_modules`: The model layers to apply LoRA to. For most Transformer models, this includes the attention projections: `["q_proj", "k_proj", "v_proj", "o_proj"]`.
    *   `lora_dropout`: A value like `0.05` or `0.1` can help prevent overfitting.

4.  **Launch and Monitor:**
    *   Run your training script.
    *   (Optional but recommended) Log metrics to a service like Weights & Biases to track training and validation loss in real-time. If validation loss starts to increase while training loss decreases, your model is overfitting.

### Phase 3: Evaluation & Inference (Day 3)

**Objective:** Test the new model and prepare it for use.

1.  **Qualitative Testing:**
    *   Write an inference script to load the base model and apply your trained LoRA adapter weights on top of it.
    *   Engage with your model. Test it against your goals:
        *   **Persona Test:** "Rewrite 'The product roadmap is strategically aligned with our quarterly KPIs' in the style of Mark Twain."
        *   **Joke Test:** "Tell me a joke about bureaucracy."
        *   **Tool-Use Test:** "What's the weather forecast for London this weekend?"
        *   **Combined Test:** "In your wittiest voice, tell me if I'll need a coat in New York on Tuesday."

2.  **Merge and Save the Final Model:**
    *   For easy deployment, merge the LoRA adapter weights directly into the base model's weights. The `peft` library makes this simple.
        ```python
        from peft import PeftModel

        # Load the base model and the LoRA adapter
        base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        lora_model = PeftModel.from_pretrained(base_model, "path/to/your/lora_adapter")

        # Merge the weights and save the final model
        merged_model = lora_model.merge_and_unload()
        merged_model.save_pretrained("TwainBot-v1-merged")
        ```
    *   You now have a standalone, fine-tuned model ready for deployment or sharing on the Hugging Face Hub.

---

## 3. Resources and Further Reading

*   **Libraries:**
    *   [Hugging Face PEFT (Parameter-Efficient Fine-Tuning)](https://huggingface.co/docs/peft/index)
    *   [Hugging Face TRL (Transformer Reinforcement Learning)](https://huggingface.co/docs/trl/index)
    *   [bitsandbytes for Quantization](https://github.com/TimDettmers/bitsandbytes)
*   **Guides & Articles:**
    *   [Hugging Face Blog: "Fine-tune a Mistral-7b model with DPO"](https://huggingface.co/blog/dpo-fine-tune) (Excellent, up-to-date guide using TRL and PEFT).
    *   [Sebastian Raschka's Blog on LLM Training](https://magazine.sebastianraschka.com/) (Deeply technical and accurate resources).
*   **Data Sources:**
    *   [Project Gutenberg](https://www.gutenberg.org/)
    *   [Hugging Face Datasets](https://huggingface.co/datasets)
*   **APIs for Tool-Use:**
    *   [OpenWeatherMap API](https://openweathermap.org/api)
