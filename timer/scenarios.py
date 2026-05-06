"""
Timing scenarios: combinations of model, context type, context size, and output requirement.
"""

SHORT_TEXT = "The quick brown fox jumps over the lazy dog."

MEDIUM_TEXT = " ".join(
    [
        "Artificial intelligence has transformed the way we interact with technology.",
        "Machine learning models can now perform tasks that once required human intelligence.",
        "Natural language processing allows computers to understand and generate human text.",
        "Deep learning architectures have enabled breakthroughs in image recognition and speech.",
        "Reinforcement learning allows agents to learn optimal behaviour through trial and error.",
        "Transfer learning lets pre-trained models be fine-tuned for specific downstream tasks.",
        "Attention mechanisms in transformers revolutionised sequence modelling tasks.",
        "Large language models are trained on vast corpora of text from the internet.",
        "Prompt engineering is the art of crafting inputs that elicit desired model outputs.",
        "Retrieval-augmented generation combines parametric and non-parametric knowledge.",
    ]
    * 5
)

LONG_TEXT = " ".join(
    [
        "In the rapidly evolving landscape of artificial intelligence, large language models have "
        "emerged as a transformative technology with far-reaching implications across virtually every "
        "domain of human activity. These models, trained on vast corpora of text data encompassing "
        "billions of words from books, websites, scientific papers, and code repositories, have "
        "demonstrated remarkable capabilities in understanding context, generating coherent prose, "
        "reasoning through complex problems, and adapting to diverse tasks with minimal additional "
        "guidance. The architecture underlying these systems, based on the transformer model "
        "introduced by Vaswani et al. in 2017, relies on self-attention mechanisms that allow the "
        "model to weigh the relevance of different tokens when producing each output. Training such "
        "models requires enormous computational resources, typically thousands of specialised "
        "accelerators running for weeks or months, consuming energy equivalent to the lifetime "
        "emissions of several automobiles. Despite these costs, the resulting systems exhibit "
        "emergent capabilities that were not explicitly trained for, including multi-step reasoning, "
        "analogy-making, and rudimentary common-sense inference. The deployment of these models "
        "raises profound questions about intellectual property, bias amplification, environmental "
        "impact, economic displacement, and the nature of intelligence itself."
    ]
    * 10
)

CODE_CONTEXT = (
    """
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.order = []

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.order.remove(key)
        self.order.append(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            oldest = self.order.pop(0)
            del self.cache[oldest]
        self.cache[key] = value
        self.order.append(key)
"""
    * 3
)

SYSTEM_PROMPT_SHORT = "You are a helpful assistant. Answer concisely."

SYSTEM_PROMPT_LONG = (
    "You are an expert software engineer, technical writer, and educator with 20 years of "
    "experience across systems programming, distributed systems, machine learning infrastructure, "
    "and developer tooling. You provide precise, accurate, and well-reasoned answers. You always "
    "cite trade-offs, acknowledge uncertainty, and tailor your explanations to the technical level "
    "implied by the question. You prefer code examples over abstract descriptions. You never "
    "hallucinate library APIs or invent facts. When asked to review code you are thorough, "
    "constructive, and specific. When asked to write code you produce idiomatic, well-commented, "
    "production-quality output. You follow PEP 8 for Python, standard Go idioms for Go, and "
    "established conventions for whatever language is in use."
)

SCENARIOS = [
    # --- tiny prompt, short answer ---
    {
        "name": "tiny_prompt_short_answer_haiku",
        "model": "fast",
        "prompt": "What is 2 + 2?",
        "system": None,
        "max_tokens": 32,
        "tags": ["tiny", "no_system"],
    },
    {
        "name": "tiny_prompt_short_answer_sonnet",
        "model": "balanced",
        "prompt": "What is 2 + 2?",
        "system": None,
        "max_tokens": 32,
        "tags": ["tiny", "no_system"],
    },
    {
        "name": "tiny_prompt_short_answer_opus",
        "model": "best",
        "prompt": "What is 2 + 2?",
        "system": None,
        "max_tokens": 32,
        "tags": ["tiny", "no_system"],
    },
    # --- short context, short answer ---
    {
        "name": "short_ctx_short_answer_haiku",
        "model": "fast",
        "prompt": f"Summarise this in one sentence: {SHORT_TEXT}",
        "system": SYSTEM_PROMPT_SHORT,
        "max_tokens": 64,
        "tags": ["short_ctx", "short_answer"],
    },
    {
        "name": "short_ctx_short_answer_sonnet",
        "model": "balanced",
        "prompt": f"Summarise this in one sentence: {SHORT_TEXT}",
        "system": SYSTEM_PROMPT_SHORT,
        "max_tokens": 64,
        "tags": ["short_ctx", "short_answer"],
    },
    {
        "name": "short_ctx_short_answer_opus",
        "model": "best",
        "prompt": f"Summarise this in one sentence: {SHORT_TEXT}",
        "system": SYSTEM_PROMPT_SHORT,
        "max_tokens": 64,
        "tags": ["short_ctx", "short_answer"],
    },
    # --- medium context, medium answer ---
    {
        "name": "medium_ctx_medium_answer_haiku",
        "model": "fast",
        "prompt": f"Summarise the following passage in 3-4 sentences:\n\n{MEDIUM_TEXT}",
        "system": SYSTEM_PROMPT_SHORT,
        "max_tokens": 256,
        "tags": ["medium_ctx", "medium_answer"],
    },
    {
        "name": "medium_ctx_medium_answer_sonnet",
        "model": "balanced",
        "prompt": f"Summarise the following passage in 3-4 sentences:\n\n{MEDIUM_TEXT}",
        "system": SYSTEM_PROMPT_SHORT,
        "max_tokens": 256,
        "tags": ["medium_ctx", "medium_answer"],
    },
    {
        "name": "medium_ctx_medium_answer_opus",
        "model": "best",
        "prompt": f"Summarise the following passage in 3-4 sentences:\n\n{MEDIUM_TEXT}",
        "system": SYSTEM_PROMPT_SHORT,
        "max_tokens": 256,
        "tags": ["medium_ctx", "medium_answer"],
    },
    # --- long context, medium answer ---
    {
        "name": "long_ctx_medium_answer_haiku",
        "model": "fast",
        "prompt": f"Extract the three most important themes from this text:\n\n{LONG_TEXT}",
        "system": SYSTEM_PROMPT_LONG,
        "max_tokens": 512,
        "tags": ["long_ctx", "medium_answer", "long_system"],
    },
    {
        "name": "long_ctx_medium_answer_sonnet",
        "model": "balanced",
        "prompt": f"Extract the three most important themes from this text:\n\n{LONG_TEXT}",
        "system": SYSTEM_PROMPT_LONG,
        "max_tokens": 512,
        "tags": ["long_ctx", "medium_answer", "long_system"],
    },
    {
        "name": "long_ctx_medium_answer_opus",
        "model": "best",
        "prompt": f"Extract the three most important themes from this text:\n\n{LONG_TEXT}",
        "system": SYSTEM_PROMPT_LONG,
        "max_tokens": 512,
        "tags": ["long_ctx", "medium_answer", "long_system"],
    },
    # --- code context, code generation ---
    {
        "name": "code_ctx_code_gen_haiku",
        "model": "fast",
        "prompt": (
            f"Given this Python code:\n\n{CODE_CONTEXT}\n\n"
            "Add a `quick_sort` function following the same style."
        ),
        "system": SYSTEM_PROMPT_LONG,
        "max_tokens": 512,
        "tags": ["code_ctx", "code_gen", "long_system"],
    },
    {
        "name": "code_ctx_code_gen_sonnet",
        "model": "balanced",
        "prompt": (
            f"Given this Python code:\n\n{CODE_CONTEXT}\n\n"
            "Add a `quick_sort` function following the same style."
        ),
        "system": SYSTEM_PROMPT_LONG,
        "max_tokens": 512,
        "tags": ["code_ctx", "code_gen", "long_system"],
    },
    {
        "name": "code_ctx_code_gen_opus",
        "model": "best",
        "prompt": (
            f"Given this Python code:\n\n{CODE_CONTEXT}\n\n"
            "Add a `quick_sort` function following the same style."
        ),
        "system": SYSTEM_PROMPT_LONG,
        "max_tokens": 512,
        "tags": ["code_ctx", "code_gen", "long_system"],
    },
    # --- long output requirement ---
    {
        "name": "long_output_haiku",
        "model": "fast",
        "prompt": "Write a detailed 500-word essay on the history of the internet.",
        "system": SYSTEM_PROMPT_SHORT,
        "max_tokens": 1024,
        "tags": ["long_output"],
    },
    {
        "name": "long_output_sonnet",
        "model": "balanced",
        "prompt": "Write a detailed 500-word essay on the history of the internet.",
        "system": SYSTEM_PROMPT_SHORT,
        "max_tokens": 1024,
        "tags": ["long_output"],
    },
    {
        "name": "long_output_opus",
        "model": "best",
        "prompt": "Write a detailed 500-word essay on the history of the internet.",
        "system": SYSTEM_PROMPT_SHORT,
        "max_tokens": 1024,
        "tags": ["long_output"],
    },
]
