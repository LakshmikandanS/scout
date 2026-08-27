import ollama
import time
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

MODEL = "qwen3:4B"
OLLAMA_HOST = "http://127.0.0.1:11434"

PROMPT = "What is SIH? Give me a concise answer in 3 sentences."

# Keep model loaded between requests.
KEEP_ALIVE = "10m"


# ============================================================
# TEST CONFIGURATIONS
# ============================================================

TESTS = [

    # -------------------------
    # Thinking
    # -------------------------

    {
        "name": "think_false",
        "options": {},
        "think": False,
    },

    {
        "name": "think_true",
        "options": {},
        "think": True,
    },

    # -------------------------
    # Temperature
    # -------------------------

    {
        "name": "temp_0.0",
        "options": {"temperature": 0.0},
        "think": False,
    },

    {
        "name": "temp_0.2",
        "options": {"temperature": 0.2},
        "think": False,
    },

    {
        "name": "temp_0.7",
        "options": {"temperature": 0.7},
        "think": False,
    },

    # -------------------------
    # Output length
    # -------------------------

    {
        "name": "predict_32",
        "options": {"num_predict": 32},
        "think": False,
    },

    {
        "name": "predict_64",
        "options": {"num_predict": 64},
        "think": False,
    },

    {
        "name": "predict_128",
        "options": {"num_predict": 128},
        "think": False,
    },

    {
        "name": "predict_256",
        "options": {"num_predict": 256},
        "think": False,
    },

    # -------------------------
    # Top P
    # -------------------------

    {
        "name": "top_p_0.5",
        "options": {"top_p": 0.5},
        "think": False,
    },

    {
        "name": "top_p_0.9",
        "options": {"top_p": 0.9},
        "think": False,
    },

    {
        "name": "top_p_1.0",
        "options": {"top_p": 1.0},
        "think": False,
    },
]


# ============================================================
# HELPERS
# ============================================================

def ns_to_sec(value):
    """Convert Ollama nanoseconds to seconds."""
    if value is None:
        return 0.0

    return value / 1_000_000_000


def run_test(client, test):

    print("\n" + "=" * 75)
    print(f"TEST: {test['name']}")
    print("=" * 75)

    start_wall = time.perf_counter()

    try:

        response = client.chat(
            model=MODEL,

            messages=[
                {
                    "role": "user",
                    "content": PROMPT,
                }
            ],

            options=test["options"],

            think=test["think"],

            # IMPORTANT:
            # prevents model unloading between tests
            keep_alive=KEEP_ALIVE,
        )

        wall_time = time.perf_counter() - start_wall

        # ====================================================
        # Ollama native metrics
        # ====================================================

        total_time = ns_to_sec(
            getattr(response, "total_duration", 0)
        )

        load_time = ns_to_sec(
            getattr(response, "load_duration", 0)
        )

        prompt_eval_time = ns_to_sec(
            getattr(response, "prompt_eval_duration", 0)
        )

        generation_time = ns_to_sec(
            getattr(response, "eval_duration", 0)
        )

        input_tokens = getattr(
            response,
            "prompt_eval_count",
            0,
        )

        output_tokens = getattr(
            response,
            "eval_count",
            0,
        )

        # ====================================================
        # Derived metrics
        # ====================================================

        tokens_per_sec = (
            output_tokens / generation_time
            if generation_time > 0
            else 0
        )

        prompt_tokens_per_sec = (
            input_tokens / prompt_eval_time
            if prompt_eval_time > 0
            else 0
        )

        # ====================================================
        # Response
        # ====================================================

        content = response.message.content

        print(f"Wall time:              {wall_time:.3f}s")
        print(f"Ollama total:           {total_time:.3f}s")
        print(f"Model load:             {load_time:.3f}s")
        print(f"Prompt evaluation:      {prompt_eval_time:.3f}s")
        print(f"Generation:             {generation_time:.3f}s")

        print(f"Input tokens:           {input_tokens}")
        print(f"Output tokens:          {output_tokens}")

        print(f"Prompt tokens/sec:      {prompt_tokens_per_sec:.2f}")
        print(f"Generation tokens/sec:  {tokens_per_sec:.2f}")

        print("\nOUTPUT:")
        print(content)

        return {
            "test": test["name"],
            "status": "success",

            "think": test["think"],

            "temperature": test["options"].get(
                "temperature"
            ),

            "top_p": test["options"].get(
                "top_p"
            ),

            "num_predict": test["options"].get(
                "num_predict"
            ),

            "wall_time": wall_time,
            "total_time": total_time,
            "load_time": load_time,
            "prompt_eval_time": prompt_eval_time,
            "generation_time": generation_time,

            "input_tokens": input_tokens,
            "output_tokens": output_tokens,

            "prompt_tokens/sec": prompt_tokens_per_sec,
            "tokens/sec": tokens_per_sec,

            "output": content,
            "error": None,
        }

    except Exception as e:

        wall_time = time.perf_counter() - start_wall

        print(f"ERROR: {type(e).__name__}: {e}")

        return {
            "test": test["name"],
            "status": "error",

            "think": test["think"],

            "temperature": test["options"].get(
                "temperature"
            ),

            "top_p": test["options"].get(
                "top_p"
            ),

            "num_predict": test["options"].get(
                "num_predict"
            ),

            "wall_time": wall_time,
            "total_time": None,
            "load_time": None,
            "prompt_eval_time": None,
            "generation_time": None,

            "input_tokens": None,
            "output_tokens": None,

            "prompt_tokens/sec": None,
            "tokens/sec": None,

            "output": None,
            "error": str(e),
        }


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # Connect to Ollama explicitly
    # ========================================================

    print("=" * 75)
    print("OLLAMA BENCHMARK")
    print("=" * 75)

    print(f"Host:  {OLLAMA_HOST}")
    print(f"Model: {MODEL}")

    client = ollama.Client(
        host=OLLAMA_HOST
    )

    # ========================================================
    # Check connection
    # ========================================================

    print("\nChecking Ollama connection...")

    try:

        models = client.list()

        print("Ollama connection: OK")

        model_names = []

        for model in models.models:
            model_names.append(model.model)

        print("Available models:")

        for name in model_names:
            print(f"  - {name}")

        if MODEL not in model_names:
            print(
                f"\nWARNING: {MODEL} was not found "
                "in ollama list."
            )

    except Exception as e:

        print("\nFAILED TO CONNECT TO OLLAMA")
        print(e)

        print(
            "\nMake sure Ollama is running on "
            f"{OLLAMA_HOST}"
        )

        return

    # ========================================================
    # Warm up
    # ========================================================

    print("\nWarming up model...")

    warmup_start = time.perf_counter()

    try:

        client.chat(
            model=MODEL,

            messages=[
                {
                    "role": "user",
                    "content": "Say hello.",
                }
            ],

            think=False,

            options={
                "num_predict": 8,
            },

            keep_alive=KEEP_ALIVE,
        )

        warmup_time = (
            time.perf_counter() - warmup_start
        )

        print(
            f"Warmup complete: {warmup_time:.3f}s"
        )

    except Exception as e:

        print(f"Warmup failed: {e}")
        return

    # ========================================================
    # Run benchmarks
    # ========================================================

    results = []

    for test in TESTS:

        result = run_test(
            client,
            test,
        )

        results.append(result)

        # Small pause between tests
        time.sleep(0.25)

    # ========================================================
    # DataFrame
    # ========================================================

    df = pd.DataFrame(results)

    print("\n\n")
    print("=" * 120)
    print("BENCHMARK RESULTS")
    print("=" * 120)

    columns = [
        "test",
        "status",
        "think",
        "temperature",
        "top_p",
        "num_predict",

        "wall_time",
        "load_time",
        "prompt_eval_time",
        "generation_time",

        "input_tokens",
        "output_tokens",

        "prompt_tokens/sec",
        "tokens/sec",
    ]

    print(
        df[columns].to_string(
            index=False
        )
    )

    # ========================================================
    # Best configurations
    # ========================================================

    successful = df[
        df["status"] == "success"
    ].copy()

    if not successful.empty:

        print("\n")
        print("=" * 100)
        print("FASTEST CONFIGURATIONS")
        print("=" * 100)

        fastest = successful.sort_values(
            "wall_time"
        ).head(5)

        print(
            fastest[
                [
                    "test",
                    "wall_time",
                    "generation_time",
                    "output_tokens",
                    "tokens/sec",
                ]
            ].to_string(
                index=False
            )
        )

        print("\n")
        print("=" * 100)
        print("BEST GENERATION THROUGHPUT")
        print("=" * 100)

        fastest_tokens = successful.sort_values(
            "tokens/sec",
            ascending=False,
        ).head(5)

        print(
            fastest_tokens[
                [
                    "test",
                    "generation_time",
                    "output_tokens",
                    "tokens/sec",
                ]
            ].to_string(
                index=False
            )
        )

    # ========================================================
    # Save
    # ========================================================

    output_file = "ollama_benchmark.csv"

    df.to_csv(
        output_file,
        index=False,
    )

    print(
        f"\nResults saved to: {output_file}"
    )


if __name__ == "__main__":
    main()